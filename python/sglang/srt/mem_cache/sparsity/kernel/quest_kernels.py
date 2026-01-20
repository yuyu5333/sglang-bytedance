import torch
import triton
import triton.language as tl


@triton.jit
def quest_compute_page_reps_kernel(
    k_buf_ptr,
    req_to_token_ptr,
    reqs_ptr,
    seq_lens_ptr,
    start_page_ptr,
    end_page_ptr,
    page_k_min_ptr,
    page_k_max_ptr,
    page_valid_ptr,
    k_buf_s_token: tl.constexpr,
    k_buf_s_head: tl.constexpr,
    k_buf_s_dim: tl.constexpr,
    req_to_token_s_req: tl.constexpr,
    req_to_token_s_tok: tl.constexpr,
    page_k_s_page: tl.constexpr,
    page_k_s_head: tl.constexpr,
    page_k_s_dim: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_REQ_TOKENS: tl.constexpr,
    MAX_K_TOKENS: tl.constexpr,
    MAX_PAGE_STORAGE: tl.constexpr,
    HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_pair = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)
    req_local = pid_pair // MAX_PAGES
    page_rel = pid_pair - req_local * MAX_PAGES
    req_idx = tl.load(reqs_ptr + req_local).to(tl.int32)
    seq_len = tl.load(seq_lens_ptr + req_local).to(tl.int32)
    start_page = tl.load(start_page_ptr + req_local).to(tl.int32)
    end_page = tl.load(end_page_ptr + req_local).to(tl.int32)
    page_id = start_page + page_rel
    page_in_range = page_id < end_page

    head_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    valid_head = head_off < HEAD_NUM
    valid_dim = dim_off < HEAD_DIM
    mask_hd = valid_head[:, None] & valid_dim[None, :]
    min_val = tl.full((BLOCK_H, BLOCK_D), float("inf"), tl.float32)
    max_val = tl.full((BLOCK_H, BLOCK_D), float("-inf"), tl.float32)

    tok_base = page_id * PAGE_SIZE
    tok_base_clamped = tl.maximum(0, tl.minimum(tok_base, MAX_REQ_TOKENS - 1))
    phys_start_tok = tl.load(
        req_to_token_ptr + req_idx * req_to_token_s_req + tok_base_clamped * req_to_token_s_tok
    ).to(tl.int32)
    phys_start_tok = tl.maximum(0, tl.minimum(phys_start_tok, MAX_K_TOKENS - 1))
    phys_page = phys_start_tok // PAGE_SIZE
    phys_page = tl.maximum(0, tl.minimum(phys_page, MAX_PAGE_STORAGE - 1))

    for j in range(PAGE_SIZE):
        tok_pos = tok_base + j
        tok_in_seq = tok_pos < seq_len
        tok_valid = page_in_range & tok_in_seq
        tok_pos_clamped = tl.maximum(0, tl.minimum(tok_pos, MAX_REQ_TOKENS - 1))
        phys_tok = tl.load(
            req_to_token_ptr + req_idx * req_to_token_s_req + tok_pos_clamped * req_to_token_s_tok
        ).to(tl.int32)
        phys_tok = tl.maximum(0, tl.minimum(phys_tok, MAX_K_TOKENS - 1))

        base = (
            k_buf_ptr
            + phys_tok * k_buf_s_token
            + head_off[:, None] * k_buf_s_head
            + dim_off[None, :] * k_buf_s_dim
        )
        vals = tl.load(base, mask=mask_hd & tok_valid, other=0.0).to(tl.float32)
        min_val = tl.minimum(min_val, tl.where(mask_hd & tok_valid, vals, min_val))
        max_val = tl.maximum(max_val, tl.where(mask_hd & tok_valid, vals, max_val))

    base_min = (
        page_k_min_ptr
        + phys_page * page_k_s_page
        + head_off[:, None] * page_k_s_head
        + dim_off[None, :] * page_k_s_dim
    )
    base_max = (
        page_k_max_ptr
        + phys_page * page_k_s_page
        + head_off[:, None] * page_k_s_head
        + dim_off[None, :] * page_k_s_dim
    )
    tl.store(base_min, min_val, mask=mask_hd & page_in_range)
    tl.store(base_max, max_val, mask=mask_hd & page_in_range)
    if (pid_h == 0) & (pid_d == 0):
        tl.store(page_valid_ptr + phys_page, tl.full((), 1, tl.int1), mask=page_in_range)


@triton.jit
def quest_update_states_kernel(
    success_indices_ptr,
    repr_constructed_ptr,
    last_constructed_ptr,
    num_pages_ptr,
):
    bid = tl.program_id(0)
    idx = tl.load(success_indices_ptr + bid)
    tl.store(repr_constructed_ptr + idx, tl.full((), 1, tl.int1))
    val = tl.load(num_pages_ptr + bid)
    tl.store(last_constructed_ptr + idx, val)


@triton.jit
def quest_update_last_kernel(
    success_indices_ptr,
    last_constructed_ptr,
    end_pages_ptr,
):
    bid = tl.program_id(0)
    idx = tl.load(success_indices_ptr + bid)
    val = tl.load(end_pages_ptr + bid)
    tl.store(last_constructed_ptr + idx, val)


def launch_compute_page_reps(
    k_buffer: torch.Tensor,
    req_to_token: torch.Tensor,
    reqs: torch.Tensor,
    seq_lens: torch.Tensor,
    start_page: torch.Tensor,
    end_page: torch.Tensor,
    page_k_min: torch.Tensor,
    page_k_max: torch.Tensor,
    page_valid: torch.Tensor,
    page_size: int,
):
    n = int(reqs.numel())
    if n == 0:
        return
    max_pages = int((end_page - start_page).max().item())
    if max_pages <= 0:
        return

    H = int(k_buffer.shape[1])
    D = int(k_buffer.shape[2])
    BLOCK_H = 8
    BLOCK_D = 64 if D >= 64 else 32
    grid = (
        n * max_pages,
        triton.cdiv(H, BLOCK_H),
        triton.cdiv(D, BLOCK_D),
    )
    quest_compute_page_reps_kernel[grid](
        k_buffer,
        req_to_token,
        reqs,
        seq_lens,
        start_page,
        end_page,
        page_k_min,
        page_k_max,
        page_valid,
        k_buffer.stride(0),
        k_buffer.stride(1),
        k_buffer.stride(2),
        req_to_token.stride(0),
        req_to_token.stride(1),
        page_k_min.stride(0),
        page_k_min.stride(1),
        page_k_min.stride(2),
        MAX_PAGES=max_pages,
        PAGE_SIZE=page_size,
        MAX_REQ_TOKENS=req_to_token.shape[1],
        MAX_K_TOKENS=k_buffer.shape[0],
        MAX_PAGE_STORAGE=page_k_min.shape[0],
        HEAD_NUM=H,
        HEAD_DIM=D,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )


def launch_update_states(success_indices, repr_constructed, last_constructed_page, num_pages_vals):
    grid = (success_indices.shape[0],)
    quest_update_states_kernel[grid](
        success_indices,
        repr_constructed,
        last_constructed_page,
        num_pages_vals,
    )


def launch_update_last_constructed(success_indices, last_constructed_page, end_pages_vals):
    grid = (success_indices.shape[0],)
    quest_update_last_kernel[grid](
        success_indices,
        last_constructed_page,
        end_pages_vals,
    )
