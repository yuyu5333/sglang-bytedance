
import torch
import triton
import triton.language as tl

@triton.jit
def quest_page_rep_kernel(
    page_k_min_ptr,
    page_k_max_ptr,
    page_valid_ptr,
    reqs_ptr,
    seq_lens_ptr,
    start_page_ptr,
    end_page_ptr,
    req_to_token_ptr,
    k_buffer_ptr,
    # Strides
    req_to_token_stride_req,
    req_to_token_stride_token,
    k_buffer_stride_token,
    k_buffer_stride_head,
    k_buffer_stride_dim,
    page_k_stride_page,
    page_k_stride_head,
    page_k_stride_dim,
    # Shapes
    req_to_token_num_tokens, # To clamp
    k_buffer_num_tokens,     # To clamp
    # Constants
    PAGE_SIZE: tl.constexpr,
    HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr
):
    req_idx = tl.program_id(0)
    page_rel_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    
    # Load request info
    req_id = tl.load(reqs_ptr + req_idx)
    s_page = tl.load(start_page_ptr + req_idx)
    e_page = tl.load(end_page_ptr + req_idx)
    
    current_page = s_page + page_rel_idx
    
    if current_page >= e_page:
        return
        
    seq_len = tl.load(seq_lens_ptr + req_idx)
    logical_token_start = current_page * PAGE_SIZE
    
    # Get physical page index from the first token of the page
    # Clamp logical_token_start to be safe for req_to_token lookup
    # logic from python: tok_start.clamp(0, req_to_token.shape[1] - 1)
    
    safe_log_tok_start = tl.minimum(logical_token_start, req_to_token_num_tokens - 1)
        
    offset_req_tok = req_id * req_to_token_stride_req + safe_log_tok_start * req_to_token_stride_token
    first_phys_tok = tl.load(req_to_token_ptr + offset_req_tok)
    phys_page_idx = first_phys_tok // PAGE_SIZE
    
    dim_offsets = tl.arange(0, BLOCK_DIM)
    dim_mask = dim_offsets < HEAD_DIM
    
    # Initialize accumulators
    min_vals = tl.full([BLOCK_DIM], float("inf"), dtype=tl.float32)
    max_vals = tl.full([BLOCK_DIM], float("-inf"), dtype=tl.float32)
    
    # Loop over tokens in the page
    for i in range(PAGE_SIZE):
        log_tok_idx = logical_token_start + i
        
        if log_tok_idx < seq_len:
            # Clamp log_tok_idx for req_to_token lookup
            safe_log_tok_idx = tl.minimum(log_tok_idx, req_to_token_num_tokens - 1)
            
            offset_rt = req_id * req_to_token_stride_req + safe_log_tok_idx * req_to_token_stride_token
            phys_tok = tl.load(req_to_token_ptr + offset_rt)
            
            # Clamp phys_tok for k_buffer lookup
            phys_tok = tl.minimum(phys_tok, k_buffer_num_tokens - 1)
            phys_tok = tl.maximum(phys_tok, 0)
            
            # Load key vector
            k_ptr_base = phys_tok * k_buffer_stride_token + head_idx * k_buffer_stride_head
            k_ptrs = k_ptr_base + dim_offsets * k_buffer_stride_dim
            
            keys = tl.load(k_buffer_ptr + k_ptrs, mask=dim_mask, other=0.0).to(tl.float32)
            
            min_vals = tl.minimum(min_vals, keys)
            max_vals = tl.maximum(max_vals, keys)
            
    # Store results
    out_ptr_base = phys_page_idx * page_k_stride_page + head_idx * page_k_stride_head
    out_ptrs = out_ptr_base + dim_offsets * page_k_stride_dim
    
    tl.store(page_k_min_ptr + out_ptrs, min_vals, mask=dim_mask)
    tl.store(page_k_max_ptr + out_ptrs, max_vals, mask=dim_mask)
    
    if head_idx == 0:
        tl.store(page_valid_ptr + phys_page_idx, 1)

@triton.jit
def quest_retrieval_score_kernel(
    scores_ptr,
    reqs_ptr,
    seq_lens_ptr,
    req_to_token_ptr,
    page_k_min_ptr,
    page_k_max_ptr,
    queries_ptr,
    # Strides
    scores_stride_req,
    scores_stride_page,
    req_to_token_stride_req,
    req_to_token_stride_token,
    page_k_stride_page,
    page_k_stride_head,
    page_k_stride_dim,
    queries_stride_req,
    queries_stride_head,
    queries_stride_dim,
    # Shapes
    req_to_token_num_tokens,
    page_k_num_pages,
    # Constants
    PAGE_SIZE: tl.constexpr,
    HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    req_idx = tl.program_id(0)
    page_idx = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + req_idx)
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE

    if page_idx >= num_pages:
        # Write -inf for invalid pages
        offset = req_idx * scores_stride_req + page_idx * scores_stride_page
        tl.store(scores_ptr + offset, float("-inf"))
        return

    req_id = tl.load(reqs_ptr + req_idx)

    # Get physical page index
    # We need the first token of the page
    log_tok_idx = page_idx * PAGE_SIZE
    # Clamp for safety
    safe_log_tok_idx = tl.minimum(log_tok_idx, req_to_token_num_tokens - 1)
    
    offset_req_tok = req_id * req_to_token_stride_req + safe_log_tok_idx * req_to_token_stride_token
    phys_tok = tl.load(req_to_token_ptr + offset_req_tok)
    phys_page_idx = phys_tok // PAGE_SIZE
    
    # Clamp physical page index
    phys_page_idx = tl.minimum(phys_page_idx, page_k_num_pages - 1)
    phys_page_idx = tl.maximum(phys_page_idx, 0)

    # Compute score: sum(where(q>=0, q*k_max, q*k_min))
    
    acc = 0.0
    
    dim_offsets = tl.arange(0, BLOCK_DIM)
    dim_mask = dim_offsets < HEAD_DIM
    
    for h in range(HEAD_NUM):
        # Load Query
        q_off = req_idx * queries_stride_req + h * queries_stride_head + dim_offsets * queries_stride_dim
        q = tl.load(queries_ptr + q_off, mask=dim_mask, other=0.0).to(tl.float32)
        
        # Load K Min/Max
        k_base = phys_page_idx * page_k_stride_page + h * page_k_stride_head
        k_off = k_base + dim_offsets * page_k_stride_dim
        
        k_min = tl.load(page_k_min_ptr + k_off, mask=dim_mask, other=0.0).to(tl.float32)
        k_max = tl.load(page_k_max_ptr + k_off, mask=dim_mask, other=0.0).to(tl.float32)
        
        # Compute term
        # criticality = torch.where(q >= 0, q * k_max, q * k_min)
        term = tl.where(q >= 0, q * k_max, q * k_min)
        
        acc += tl.sum(term)
        
    # Store score
    offset = req_idx * scores_stride_req + page_idx * scores_stride_page
    tl.store(scores_ptr + offset, acc)
