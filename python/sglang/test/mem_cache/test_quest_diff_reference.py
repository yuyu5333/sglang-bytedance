import random

import torch


def _compact_keep_order(pages: torch.Tensor, topk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    keep_mask = pages != -1
    kept_pages = pages[keep_mask]
    kept_topk = topk[keep_mask]
    return kept_pages, kept_topk


def triton_style_reference(
    curr_top_k_pages: torch.Tensor,
    req_pool_indices: torch.Tensor,
    valid_lengths: torch.Tensor,
    seq_lens: torch.Tensor,
    sparse_mask: torch.Tensor,
    req_to_tokens_host: torch.Tensor,
    last_top_k: torch.Tensor,
    last_page_ids: torch.Tensor,
    layer_id: int,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bs, top_k_pages = curr_top_k_pages.shape
    physical_pages = torch.full((bs, top_k_pages), -1, dtype=torch.int64)
    load_tokens = torch.full((bs, top_k_pages * page_size), -1, dtype=torch.int64)
    load_tokens_host = torch.full((bs, top_k_pages * page_size), -1, dtype=torch.int64)

    for b in range(bs):
        req_idx = int(req_pool_indices[b].item())
        sl = int(seq_lens[b].item())
        vl = int(valid_lengths[b].item())
        if (not bool(sparse_mask[b].item())) or sl <= 0:
            continue

        last_topk_row = last_top_k[req_idx, layer_id].clone()
        last_pages_row = last_page_ids[req_idx, layer_id].clone()
        curr_topk_row = curr_top_k_pages[b].clone()
        curr_pages_row = torch.full_like(curr_topk_row, -1)
        load_mask = torch.zeros_like(curr_topk_row, dtype=torch.bool)

        for i in range(top_k_pages):
            val = int(curr_topk_row[i].item())
            if val == -1:
                continue
            for j in range(last_topk_row.numel()):
                if int(last_topk_row[j].item()) == val and int(last_pages_row[j].item()) != -1:
                    curr_pages_row[i] = last_pages_row[j]
                    last_pages_row[j] = -1
                    break

        kept_pages, kept_topk = _compact_keep_order(last_pages_row, last_topk_row)
        victim_count = int(kept_pages.numel())

        fill_needed = 0
        for i in range(top_k_pages):
            if i >= vl or int(curr_topk_row[i].item()) == -1:
                curr_topk_row[i] = -1
                curr_pages_row[i] = -1
                continue
            if int(curr_pages_row[i].item()) == -1:
                fill_needed += 1

        remaining_after_used = max(victim_count - fill_needed, 0)
        used = 0
        for i in range(top_k_pages):
            if i >= vl or int(curr_topk_row[i].item()) == -1:
                continue
            if int(curr_pages_row[i].item()) == -1:
                victim_idx = remaining_after_used + used
                if victim_idx < victim_count:
                    curr_pages_row[i] = kept_pages[victim_idx]
                    load_mask[i] = True
                    used += 1

        for i in range(top_k_pages):
            if not bool(load_mask[i].item()):
                continue
            log_page = int(curr_topk_row[i].item())
            phys_page = int(curr_pages_row[i].item())
            for token_offset in range(page_size):
                token_idx = log_page * page_size + token_offset
                if token_idx >= sl:
                    continue
                host_token = int(req_to_tokens_host[req_idx, token_idx].item())
                if host_token == -1:
                    continue
                out_idx = i * page_size + token_offset
                load_tokens[b, out_idx] = phys_page * page_size + token_offset
                load_tokens_host[b, out_idx] = host_token

        for i in range(top_k_pages):
            if i < vl and int(curr_topk_row[i].item()) != -1:
                physical_pages[b, i] = curr_pages_row[i]

        updated_last_topk = torch.full_like(last_topk_row, -1)
        updated_last_pages = torch.full_like(last_pages_row, -1)
        for i in range(last_topk_row.numel()):
            if i < top_k_pages:
                if i < vl and int(curr_topk_row[i].item()) != -1:
                    updated_last_topk[i] = curr_topk_row[i]
                    updated_last_pages[i] = curr_pages_row[i]
            else:
                src = i - top_k_pages
                if src < remaining_after_used:
                    updated_last_topk[i] = kept_topk[src]
                    updated_last_pages[i] = kept_pages[src]

        last_top_k[req_idx, layer_id] = updated_last_topk
        last_page_ids[req_idx, layer_id] = updated_last_pages

    return physical_pages, load_tokens, load_tokens_host, last_top_k, last_page_ids


def cuda_style_reference(
    curr_top_k_pages: torch.Tensor,
    req_pool_indices: torch.Tensor,
    valid_lengths: torch.Tensor,
    seq_lens: torch.Tensor,
    sparse_mask: torch.Tensor,
    req_to_tokens_host: torch.Tensor,
    last_top_k: torch.Tensor,
    last_page_ids: torch.Tensor,
    layer_id: int,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return triton_style_reference(
        curr_top_k_pages=curr_top_k_pages,
        req_pool_indices=req_pool_indices,
        valid_lengths=valid_lengths,
        seq_lens=seq_lens,
        sparse_mask=sparse_mask,
        req_to_tokens_host=req_to_tokens_host,
        last_top_k=last_top_k,
        last_page_ids=last_page_ids,
        layer_id=layer_id,
        page_size=page_size,
    )


def test_quest_diff_reference_alignment_cpu():
    torch.manual_seed(0)
    random.seed(0)

    page_size = 4
    bs = 4
    top_k_pages = 4
    hot_buffer_pages = 8
    num_layers = 3
    pool_size = 16
    max_context_len = 64
    layer_id = 1

    req_pool_indices = torch.tensor(random.sample(range(pool_size), bs), dtype=torch.int64)
    seq_lens = torch.tensor([random.randint(1, max_context_len // 2) for _ in range(bs)], dtype=torch.int64)
    valid_lengths = torch.tensor([random.randint(0, top_k_pages) for _ in range(bs)], dtype=torch.int64)
    sparse_mask = torch.ones((bs,), dtype=torch.int64)

    curr_top_k_pages = torch.full((bs, top_k_pages), -1, dtype=torch.int64)
    for b in range(bs):
        vl = int(valid_lengths[b].item())
        max_pages = max(int((seq_lens[b].item() + page_size - 1) // page_size), 1)
        pages = random.sample(range(max_pages), min(vl, max_pages))
        curr_top_k_pages[b, : len(pages)] = torch.tensor(pages, dtype=torch.int64)

    req_to_tokens_host = torch.full((pool_size, max_context_len), -1, dtype=torch.int64)
    for req in range(pool_size):
        for t in range(max_context_len):
            req_to_tokens_host[req, t] = req * 10000 + t

    last_top_k = torch.full((pool_size, num_layers, hot_buffer_pages), -1, dtype=torch.int64)
    last_page_ids = torch.full((pool_size, num_layers, hot_buffer_pages), -1, dtype=torch.int64)
    for req in range(pool_size):
        for l in range(num_layers):
            last_top_k[req, l] = torch.arange(hot_buffer_pages, dtype=torch.int64)
            last_page_ids[req, l] = torch.arange(req * 1000, req * 1000 + hot_buffer_pages, dtype=torch.int64)

    physical_pages_a, load_tokens_a, load_tokens_host_a, last_top_k_a, last_page_ids_a = triton_style_reference(
        curr_top_k_pages=curr_top_k_pages.clone(),
        req_pool_indices=req_pool_indices,
        valid_lengths=valid_lengths.clone(),
        seq_lens=seq_lens.clone(),
        sparse_mask=sparse_mask.clone(),
        req_to_tokens_host=req_to_tokens_host.clone(),
        last_top_k=last_top_k.clone(),
        last_page_ids=last_page_ids.clone(),
        layer_id=layer_id,
        page_size=page_size,
    )

    physical_pages_b, load_tokens_b, load_tokens_host_b, last_top_k_b, last_page_ids_b = cuda_style_reference(
        curr_top_k_pages=curr_top_k_pages.clone(),
        req_pool_indices=req_pool_indices,
        valid_lengths=valid_lengths.clone(),
        seq_lens=seq_lens.clone(),
        sparse_mask=sparse_mask.clone(),
        req_to_tokens_host=req_to_tokens_host.clone(),
        last_top_k=last_top_k.clone(),
        last_page_ids=last_page_ids.clone(),
        layer_id=layer_id,
        page_size=page_size,
    )

    assert torch.equal(physical_pages_a, physical_pages_b)
    assert torch.equal(load_tokens_a, load_tokens_b)
    assert torch.equal(load_tokens_host_a, load_tokens_host_b)
    assert torch.equal(last_top_k_a, last_top_k_b)
    assert torch.equal(last_page_ids_a, last_page_ids_b)


def test_quest_diff_victim_from_tail_cpu():
    page_size = 4
    hot_buffer_pages = 8
    num_layers = 2
    pool_size = 4
    max_context_len = 64
    layer_id = 0

    req_pool_indices = torch.tensor([1], dtype=torch.int64)
    seq_lens = torch.tensor([30], dtype=torch.int64)
    valid_lengths = torch.tensor([4], dtype=torch.int64)
    sparse_mask = torch.tensor([1], dtype=torch.int64)

    curr_top_k_pages = torch.tensor([[0, 1, 4, 5]], dtype=torch.int64)

    req_to_tokens_host = torch.full((pool_size, max_context_len), -1, dtype=torch.int64)
    for req in range(pool_size):
        for t in range(max_context_len):
            req_to_tokens_host[req, t] = req * 10000 + t

    last_top_k = torch.full((pool_size, num_layers, hot_buffer_pages), -1, dtype=torch.int64)
    last_page_ids = torch.full((pool_size, num_layers, hot_buffer_pages), -1, dtype=torch.int64)
    last_top_k[1, layer_id] = torch.tensor([0, 1, 2, 3, 6, 7, 8, 9], dtype=torch.int64)
    last_page_ids[1, layer_id] = torch.arange(1000, 1000 + hot_buffer_pages, dtype=torch.int64)

    physical_pages, _, _, last_top_k_out, last_page_ids_out = triton_style_reference(
        curr_top_k_pages=curr_top_k_pages.clone(),
        req_pool_indices=req_pool_indices,
        valid_lengths=valid_lengths.clone(),
        seq_lens=seq_lens.clone(),
        sparse_mask=sparse_mask.clone(),
        req_to_tokens_host=req_to_tokens_host.clone(),
        last_top_k=last_top_k.clone(),
        last_page_ids=last_page_ids.clone(),
        layer_id=layer_id,
        page_size=page_size,
    )

    assert torch.equal(physical_pages[0, :2], torch.tensor([1000, 1001], dtype=torch.int64))
    assert torch.equal(physical_pages[0, 2:], torch.tensor([1006, 1007], dtype=torch.int64))

    expected_last_pages = torch.tensor([1000, 1001, 1006, 1007, 1002, 1003, 1004, 1005], dtype=torch.int64)
    assert torch.equal(last_page_ids_out[1, layer_id], expected_last_pages)
    expected_last_topk = torch.tensor([0, 1, 4, 5, 2, 3, 6, 7], dtype=torch.int64)
    assert torch.equal(last_top_k_out[1, layer_id], expected_last_topk)
