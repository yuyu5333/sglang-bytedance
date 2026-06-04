"""
Frequency-Domain KV cache sparse attention algorithm.

Inspired by frequency-domain image steganography (FFT/DCT watermarking):
treat each KV page as a length-P "signal" along the token axis, transform
it with a real DCT-II, and only retain the top-r low-frequency coefficients
as the page representation. Page importance for top-k retrieval is then
estimated entirely in the frequency domain, without ever materialising the
full K block.

Mathematically:

    For a page K of shape [P, H, D] (P = page_size, H = num_heads, D = head_dim),
    the DCT-II coefficients along the token axis are

        Y[k, h, d] = sum_{n=0..P-1} C[k, n] * K[n, h, d]   for k = 0..P-1

    where C is the orthonormal DCT-II basis. We store only the first
    `num_freq_keep` rows Y[0..r-1, :, :] per page (r << P).

    Given a query q in R^{H, D}, the dot product with any reconstructed
    key  K~[n] = sum_k C[k, n] * Y[k]  satisfies

        <q, K~[n]> = sum_k C[k, n] * <q, Y[k]>

    so the L1 envelope  S = sum_k |<q, Y[k]>|  is a frequency-domain
    surrogate of the attention score for that page (it is a tight upper
    bound up to the DCT basis inf-norm and is monotone in true relevance
    when the page energy is concentrated in the low-frequency band).

This delivers a cache where:

  * Storage per page is r/P of a Quest-style page representation (and we
    can push r down to 1-4 for >32x compression of the score table);
  * The DC term Y[0] is exactly proportional to the page mean key, so the
    algorithm strictly generalises mean-pooling page selection;
  * Higher frequency terms (Y[1], Y[2], ...) capture intra-page key
    variation -- they're the "high-frequency watermark" that lets us
    distinguish pages whose mean is similar but whose internal structure
    differs.
"""

import logging
import math

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
    BaseSparseAlgorithmImpl,
)

logger = logging.getLogger(__name__)


def _build_dct_basis(
    page_size: int, num_freq_keep: int, device: torch.device, dtype=torch.float32
) -> torch.Tensor:
    """Build the truncated orthonormal DCT-II basis matrix.

    Returns a tensor C of shape [num_freq_keep, page_size] such that

        Y[k] = sum_n C[k, n] * X[n]

    is the k-th DCT-II coefficient of the length-P signal X. The basis is
    orthonormal so the inverse transform is C.T (over the kept rows it
    becomes a projection onto the low-frequency subspace).
    """
    if num_freq_keep < 1 or num_freq_keep > page_size:
        raise ValueError(
            f"num_freq_keep ({num_freq_keep}) must be in [1, page_size={page_size}]"
        )

    n = torch.arange(page_size, device=device, dtype=dtype)
    k = torch.arange(num_freq_keep, device=device, dtype=dtype).unsqueeze(1)
    # C[k, n] = a_k * cos(pi * (2n + 1) * k / (2P))
    basis = torch.cos(math.pi * (2.0 * n + 1.0) * k / (2.0 * page_size))
    # Orthonormal scaling: a_0 = sqrt(1/P), a_{k>0} = sqrt(2/P).
    scale = torch.full(
        (num_freq_keep, 1), math.sqrt(2.0 / page_size), device=device, dtype=dtype
    )
    if num_freq_keep > 0:
        scale[0, 0] = math.sqrt(1.0 / page_size)
    return basis * scale


class FreqDomainAlgorithm(BaseSparseAlgorithmImpl):
    """Frequency-domain page-wise sparse attention.

    Configurable knobs (all optional, set via ``sparse_extra_config``):

      - ``num_freq_keep`` (int, default 4): number of low-frequency DCT
        coefficients retained per page. Smaller = more aggressive
        compression, larger = better fidelity.
      - ``score_mode`` (str, default ``"l1"``): how to aggregate
        ``<q, Y[k]>`` across kept frequencies.
          * ``"l1"``: ``sum_k |<q, Y[k]>|``  (frequency-domain L1 envelope,
            tight upper bound on true page max-score).
          * ``"l2"``: ``sqrt(sum_k <q, Y[k]>^2)``  (Parseval-style energy).
          * ``"dc"``: ``<q, Y[0]>`` only  (degenerates to mean-pooling).
    """

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        self.num_freq_keep = int(
            config.sparse_extra_config.get("num_freq_keep", 4)
        )
        self.score_mode = str(
            config.sparse_extra_config.get("score_mode", "l1")
        ).lower()
        if self.score_mode not in ("l1", "l2", "dc"):
            raise ValueError(
                f"Unknown freq_domain score_mode: {self.score_mode!r}"
                " (expected 'l1', 'l2' or 'dc')"
            )

        # Per-layer storage of low-frequency DCT coefficients.
        self.page_freq_coeffs = {}
        self.page_valid = {}
        # DCT basis cache: only depends on (page_size, num_freq_keep, device).
        self._dct_basis = None
        self._debug_repr_log_count = 0
        self._debug_score_log_count = 0
        # #region debug-point B:freq-domain-init
        exec(
            "try:\n import json, urllib.request\n _p='.dbg/freq-domain-accuracy.env'; _u='http://127.0.0.1:7777/event'; _s='freq-domain-accuracy'\n"
            " try:\n  with open(_p) as _f: _c=_f.read(); _u=next((l.split('=',1)[1] for l in _c.splitlines() if l.startswith('DEBUG_SERVER_URL=')), _u); _s=next((l.split('=',1)[1] for l in _c.splitlines() if l.startswith('DEBUG_SESSION_ID=')), _s)\n"
            " except Exception:\n  pass\n"
            " urllib.request.urlopen(urllib.request.Request(_u, data=json.dumps({'sessionId': _s, 'runId': 'pre-fix', 'hypothesisId': 'B', 'location': 'freq_domain:__init__', 'msg': '[DEBUG] freq_domain algorithm initialized', 'data': {'num_freq_keep': int(self.num_freq_keep), 'score_mode': self.score_mode, 'page_size': int(self.page_size), 'sparsity_ratio': float(self.sparsity_ratio), 'num_recent_pages': int(self.num_recent_pages)}}).encode(), headers={'Content-Type': 'application/json'}), timeout=0.2).read()\n"
            "except Exception:\n pass"
        )
        # #endregion

    # ------------------------------------------------------------------
    # Pool initialisation
    # ------------------------------------------------------------------
    def _initialize_representation_pools(
        self, start_layer: int, end_layer: int, total_num_pages: int
    ):
        key_buf = self.token_to_kv_pool.get_key_buffer(start_layer)
        head_num, head_dim = key_buf.shape[1], key_buf.shape[2]

        # Cap retained frequencies at the actual page size (page_size could
        # legitimately be 1 for token-wise sparsity).
        effective_freq = min(self.num_freq_keep, max(self.page_size, 1))
        if effective_freq != self.num_freq_keep:
            logger.warning(
                "FreqDomain: clamping num_freq_keep from %d to %d (page_size=%d)",
                self.num_freq_keep,
                effective_freq,
                self.page_size,
            )
        self.num_freq_keep = effective_freq

        for layer_id in range(start_layer, end_layer):
            self.page_freq_coeffs[layer_id] = torch.zeros(
                (total_num_pages, self.num_freq_keep, head_num, head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            self.page_valid[layer_id] = torch.zeros(
                total_num_pages, dtype=torch.bool, device=self.device
            )

        self._dct_basis = _build_dct_basis(
            page_size=max(self.page_size, 1),
            num_freq_keep=self.num_freq_keep,
            device=self.device,
            dtype=torch.float32,
        )

        logger.info(
            "Initialized FreqDomain page reps: %d pages, %d layers, "
            "head_num=%d, head_dim=%d, page_size=%d, num_freq_keep=%d, score_mode=%s",
            total_num_pages,
            end_layer - start_layer,
            head_num,
            head_dim,
            self.page_size,
            self.num_freq_keep,
            self.score_mode,
        )
        # #region debug-point B:init-representation-pool
        exec(
            "try:\n import json, urllib.request\n _p='.dbg/freq-domain-accuracy.env'; _u='http://127.0.0.1:7777/event'; _s='freq-domain-accuracy'\n"
            " try:\n  with open(_p) as _f: _c=_f.read(); _u=next((l.split('=',1)[1] for l in _c.splitlines() if l.startswith('DEBUG_SERVER_URL=')), _u); _s=next((l.split('=',1)[1] for l in _c.splitlines() if l.startswith('DEBUG_SESSION_ID=')), _s)\n"
            " except Exception:\n  pass\n"
            " urllib.request.urlopen(urllib.request.Request(_u, data=json.dumps({'sessionId': _s, 'runId': 'pre-fix', 'hypothesisId': 'B', 'location': 'freq_domain:_initialize_representation_pools', 'msg': '[DEBUG] freq_domain representation pools initialized', 'data': {'start_layer': int(start_layer), 'end_layer': int(end_layer), 'total_num_pages': int(total_num_pages), 'num_freq_keep': int(self.num_freq_keep), 'page_size': int(self.page_size), 'head_num': int(head_num), 'head_dim': int(head_dim)}}).encode(), headers={'Content-Type': 'application/json'}), timeout=0.2).read()\n"
            "except Exception:\n pass"
        )
        # #endregion

    # ------------------------------------------------------------------
    # Representation construction (prefill + decode share this path)
    # ------------------------------------------------------------------
    def _compute_page_representations(
        self,
        layer_id: int,
        reqs: torch.Tensor,
        seq_lens: torch.Tensor,
        start_page,
        end_page: torch.Tensor,
        k_buffer: torch.Tensor,
    ):
        if isinstance(start_page, int):
            start_page = torch.full_like(end_page, start_page)

        device = k_buffer.device
        req_to_token = self.req_to_token_pool.req_to_token
        n = reqs.shape[0]
        max_pages = int((end_page - start_page).max().item())
        if max_pages <= 0:
            return

        page_size = self.page_size
        # [n, max_pages] -- absolute page index in the request token stream.
        pg_off = torch.arange(max_pages, device=device).unsqueeze(0)
        pg_id = start_page.unsqueeze(1) + pg_off
        pg_mask = pg_id < end_page.unsqueeze(1)

        tok_start = pg_id * page_size
        tok_off = torch.arange(page_size, device=device).view(1, 1, -1)
        tok_pos = tok_start.unsqueeze(2) + tok_off
        tok_mask = (
            tok_pos
            < (tok_start + page_size).clamp(max=seq_lens.unsqueeze(1)).unsqueeze(2)
        ) & pg_mask.unsqueeze(2)

        phys_tok = req_to_token[
            reqs.view(n, 1, 1).expand(n, max_pages, page_size),
            tok_pos.clamp(0, req_to_token.shape[1] - 1),
        ].clamp(0, k_buffer.shape[0] - 1)

        # keys: [n, max_pages, P, H, D]
        keys = k_buffer[phys_tok].to(torch.float32)

        # Zero out padding tokens so the DCT sees a clean signal of length P.
        keys = keys * tok_mask.unsqueeze(-1).unsqueeze(-1).to(keys.dtype)

        # DCT-II along the page-token axis:
        #   coeffs[..., k, h, d] = sum_n basis[k, n] * keys[..., n, h, d]
        # basis: [r, P]  ->  einsum 'kn,nphd...' style.
        basis = self._dct_basis  # [r, P]
        # Reshape keys to [n*max_pages, P, H*D] for a single matmul, then back.
        n_pg = n * max_pages
        H = keys.shape[3]
        D = keys.shape[4]
        flat_keys = keys.reshape(n_pg, page_size, H * D)
        # [r, P] @ [n_pg, P, H*D] -> [n_pg, r, H*D]
        coeffs = torch.einsum("kn,bnf->bkf", basis, flat_keys)
        coeffs = coeffs.reshape(n, max_pages, self.num_freq_keep, H, D)

        # Map logical pages back to physical page indices (each page maps to
        # the physical page that owns its first token).
        phys_pg = (
            req_to_token[
                reqs.unsqueeze(1).expand(n, max_pages),
                tok_start.clamp(0, req_to_token.shape[1] - 1),
            ]
            // page_size
        )

        idx = pg_mask.nonzero(as_tuple=False)
        if idx.numel() == 0:
            return

        target_pages = phys_pg[idx[:, 0], idx[:, 1]].clamp(
            0, self.page_freq_coeffs[layer_id].shape[0] - 1
        )
        self.page_freq_coeffs[layer_id][target_pages] = coeffs[idx[:, 0], idx[:, 1]]
        self.page_valid[layer_id][target_pages] = True
        if self._debug_repr_log_count < 4:
            self._debug_repr_log_count += 1
            valid_tokens = tok_mask.sum(dim=-1)
            tail_pages = (valid_tokens < page_size) & pg_mask
            # #region debug-point C:compute-page-representations
            exec(
                "try:\n import json, urllib.request\n _p='.dbg/freq-domain-accuracy.env'; _u='http://127.0.0.1:7777/event'; _s='freq-domain-accuracy'\n"
                " try:\n  with open(_p) as _f: _c=_f.read(); _u=next((l.split('=',1)[1] for l in _c.splitlines() if l.startswith('DEBUG_SERVER_URL=')), _u); _s=next((l.split('=',1)[1] for l in _c.splitlines() if l.startswith('DEBUG_SESSION_ID=')), _s)\n"
                " except Exception:\n  pass\n"
                " urllib.request.urlopen(urllib.request.Request(_u, data=json.dumps({'sessionId': _s, 'runId': 'pre-fix', 'hypothesisId': 'C', 'location': 'freq_domain:_compute_page_representations', 'msg': '[DEBUG] freq_domain page representations computed', 'data': {'layer_id': int(layer_id), 'batch_size': int(n), 'max_pages': int(max_pages), 'page_size': int(page_size), 'valid_page_count': int(idx.shape[0]), 'tail_page_count': int(tail_pages.sum().item()), 'valid_tokens_min': int(valid_tokens[pg_mask].min().item()) if pg_mask.any() else 0, 'valid_tokens_max': int(valid_tokens[pg_mask].max().item()) if pg_mask.any() else 0, 'valid_tokens_sample': valid_tokens[pg_mask][: min(8, valid_tokens[pg_mask].numel())].tolist() if pg_mask.any() else []}}).encode(), headers={'Content-Type': 'application/json'}), timeout=0.2).read()\n"
                "except Exception:\n pass"
            )
            # #endregion

    # ------------------------------------------------------------------
    # Top-k page scoring in the frequency domain
    # ------------------------------------------------------------------
    def _retrieve_page_scores(
        self,
        layer_id: int,
        phys_pages: torch.Tensor,
        req_pool_indices: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        phys_pages_clamped = phys_pages.clamp(
            0, self.page_freq_coeffs[layer_id].shape[0] - 1
        )

        # coeffs: [bs, P_query, r, H, D]
        coeffs = self.page_freq_coeffs[layer_id][phys_pages_clamped]
        valid_mask = self.page_valid[layer_id][phys_pages_clamped]

        head_dim = coeffs.shape[-1]
        kv_heads = coeffs.shape[-2]

        # Align query shape to KV heads (mirrors Quest's GQA handling).
        if queries.dim() == 2:
            bs, hidden = queries.shape
            if hidden % head_dim != 0:
                raise ValueError(
                    f"FreqDomain query hidden size {hidden} not divisible by "
                    f"head_dim {head_dim}"
                )
            q_heads = hidden // head_dim
            q = queries.view(bs, q_heads, head_dim)
        elif queries.dim() == 3:
            q = queries
        else:
            raise ValueError(
                f"Unsupported query shape for FreqDomain: {queries.shape}"
            )

        q_heads = q.shape[1]
        if q_heads != kv_heads:
            if q_heads % kv_heads != 0:
                raise ValueError(
                    f"Query heads {q_heads} not divisible by KV heads {kv_heads}"
                )
            group = q_heads // kv_heads
            q = q.view(q.shape[0], kv_heads, group, head_dim).mean(dim=2)

        q = q.to(coeffs.dtype)  # [bs, H, D]

        # <q, Y[k]>  for every kept frequency k:
        #   inner: [bs, P_query, r]
        # We sum over the (H, D) axes of q*coeffs.
        # coeffs: [bs, Pq, r, H, D]; q: [bs, 1, 1, H, D]
        inner = (coeffs * q.unsqueeze(1).unsqueeze(2)).sum(dim=(-1, -2))

        if self.score_mode == "dc":
            # DC term only -- equivalent to scoring against the page mean.
            criticality = inner[..., 0]
        elif self.score_mode == "l2":
            criticality = torch.sqrt(torch.clamp((inner * inner).sum(dim=-1), min=0.0))
        else:  # "l1" (default): tight envelope on the true page max-score.
            criticality = inner.abs().sum(dim=-1)

        criticality = torch.where(
            valid_mask, criticality, torch.full_like(criticality, float("-inf"))
        )
        if self._debug_score_log_count < 8:
            self._debug_score_log_count += 1
            finite_scores = criticality[torch.isfinite(criticality)]
            # #region debug-point D:retrieve-page-scores
            exec(
                "try:\n import json, urllib.request\n _p='.dbg/freq-domain-accuracy.env'; _u='http://127.0.0.1:7777/event'; _s='freq-domain-accuracy'\n"
                " try:\n  with open(_p) as _f: _c=_f.read(); _u=next((l.split('=',1)[1] for l in _c.splitlines() if l.startswith('DEBUG_SERVER_URL=')), _u); _s=next((l.split('=',1)[1] for l in _c.splitlines() if l.startswith('DEBUG_SESSION_ID=')), _s)\n"
                " except Exception:\n  pass\n"
                " urllib.request.urlopen(urllib.request.Request(_u, data=json.dumps({'sessionId': _s, 'runId': 'pre-fix', 'hypothesisId': 'D', 'location': 'freq_domain:_retrieve_page_scores', 'msg': '[DEBUG] freq_domain page scores computed', 'data': {'layer_id': int(layer_id), 'score_mode': self.score_mode, 'kv_heads': int(kv_heads), 'q_heads': int(q_heads), 'gqa_group': int(q_heads // kv_heads) if q_heads >= kv_heads and kv_heads > 0 and q_heads % kv_heads == 0 else None, 'phys_pages_shape': list(phys_pages.shape), 'valid_count': int(valid_mask.sum().item()), 'finite_score_count': int(finite_scores.numel()), 'score_min': float(finite_scores.min().item()) if finite_scores.numel() > 0 else None, 'score_max': float(finite_scores.max().item()) if finite_scores.numel() > 0 else None, 'score_mean': float(finite_scores.mean().item()) if finite_scores.numel() > 0 else None, 'score_sample': finite_scores[: min(8, finite_scores.numel())].tolist() if finite_scores.numel() > 0 else []}}).encode(), headers={'Content-Type': 'application/json'}), timeout=0.2).read()\n"
                "except Exception:\n pass"
            )
            # #endregion

        return criticality
