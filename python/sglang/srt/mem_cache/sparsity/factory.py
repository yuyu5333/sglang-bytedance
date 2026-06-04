import json
import logging
from typing import Optional

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import BaseSparseAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.deepseek_nsa import DeepSeekNSAAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.freq_domain_algorithm import (
    FreqDomainAlgorithm,
)
from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import (
    FlashAttentionAdaptor,
    NSABackendAdaptor,
)
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import (
    SparseConfig,
    SparseCoordinator,
)

logger = logging.getLogger(__name__)

_global_sparse_coordinator: Optional[SparseCoordinator] = None

_ALGORITHM_REGISTRY = {
    "quest": lambda config, device, **kw: QuestAlgorithm(config, device, **kw),
    "deepseek_nsa": lambda config, device, **kw: DeepSeekNSAAlgorithm(
        config, device, **kw
    ),
    "freq_domain": lambda config, device, **kw: FreqDomainAlgorithm(
        config, device, **kw
    ),
}


def _create_sparse_algorithm(
    config: SparseConfig,
    device: torch.device,
    **kwargs,
) -> BaseSparseAlgorithm:
    algorithm_name = config.algorithm.lower()
    factory = _ALGORITHM_REGISTRY.get(algorithm_name)

    if factory is None:
        raise ValueError(f"Unknown sparse algorithm: {algorithm_name}")

    # #region debug-point A:create-sparse-algorithm
    exec(
        "try:\n import json, urllib.request\n _p='.dbg/freq-domain-accuracy.env'; _u='http://127.0.0.1:7777/event'; _s='freq-domain-accuracy'\n"
        " try:\n  with open(_p) as _f: _c=_f.read(); _u=next((l.split('=',1)[1] for l in _c.splitlines() if l.startswith('DEBUG_SERVER_URL=')), _u); _s=next((l.split('=',1)[1] for l in _c.splitlines() if l.startswith('DEBUG_SESSION_ID=')), _s)\n"
        " except Exception:\n  pass\n"
        " urllib.request.urlopen(urllib.request.Request(_u, data=json.dumps({'sessionId': _s, 'runId': 'pre-fix', 'hypothesisId': 'A', 'location': 'factory:_create_sparse_algorithm', 'msg': '[DEBUG] sparse algorithm requested', 'data': {'algorithm': algorithm_name, 'backend': config.backend, 'page_size': config.page_size}}).encode(), headers={'Content-Type': 'application/json'}), timeout=0.2).read()\n"
        "except Exception:\n pass"
    )
    # #endregion
    return factory(config, device, **kwargs)


def _create_backend_adaptor(
    backend: str,
    device: torch.device,
    sparse_algorithm: BaseSparseAlgorithm,
    req_to_token_pool,
):
    """Create backend adaptor."""
    if isinstance(sparse_algorithm, DeepSeekNSAAlgorithm):
        return NSABackendAdaptor(device, req_to_token_pool)

    if backend in ["fa3", "flashattention"]:
        return FlashAttentionAdaptor(device)

    raise ValueError(f"Unknown attention backend: {backend}")


def _parse_sparse_config(server_args) -> SparseConfig:
    """Parse hierarchical sparse config from JSON string.

    Required fields with defaults: top_k (2048), device_buffer_size (2*top_k),
    host_to_device_ratio (2).
    Optional fields (default None): algorithm, backend, min_sparse_prompt_len,
    page_size. All remaining fields go to sparse_extra_config.
    """
    extra_config_str = server_args.hisparse_config
    if extra_config_str is not None:
        try:
            extra_config = json.loads(extra_config_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse hisparse_config: {e}") from e
    else:
        extra_config = {}

    top_k = extra_config.pop("top_k", 2048)
    device_buffer_size = extra_config.pop("device_buffer_size", 2 * top_k)
    host_to_device_ratio = extra_config.pop("host_to_device_ratio", 2)

    if device_buffer_size < top_k:
        raise ValueError(
            f"device_buffer_size ({device_buffer_size}) must be no smaller than top_k ({top_k})"
        )

    algorithm = extra_config.pop("algorithm", None)
    backend = extra_config.pop("backend", None)
    min_sparse_prompt_len = extra_config.pop("min_sparse_prompt_len", None)
    page_size = extra_config.pop("page_size", None)

    # #region debug-point A:parse-hisparse-config
    exec(
        "try:\n import json, urllib.request\n _p='.dbg/freq-domain-accuracy.env'; _u='http://127.0.0.1:7777/event'; _s='freq-domain-accuracy'\n"
        " try:\n  with open(_p) as _f: _c=_f.read(); _u=next((l.split('=',1)[1] for l in _c.splitlines() if l.startswith('DEBUG_SERVER_URL=')), _u); _s=next((l.split('=',1)[1] for l in _c.splitlines() if l.startswith('DEBUG_SESSION_ID=')), _s)\n"
        " except Exception:\n  pass\n"
        " urllib.request.urlopen(urllib.request.Request(_u, data=json.dumps({'sessionId': _s, 'runId': 'pre-fix', 'hypothesisId': 'A', 'location': 'factory:_parse_sparse_config', 'msg': '[DEBUG] hisparse config parsed', 'data': {'algorithm': algorithm, 'backend': backend, 'top_k': top_k, 'device_buffer_size': device_buffer_size, 'host_to_device_ratio': host_to_device_ratio, 'page_size': page_size, 'extra_keys': sorted(list(extra_config.keys()))}}).encode(), headers={'Content-Type': 'application/json'}), timeout=0.2).read()\n"
        "except Exception:\n pass"
    )
    # #endregion
    return SparseConfig(
        top_k=top_k,
        device_buffer_size=device_buffer_size,
        host_to_device_ratio=host_to_device_ratio,
        algorithm=algorithm,
        backend=backend,
        page_size=page_size,
        min_sparse_prompt_len=min_sparse_prompt_len,
        sparse_extra_config=extra_config,
    )


def parse_hisparse_config(server_args) -> SparseConfig:
    """Parse hisparse config from server_args, returning defaults if no config provided."""
    return _parse_sparse_config(server_args)


def create_sparse_coordinator(
    device: torch.device,
    req_to_token_pool,
    token_to_kv_pool,
    start_layer: int,
    end_layer: int,
    server_args,
    **kwargs,
) -> SparseCoordinator:
    config = _parse_sparse_config(server_args)
    algorithm = _create_sparse_algorithm(config, device, **kwargs)
    backend_adaptor = _create_backend_adaptor(
        config.backend, device, algorithm, req_to_token_pool
    )

    coordinator = SparseCoordinator(
        config=config,
        algorithm=algorithm,
        backend_adaptor=backend_adaptor,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
        start_layer=start_layer,
        end_layer=end_layer,
        device=device,
    )
    register_sparse_coordinator(coordinator)
    # #region debug-point A:create-sparse-coordinator
    exec(
        "try:\n import json, urllib.request\n _p='.dbg/freq-domain-accuracy.env'; _u='http://127.0.0.1:7777/event'; _s='freq-domain-accuracy'\n"
        " try:\n  with open(_p) as _f: _c=_f.read(); _u=next((l.split('=',1)[1] for l in _c.splitlines() if l.startswith('DEBUG_SERVER_URL=')), _u); _s=next((l.split('=',1)[1] for l in _c.splitlines() if l.startswith('DEBUG_SESSION_ID=')), _s)\n"
        " except Exception:\n  pass\n"
        " urllib.request.urlopen(urllib.request.Request(_u, data=json.dumps({'sessionId': _s, 'runId': 'pre-fix', 'hypothesisId': 'A', 'location': 'factory:create_sparse_coordinator', 'msg': '[DEBUG] sparse coordinator created', 'data': {'algorithm_type': type(algorithm).__name__, 'backend_type': type(backend_adaptor).__name__, 'start_layer': start_layer, 'end_layer': end_layer}}).encode(), headers={'Content-Type': 'application/json'}), timeout=0.2).read()\n"
        "except Exception:\n pass"
    )
    # #endregion
    return coordinator


def register_sparse_coordinator(coordinator: SparseCoordinator) -> None:
    global _global_sparse_coordinator
    _global_sparse_coordinator = coordinator


def get_sparse_coordinator() -> Optional[SparseCoordinator]:
    return _global_sparse_coordinator
