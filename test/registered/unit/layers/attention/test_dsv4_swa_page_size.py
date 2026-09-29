from types import SimpleNamespace

from sglang.srt.layers.attention.deepseek_v4_backend import _swa_cache_page_size


def test_swa_page_size_uses_paged_pool():
    pool = SimpleNamespace(
        request_window=None,
        swa_kv_pool=SimpleNamespace(page_size=256),
    )

    assert _swa_cache_page_size(pool) == 256


def test_swa_page_size_uses_request_window():
    pool = SimpleNamespace(
        request_window=SimpleNamespace(page_size=128),
        swa_kv_pool=None,
    )

    assert _swa_cache_page_size(pool) == 128
