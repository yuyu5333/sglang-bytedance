"""Apply a physical-layout namespace at the common storage API boundary."""

from __future__ import annotations

import hashlib
from dataclasses import replace


class LayoutNamespacedStorage:
    """Delegate storage IO with new keys, without mutating radix-tree hashes.

    The wrapped backend may call its own v1/v2 methods internally; those calls
    see already transformed keys and are not wrapped a second time.
    """

    def __init__(self, backend, namespace: str):
        self.backend = backend
        self.namespace = namespace

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def _key(self, key):
        return hashlib.sha256(f"{self.namespace}:{key}".encode()).hexdigest()

    def _keys(self, keys):
        return None if keys is None else [self._key(key) for key in keys]

    def _extra(self, extra_info):
        if extra_info is None:
            return None
        return replace(extra_info, prefix_keys=self._keys(extra_info.prefix_keys))

    def _transfers(self, transfers):
        if transfers is None:
            return None
        return [replace(t, keys=self._keys(t.keys)) for t in transfers]

    def get(self, key, *args, **kwargs):
        return self.backend.get(self._key(key), *args, **kwargs)

    def set(self, key, *args, **kwargs):
        return self.backend.set(self._key(key), *args, **kwargs)

    def exists(self, key):
        return self.backend.exists(self._key(key))

    def batch_get(self, keys, *args, **kwargs):
        return self.backend.batch_get(self._keys(keys), *args, **kwargs)

    def batch_set(self, keys, *args, **kwargs):
        return self.backend.batch_set(self._keys(keys), *args, **kwargs)

    def batch_exists(self, keys, extra_info=None):
        return self.backend.batch_exists(self._keys(keys), self._extra(extra_info))

    def batch_get_v1(self, keys, host_indices, extra_info=None):
        return self.backend.batch_get_v1(
            self._keys(keys), host_indices, self._extra(extra_info)
        )

    def batch_set_v1(self, keys, host_indices, extra_info=None):
        return self.backend.batch_set_v1(
            self._keys(keys), host_indices, self._extra(extra_info)
        )

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        return self.backend.batch_exists_v2(
            self._keys(keys), self._transfers(pool_transfers), self._extra(extra_info)
        )

    def batch_get_v2(self, transfers, extra_info=None):
        return self.backend.batch_get_v2(
            self._transfers(transfers), self._extra(extra_info)
        )

    def batch_set_v2(self, transfers, extra_info=None):
        return self.backend.batch_set_v2(
            self._transfers(transfers), self._extra(extra_info)
        )
