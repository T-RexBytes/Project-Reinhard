"""
app/benchmark/cache.py
----------------------
Content-addressed, dependency-free stage caching.

Purpose (Build order §9.1): when a human confirms/overrides at a gate the DSP
re-verification must re-run *downstream stages only*. Spectral detection and
characterization (PSD, noise floor, SNR, bandwidth, features, symbol rate) are
pure functions of the input IQ + a handful of scalars, so their outputs are
safe to reuse across gate overrides — this is what keeps a gate override under
the sub-5s budget.

Design:
  - Key = SHA-256 over (stage name, input array bytes, canonical params).
    No file path, no dataset origin — two identical bursts produce one entry.
  - Memory LRU (fast, bounded) plus an optional disk directory for durable
    reuse across runs / API restarts.
  - Values are numpy arrays (saved as .npz) or JSON-serializable dicts (.json).
  - Pure-python, no framework dependency.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _canonical_params(params: Optional[dict]) -> str:
    """Deterministic, order-independent JSON of a params dict."""
    if not params:
        return "{}"
    return json.dumps(params, sort_keys=True, default=str, separators=(",", ":"))


def make_cache_key(
    stage: str,
    data: Optional[np.ndarray] = None,
    params: Optional[dict] = None,
) -> str:
    """Content-addressed cache key for a stage computation.

    ``data`` may be raw IQ, a PSD array, a bit stream, etc. Only the bytes and
    the canonical param string feed the hash, so equal inputs always collide.
    """
    h = hashlib.sha256()
    h.update(stage.encode("utf-8"))
    if data is not None:
        arr = np.ascontiguousarray(data)
        h.update(arr.tobytes())
        h.update(arr.shape.__str__().encode("utf-8"))
    h.update(_canonical_params(params).encode("utf-8"))
    return h.hexdigest()


class StageCache:
    """Bounded content-addressed cache with optional disk persistence."""

    def __init__(
        self,
        max_entries: int = 256,
        disk_dir: Optional[str | Path] = None,
        enabled: bool = True,
    ):
        self.max_entries = int(max_entries)
        self.enabled = enabled
        self._mem: dict[str, Any] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self.disk_dir: Optional[Path] = Path(disk_dir) if disk_dir else None
        if self.disk_dir is not None:
            self.disk_dir.mkdir(parents=True, exist_ok=True)

    # -- hot-path helpers ----------------------------------------------------
    def get(self, stage: str, data: Optional[np.ndarray] = None,
            params: Optional[dict] = None) -> Optional[Any]:
        if not self.enabled:
            return None
        key = make_cache_key(stage, data, params)
        with self._lock:
            if key in self._mem:
                self._order.remove(key)
                self._order.append(key)     # LRU touch
                return self._mem[key]
        if self.disk_dir is not None:
            value = self._read_disk(key)
            if value is not None:
                self._store_mem(key, value)
                return value
        return None

    def put(self, stage: str, data: Optional[np.ndarray] = None,
            params: Optional[dict] = None, value: Any = None) -> None:
        if not self.enabled:
            return
        key = make_cache_key(stage, data, params)
        self._store_mem(key, value)
        if self.disk_dir is not None:
            self._write_disk(key, value)

    def clear(self) -> None:
        with self._lock:
            self._mem.clear()
            self._order.clear()

    def stats(self) -> dict:
        """Hit/miss/size counters (lightweight, not atomic across callers)."""
        with self._lock:
            return {
                "size": len(self._mem),
                "max_entries": self.max_entries,
                "enabled": self.enabled,
                "disk_dir": str(self.disk_dir) if self.disk_dir else None,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._mem)

    # -- internals -----------------------------------------------------------
    def _store_mem(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._mem:                # duplicate guard (see doctest)
                self._order.remove(key)
            self._mem[key] = value
            self._order.append(key)
            while len(self._order) > self.max_entries:
                victim = self._order.pop(0)
                self._mem.pop(victim, None)

    def _disk_path(self, key: str, kind: str) -> Path:
        return self.disk_dir / f"{key}.{kind}"

    def _read_disk(self, key: str) -> Optional[Any]:
        npz = self._disk_path(key, "npz")
        js = self._disk_path(key, "json")
        if npz.exists():
            try:
                with np.load(npz) as f:
                    return {k: f[k] for k in f.files}
            except Exception:
                return None
        if js.exists():
            try:
                return json.loads(js.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _write_disk(self, key: str, value: Any) -> None:
        try:
            if isinstance(value, dict) and all(
                isinstance(v, np.ndarray) for v in value.values()
            ):
                np.savez(self._disk_path(key, "npz"), **value)
            else:
                js = self._disk_path(key, "json")
                js.write_text(json.dumps(value, default=str), encoding="utf-8")
        except Exception:
            pass    # cache is an optimization — never fail the pipeline

    def __len__(self) -> int:
        with self._lock:
            return len(self._mem)


# Module-level default cache shared by the advisor / generator infra.
_default_cache = StageCache()

def get_cache() -> StageCache:
    """Process-wide default StageCache instance."""
    return _default_cache