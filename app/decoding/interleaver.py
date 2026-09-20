"""
app/decoding/interleaver.py
----------------------------
Constrained de-interleaver candidates: **None (bypass)** and **Block (N, M)**.

A block interleaver arranges `N` rows x `M` columns: data is written in row
major order and read out in column major order (or vice versa) so that bursts
of channel errors are spread across multiple FEC codewords.

Interleaver conventions supported here (both derived from a single matrix):
  - `InterleaverType.NONE`   : identity pass-through.
  - `InterleaverType.BLOCK`  : write-by-rows / read-by-columns.

Parameters
  N : number of rows
  M : number of columns
The interleaver matrix holds `N*M` bits (or bytes).

Every function round-trips: `interleave(deinterleave(x)) == x` and
`deinterleave(interleave(x)) == x` for properly sized inputs.

Enforce the *constrained candidate* philosophy: the parameter space explored
upstream/downstream is `NONE` plus a small grid of BLOCK (N, M) values, never
arbitrary pseudo-random or diagonal interleavers (research-grade / P2).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Union, Sequence

# Both integers and arrays of bits/bytes are supported as payloads.
Payload = Union[np.ndarray, Sequence[int], list[int], bytes]


class InterleaverType:
    """Constrained interleaver candidate family."""
    NONE = "none"
    BLOCK = "block"


@dataclass
class InterleaverSpec:
    """A fully-specified interleaver candidate."""
    type: str                  # InterleaverType.NONE | BLOCK
    n_rows: int = 0            # BLOCK only: number of rows
    n_cols: int = 0            # BLOCK only: number of columns

    @property
    def size(self) -> int:
        if self.type == InterleaverType.NONE:
            return 1.0
        return self.n_rows * self.n_cols

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "size": int(self.size) if self.type == InterleaverType.BLOCK else None,
        }

    def __str__(self) -> str:
        if self.type == InterleaverType.NONE:
            return "none"
        return f"block({self.n_rows}, {self.n_cols})"


def none_spec() -> InterleaverSpec:
    """Identity / bypass candidate."""
    return InterleaverSpec(type=InterleaverType.NONE)


def block_spec(n_rows: int, n_cols: int) -> InterleaverSpec:
    """Block candidate with explicit N (rows) x M (cols)."""
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("Block interleaver N and M must be positive integers.")
    return InterleaverSpec(type=InterleaverType.BLOCK, n_rows=int(n_rows), n_cols=int(n_cols))


# ── Primitive interleave / deinterleave ─────────────────────────────────────

def _as_flat_array(data: Payload) -> tuple[np.ndarray, object, bool]:
    """Coerce payload to a flat int array, remembering original container type."""
    if isinstance(data, bytes):
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
        return arr, "bytes"
    arr = np.asarray(data)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    flat = np.ravel(arr).astype(np.int64)
    dtype_kind = "bytes" if isinstance(data, (bytes, bytearray, np.uint8)) and arr.dtype == np.uint8 else "int"
    return flat, dtype_kind


def interleave(data: Payload, spec: InterleaverSpec) -> np.ndarray:
    """Apply the interleaver described by `spec`.

    Args:
        data: 1-D sequence of bits/bytes, length L.
        spec: NONE -> identity; BLOCK(N, M) -> length must be N*M.

    Returns:
        Interleaved 1-D int array.
    """
    arr, _ = _as_flat_array(data)
    if spec.type == InterleaverType.NONE:
        return arr.copy()
    return _block_interleave(arr, spec.n_rows, spec.n_cols)


def deinterleave(data: Payload, spec: InterleaverSpec) -> np.ndarray:
    """Inverse of `interleave`. Same size conventions."""
    arr, _ = _as_flat_array(data)
    if spec.type == InterleaverType.NONE:
        return arr.copy()
    return _block_deinterleave(arr, spec.n_rows, spec.n_cols)


def _block_interleave(arr: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Block interleave: the stream (length must be a multiple of N*M) is laid
    out as N x M blocks, written row-major per block and read column-major.

    Block-alignment is *required* because a length that is not a whole number
    of N*M blocks cannot round-trip under the same-length in/out contract
    (block-wise permutations are not invariant to truncation). Candidate chains
    whose interleaver size does not divide the stream length are simply skipped
    by the decode search.
    """
    total = n_rows * n_cols
    L = len(arr)
    if L % total != 0:
        raise ValueError(
            f"Block interleaver ({n_rows}x{n_cols}={total}) requires the stream "
            f"length {L} to be a multiple of {total}."
        )
    n_blocks = L // total
    out = np.empty_like(arr)
    for b in range(n_blocks):
        block = arr[b * total:(b + 1) * total].reshape(n_rows, n_cols)
        out[b * total:(b + 1) * total] = block.T.reshape(-1)
    return out


def _block_deinterleave(arr: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Inverse of `_block_interleave` (same block-alignment requirement)."""
    total = n_rows * n_cols
    L = len(arr)
    if L % total != 0:
        raise ValueError(
            f"Block deinterleaver ({n_rows}x{n_cols}={total}) requires the stream "
            f"length {L} to be a multiple of {total}."
        )
    n_blocks = L // total
    out = np.empty_like(arr)
    for b in range(n_blocks):
        block = arr[b * total:(b + 1) * total].reshape(n_cols, n_rows)
        out[b * total:(b + 1) * total] = block.T.reshape(-1)
    return out


# ── Candidate grid generator (for the hypothesis search) ────────────────────

def block_candidate_grid(max_rows: int = 8, max_cols: int = 16) -> list[InterleaverSpec]:
    """Generate a constrained set of BLOCK candidates for the search.

    Only "small" row x column pairs that exhaustively tile common FEC block
    lengths are scored. The composite size = N*M corresponds to a codeword
    granularity the decoder search will try to align to.
    """
    specs = []
    for n in range(1, max_rows + 1):
        for m in range(2, max_cols + 1):
            specs.append(block_spec(n, m))
    return specs


def default_candidate_specs(max_rows: int = 6, max_cols: int = 8) -> list[InterleaverSpec]:
    """Default interleaver candidates scored in `decode` search.

    Always includes NONE first (bypass), then a bounded BLOCK grid.
    """
    return [none_spec()] + block_candidate_grid(max_rows, max_cols)