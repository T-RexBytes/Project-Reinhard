"""
app/decoding/frame.py
---------------------
Frame-level primitives: CRC-16 / CRC-32, preamble (sync-word) correlation, and
frame-boundary detection.

The demod + FEC chain produces a sequence of bits; the frame layer finds where
a transmission "frame" starts (preamble / sync word), extracts the candidate
header + payload region, and verifies it with a CRC so the hypothesis engine
can lock onto the correct (modulation -> demux -> FEC -> frame) DAG.

Constrained candidate philosophy
--------------------------------
Only well-known CRC variants are shipped:
  - CRC-16/IBM (reflected poly 0xA001, init 0xFFFF)
  - CRC-16/CCITT (reflected poly 0x8408, init 0xFFFF)
  - CRC-32/IEEE  (reflected poly 0xEDB88320, init 0xFFFFFFFF, xorout 0xFFFFFFFF)

Arbitrary generator discovery / adaptive CRC sniffing is out of scope (P2).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence, Union

# ── Reflected CRC table construction ─────────────────────────────────────────

def _make_crc_table(width: int, reflected_poly: int) -> np.ndarray:
    """Build a length-256 lookup table for a *reflected* CRC polynomial."""
    table = np.zeros(256, dtype=np.uint64)
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ reflected_poly
            else:
                crc >>= 1
        table[i] = crc & ((1 << width) - 1) if width < 64 else crc
    return table


def _make_crc16_table_msb(poly: int) -> np.ndarray:
    """Build a length-256 table for a *non-reflected* CRC-16 polynomial
    (used by the classic CCITT-FALSE flavour 0x1021)."""
    table = np.zeros(256, dtype=np.uint64)
    for i in range(256):
        crc = i << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
        table[i] = crc
    return table


_CRC16_IBM_TABLE = _make_crc_table(16, 0xA001)
_CRC16_CCITT_TABLE = _make_crc16_table_msb(0x1021)
_CRC32_IEEE_TABLE = _make_crc_table(32, 0xEDB88320)

CRC16_IBM = "crc16_ibm"
CRC16_CCITT = "crc16_ccitt"
CRC32_IEEE = "crc32_ieee"
CRC_VARIANTS = (CRC16_IBM, CRC16_CCITT, CRC32_IEEE)


def _as_bytes(data: Union[bytes, bytearray, Sequence[int], np.ndarray]) -> bytes:
    """Coerce payload to bytes (int array entries must fit in 8 bits)."""
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    arr = np.asarray(data).ravel()
    if arr.size == 0:
        return b""
    return bytes(int(v) & 0xFF for v in arr)


def crc16(data, variant: str = CRC16_IBM) -> int:
    """CRC-16 checksum of *byte-oriented* payload.

    Args:
        data: bytes or array of 8-bit symbols.
        variant: CRC16_IBM (0xA001) or CRC16_CCITT (0x8408).

    Returns:
        Computed 16-bit CRC value (init 0xFFFF, no final xor).
    """
    b = _as_bytes(data)
    if variant == CRC16_CCITT:
        table, init, xorout = _CRC16_CCITT_TABLE, 0xFFFF, 0x0000
    else:
        table, init, xorout = _CRC16_IBM_TABLE, 0xFFFF, 0x0000
    crc = init
    for byte in b:
        if variant == CRC16_CCITT:
            crc = ((crc << 8) ^ int(table[((crc >> 8) ^ byte) & 0xFF])) & 0xFFFF
        else:
            crc = (crc >> 8) ^ int(table[(crc ^ byte) & 0xFF])
    crc ^= xorout
    return crc & 0xFFFF


def crc32(data) -> int:
    """CRC-32/IEEE checksum of *byte-oriented* payload (init/xorout 0xFFFFFFFF)."""
    b = _as_bytes(data)
    crc = 0xFFFFFFFF
    for byte in b:
        crc = (crc >> 8) ^ int(_CRC32_IEEE_TABLE[(crc ^ byte) & 0xFF])
    return crc ^ 0xFFFFFFFF


def crc_bytes(data, variant: str = CRC16_IBM) -> bytes:
    """CRC as big-endian bytes for appending to a frame (2 or 4 bytes)."""
    if variant == CRC32_IEEE:
        return crc32(data).to_bytes(4, "big")
    return crc16(data, variant).to_bytes(2, "big")


def crc_check(data, crc_value: int, variant: str = CRC16_IBM, expected_lsb: bool = False) -> bool:
    """Verify a stored CRC value against the payload, recomputing the checksum.

    Args:
        data: payload bytes/symbols the CRC covers.
        crc_value: CRC as stored in the frame (int).
        variant: CRC16_IBM / CRC16_CCITT / CRC32_IEEE.
        expected_lsb: whether the frame serializes the CRC little-endian
                      (reflected CRC families typically store the low byte first).

    Returns:
        True when ``crc_value`` equals the recomputed checksum (after unpacking
        from the on-wire byte order).
    """
    width = 4 if variant == CRC32_IEEE else 2
    stored = int(crc_value).to_bytes(width, "little" if expected_lsb else "big")
    recomputed = crc32(_as_bytes(data)) if variant == CRC32_IEEE else crc16(_as_bytes(data), variant)
    return int.from_bytes(stored, "little" if expected_lsb else "big") == recomputed


# ── Bit-stream helpers ───────────────────────────────────────────────────────

def bits_to_bytes(bits: np.ndarray) -> np.ndarray:
    """Pack a 0/1 bit array into 8-bit symbol bytes (MSB-first within byte)."""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    n_full = len(bits) // 8
    msb_first = bits[: n_full * 8].reshape(n_full, 8)
    vals = (msb_first * (1 << np.arange(7, -1, -1))).sum(axis=1)
    if n_full * 8 < len(bits):
        tail = bits[n_full * 8:]
        vals = np.concatenate([vals, [(tail * (1 << np.arange(len(tail) - 1, -1, -1))).sum()]])
    return vals.astype(np.uint8)


def bytes_to_bits(data, n_bits: Optional[int] = None) -> np.ndarray:
    """Unpack byte array into MSB-first 0/1 bits (optionally truncated)."""
    b = _as_bytes(data)
    bits = np.unpackbits(np.frombuffer(b, dtype=np.uint8))
    return bits[:n_bits] if n_bits is not None else bits


# ── Preamble / sync-word correlation ────────────────────────────────────────

def correlate_preamble(
    bits: np.ndarray,
    preamble: Sequence[int],
    search_length: Optional[int] = None,
) -> tuple[Optional[int], float]:
    """Locate the best preamble match in a bit stream by sliding Hamming match.

    Args:
        bits: full demodulated bit stream (0/1).
        preamble: known sync-word.
        search_length: cap the number of candidate offsets searched (None = all).

    Returns:
        ``(best_offset, best_hamming_matrix)`` or ``(None, 0.0)`` if nothing
        matched better than a random-guess baseline.
    """
    if len(bits) == 0:
        return None, 0.0
    p = np.asarray(preamble, dtype=np.int64).ravel()
    Lp = len(p)
    if Lp == 0:
        return None, 0.0
    n_search = int(search_length or max(0, len(bits) - Lp + 1))
    if n_search <= 0:
        return None, 0.0
    best_offset = None
    best_err = Lp  # Hamming distance
    for off in range(n_search):
        window = bits[off:off + Lp]
        if len(window) < Lp:
            break
        err = int(np.count_nonzero(window != p))
        if err < best_err:
            best_err = err
            best_offset = off
    if best_offset is None:
        return None, 0.0
    return best_offset, 1.0 - best_err / Lp


# ── Frame descriptor / sync ─────────────────────────────────────────────────

@dataclass
class FrameDescriptor:
    """A concrete frame-format candidate."""
    name: str
    preamble: list[int]          # known sync word (bits)
    header_bits: int = 0         # candidate header length in bits (0 = none)
    payload_symbols: int = 0     # candidate payload length in 8-bit symbols
    crc_variant: str = CRC16_IBM
    crc_at_end: bool = True      # CRC appended after payload (vs in header)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "preamble": self.preamble,
            "header_bits": self.header_bits,
            "payload_symbols": self.payload_symbols,
            "crc_variant": self.crc_variant,
            "crc_at_end": self.crc_at_end,
        }


@dataclass
class FrameSyncResult:
    """Result of attempting to sync a frame descriptor onto a bit stream."""
    found: bool
    descriptor_name: str
    offset: Optional[int] = None
    correlation: float = 0.0
    header_bits: Optional[list[int]] = None
    payload_bytes: Optional[bytes] = None
    payload_length: Optional[int] = None
    crc_valid: Optional[bool] = None
    crc_value: Optional[int] = None
    why: str = ""

    def to_dict(self) -> dict:
        return {
            "found": self.found,
            "descriptor": self.descriptor_name,
            "offset": self.offset,
            "correlation": round(float(self.correlation), 3),
            "header_bits": self.header_bits[:64] if self.header_bits is not None else None,
            "payload_bytes": (
                self.payload_bytes[:64].hex() if self.payload_bytes is not None else None
            ),
            "payload_length": self.payload_length,
            "crc_valid": self.crc_valid,
            "crc_value": self.crc_value,
            "why": self.why,
        }


def _parse_frame_at(bits, descriptor, start):
    """Parse a header (optional) + payload region + trailing CRC at `start`.

    Returns ``(header_bits, payload_bytes, payload_length, crc_value, crc_valid,
    why)`` or ``None`` if the bit stream does not contain the full region.
    """
    n_header = int(descriptor.header_bits)
    header_bits = bits[start:start + n_header].tolist() if n_header > 0 else []
    payload_start = start + n_header
    n_payload_bytes = int(descriptor.payload_symbols)
    payload_region = bits[payload_start:payload_start + n_payload_bytes * 8]
    if len(payload_region) < n_payload_bytes * 8:
        return None
    payload_bytes = bits_to_bytes(payload_region).tobytes()

    crc_width = {CRC32_IEEE: 4}.get(descriptor.crc_variant, 2)
    crc_region = bits[payload_start + n_payload_bytes * 8:
                      payload_start + n_payload_bytes * 8 + crc_width * 8]
    if len(crc_region) < crc_width * 8:
        return header_bits, payload_bytes, n_payload_bytes, None, None, "CRC field truncated"

    crc_raw = bits_to_bytes(crc_region).tobytes()
    crc_value = int(crc_raw.hex(), 16)
    ok = crc_check(payload_bytes, crc_value, descriptor.crc_variant, expected_lsb=False) or \
        crc_check(payload_bytes, int.from_bytes(crc_raw, "little"), descriptor.crc_variant, expected_lsb=True)
    crc_var = f"CRC-{16 if crc_width == 2 else 32}"
    why = f"payload {n_payload_bytes} B, {crc_var} {'OK' if ok else 'MISMATCH'}."
    return header_bits, payload_bytes, n_payload_bytes, crc_value, ok, why


def frame_sync(
    bits: np.ndarray,
    descriptor: FrameDescriptor,
    min_correlation: float = 0.75,
    search_length: Optional[int] = None,
    payload_lengths: Optional[Sequence[int]] = None,
) -> FrameSyncResult:
    """Search a bit stream for `descriptor.preamble` and parse a candidate frame.

    Args:
        bits: demodulated bit stream (0/1).
        descriptor: frame-format candidate.
        min_correlation: minimum preamble match to treat as a lock. Defaults to
            0.75. For short preambles (<= 16 bits) the threshold is automatically
            raised to 0.85 to reduce spurious perfect matches on random bit
            streams (F5 stronger frame locking).
        search_length: cap the number of candidate offsets searched.
        payload_lengths: candidate payload byte lengths to verify via CRC. When
            provided, the sync performs a CRC-assisted length scan over the
            preamble-locked region and prefers the first length that yields a
            valid CRC (payload length is part of the unknown at RX).

    Returns:
        A FrameSyncResult with a ``why`` explaining the outcome. When preamble
        correlation is 1.00 but CRC fails, the ``why`` is annotated with a
        "Potential false lock" warning (pitfall X6).
    """
    bits = np.asarray(bits, dtype=np.int64).ravel()

    # F5: Tighten threshold automatically for short preambles to reduce
    # spurious perfect-correlation false locks on noise (X6 pitfall).
    effective_min_corr = min_correlation
    preamble_len = len(descriptor.preamble) if descriptor.preamble else 0
    if preamble_len < 16:
        effective_min_corr = max(effective_min_corr, 0.85)

    offset, corr = correlate_preamble(bits, descriptor.preamble, search_length)
    if offset is None or corr < effective_min_corr:
        return FrameSyncResult(
            found=False,
            descriptor_name=descriptor.name,
            offset=offset,
            correlation=corr,
            why=(f"Preamble correlation {corr:.3f} below threshold {effective_min_corr:.2f} "
                 f"(preamble_len={preamble_len}); not frame-locked."),
        )

    lengths = list(payload_lengths) if payload_lengths is not None else [int(descriptor.payload_symbols)]
    base_why = (f"Preamble locked at bit {offset} (corr {corr:.3f}); "
                f"payload length scan {lengths[0]}..{lengths[-1]} B.")

    best: Optional[FrameSyncResult] = None
    for L in lengths:
        candidate = FrameDescriptor(
            name=descriptor.name,
            preamble=descriptor.preamble,
            header_bits=descriptor.header_bits,
            payload_symbols=L,
            crc_variant=descriptor.crc_variant,
            crc_at_end=descriptor.crc_at_end,
        )
        parsed = _parse_frame_at(bits, candidate, offset + len(descriptor.preamble))
        if parsed is None:
            continue
        header_bits, payload_bytes, n_payload, crc_value, crc_ok, why = parsed

        # F5: Annotate potential false lock when corr is perfect but CRC fails
        # (pitfall X6: a 1.000 correlation on a short preamble is not reliable).
        false_lock_note = ""
        if corr >= 0.999 and not crc_ok:
            false_lock_note = " [WARN: corr=1.000 + CRC fail -> potential false lock (X6)]"

        res = FrameSyncResult(
            found=True,
            descriptor_name=descriptor.name,
            offset=offset,
            correlation=corr,
            header_bits=header_bits,
            payload_bytes=payload_bytes,
            payload_length=n_payload,
            crc_valid=crc_ok,
            crc_value=crc_value,
            why=f"Frame locked at bit {offset} (preamble corr {corr:.3f}); {why}{false_lock_note}",
        )
        if crc_ok:
            return res
        if best is None:
            best = res
    if best is not None:
        best.why = base_why.rstrip(";") + "; " + best.why + " (no length CRC-verified)"
        return best
    return FrameSyncResult(
        found=True,
        descriptor_name=descriptor.name,
        offset=offset,
        correlation=corr,
        why=base_why,
    )


def default_frame_descriptors() -> list[FrameDescriptor]:
    """Constrained default frame-format candidates scored during decode search.

    A small set of practical satellite/terrestrial frame assumptions:
      - no header, no FEC payload = pure CRC-16/32 and preamble variants
      - optional 16-bit header + 100-byte payload with CRC-32
    These are *starting points*; the hypothesis engine widens per modulation.
    """
    return [
        FrameDescriptor(name="preamble+crc16", preamble=[0, 1, 1, 1, 1, 1, 1, 0] * 2,
                        header_bits=0, payload_symbols=100, crc_variant=CRC16_IBM),
        FrameDescriptor(name="preamble+crc32", preamble=[0, 1, 1, 1, 1, 1, 1, 0] * 2,
                        header_bits=0, payload_symbols=100, crc_variant=CRC32_IEEE),
        FrameDescriptor(name="preamble+header+crc32", preamble=[0, 1, 1, 1, 1, 1, 1, 0] * 2,
                        header_bits=16, payload_symbols=100, crc_variant=CRC32_IEEE),
    ]