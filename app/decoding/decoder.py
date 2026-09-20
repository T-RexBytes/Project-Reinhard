"""
app/decoding/decoder.py
-----------------------
Orchestration of the decode chain: de-interleave -> FEC -> frame/CRC.

Given the demodulated hard bits and/or soft LLRs, this module runs one or more
**constrained candidate chains** and returns a `DecodingResult` per candidate:

    deinterleave (none | block N,M)   ->   viterbi (k=7, r=1/2) optional
    ->  Reed-Solomon (255,223 / 239) optional   ->   frame sync + CRC-16/32

`search_decode` enumerates the full candidate grid (bounded by philosophy of
the project) and returns *ranked* chains so the hypothesis engine can fuse
with SNR / EVM / HOC confidence later.

Provenance / explainability: every result carries a human-readable `why`.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

from app.decoding.interleaver import (
    InterleaverSpec,
    none_spec,
    block_spec,
    default_candidate_specs,
    deinterleave,
)
from app.decoding.convolutional import encode_convolutional, decode_viterbi
from app.decoding.reed_solomon import (
    RSParams,
    RS255_223,
    RS255_239,
    rs_decode,
)
from app.decoding.frame import (
    FrameDescriptor,
    FrameSyncResult,
    frame_sync,
    bits_to_bytes,
    bytes_to_bits,
    CRC16_IBM,
    CRC32_IEEE,
)
from app.models import ProvenanceStep

# FEC candidate modes of the decoding chain
FEC_NONE = "none"
FEC_VITERBI = "viterbi"          # constraint-7 rate-1/2 convolutional
FEC_RS223 = "rs255_223"
FEC_RS239 = "rs255_239"
FEC_VITERBI_RS223 = "viterbi+rs255_223"
FEC_VITERBI_RS239 = "viterbi+rs255_239"
FEC_MODES = (FEC_NONE, FEC_VITERBI, FEC_RS223, FEC_RS239,
             FEC_VITERBI_RS223, FEC_VITERBI_RS239)

# Payload byte lengths swept during CRC-assisted frame-length discovery.
# Bounded by the constrained-candidate philosophy; the hypothesis engine widens
# this set when the analyst supplies a hint.
DEFAULT_PAYLOAD_LENGTHS = (
    16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64,
    72, 80, 88, 96, 100, 112, 128, 144, 160, 192, 200, 224, 248,
)


@dataclass
class DecodingSpec:
    """One fully-resolved decode-chain candidate."""
    name: str
    deinterleaver: InterleaverSpec = field(default_factory=none_spec)
    fec: str = FEC_NONE                       # FEC mode string
    frame_descriptor: Optional[FrameDescriptor] = None
    apply_viterbi: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "deinterleaver": str(self.deinterleaver),
            "fec": self.fec,
            "frame": self.frame_descriptor.to_dict() if self.frame_descriptor else None,
        }


@dataclass
class DecodingResult:
    """Result of decoding a bit (or LLR) stream with one chain candidate."""
    spec_name: str
    chain: str                        # human-readable chain description
    ok: bool = False
    crc_valid: Optional[bool] = None
    payload_bits: list[int] = field(default_factory=list)
    payload_bytes: Optional[bytes] = None
    num_viterbi_errors: int = 0       # post-decoder residual hard bit errors
    num_rs_symbols_corrected: int = 0
    preamble_correlation: float = 0.0
    frame_offset: Optional[int] = None
    used_soft: bool = False
    fec: str = FEC_NONE
    why: str = ""

    def to_dict(self) -> dict:
        return {
            "spec_name": self.spec_name,
            "chain": self.chain,
            "ok": bool(self.ok),
            "crc_valid": self.crc_valid,
            "payload_bits_sample": [int(b) for b in self.payload_bits[:256]],
            "payload_bytes_hex": (
                self.payload_bytes[:64].hex() if self.payload_bytes else None
            ),
            "num_viterbi_errors": int(self.num_viterbi_errors),
            "num_rs_symbols_corrected": int(self.num_rs_symbols_corrected),
            "preamble_correlation": round(float(self.preamble_correlation), 3),
            "frame_offset": int(self.frame_offset) if self.frame_offset is not None else None,
            "used_soft": bool(self.used_soft),
            "why": self.why,
        }


def _frame_descriptor(name: str) -> Optional[FrameDescriptor]:
    """Build the canonical frame descriptor for a candidate frame name."""
    pre = [0, 1, 1, 1, 1, 1, 1, 0] * 2
    if name == "preamble+crc16":
        return FrameDescriptor(name="preamble+crc16", preamble=pre,
                               header_bits=0, payload_symbols=100, crc_variant=CRC16_IBM)
    if name == "preamble+crc32":
        return FrameDescriptor(name="preamble+crc32", preamble=pre,
                               header_bits=0, payload_symbols=100, crc_variant=CRC32_IEEE)
    return None


# ── THE shared decode-chain candidate grid ───────────────────────────────────
# One source of truth for the decode search space. The standalone leaderboard
# (`make_default_specs`, `/api/decode`) and the hypothesis engine's group decode
# search both consume this same grid, so their answers are directly comparable
# (identical spec names, identical ranking by `search_decode`).
DEFAULT_FRAME_NAMES = ("preamble+crc16", "preamble+crc32")


def build_decode_spec_grid(
    interleaver_specs: Optional[list[InterleaverSpec]] = None,
    fec_modes: Optional[Sequence[str]] = None,
    frame_names: Optional[Sequence[str]] = DEFAULT_FRAME_NAMES,
    burst_bits: Optional[int] = None,
) -> list[DecodingSpec]:
    """Build the canonical ``DecodingSpec`` candidate table.

    Args:
        interleaver_specs: deinterleaver candidates. Defaults to NONE + the
            bounded BLOCK grid from ``default_candidate_specs()``.
        fec_modes: FEC mode strings. Defaults to the full ``FEC_MODES`` table.
        frame_names: frame/CRC candidates. Defaults to crc16 + crc32.
        burst_bits: when provided, block interleaver specs whose block size
            (N*M) exceeds ``burst_bits // 4`` are pruned from the grid. This
            prevents the search from wasting cycles on block sizes that require
            more than 4x the estimated burst length to form even one codeblock
            (F9 signal-scoped grid). Does not affect the NONE spec.

    Every spec uses the canonical ``<fec>|<interleaver>|<frame>`` naming so the
    hypothesis engine's ``_chain_to_decoding_spec`` entries map 1:1 onto this
    grid (same string form incl. ``block(4, 8)`` spacing).
    """
    if interleaver_specs is None:
        interleaver_specs = default_candidate_specs()
    if fec_modes is None:
        fec_modes = FEC_MODES
    frames = [f for f in (_frame_descriptor(n) for n in frame_names) if f is not None]

    # F9: prune block specs that are larger than the signal can support.
    if burst_bits is not None and burst_bits > 0:
        max_block_size = max(1, burst_bits // 4)
        filtered: list[InterleaverSpec] = []
        for ilv in interleaver_specs:
            if ilv.type == "block" and ilv.size > max_block_size:
                continue  # implausible given this burst length
            filtered.append(ilv)
        interleaver_specs = filtered if filtered else interleaver_specs

    specs: list[DecodingSpec] = []
    for ilv in interleaver_specs:
        for fec in fec_modes:
            for frame in frames:
                specs.append(DecodingSpec(
                    name=f"{fec}|{ilv}|{frame.name}",
                    deinterleaver=ilv,
                    fec=fec,
                    frame_descriptor=frame,
                    apply_viterbi="viterbi" in fec,
                ))
    return specs


def make_default_specs() -> list[DecodingSpec]:
    """Default decode candidates for the search (in priority order).

    Delegates to the single shared grid ``build_decode_spec_grid`` so the
    standalone leaderboard always searches the same space as the hypothesis
    engine's group-decode step (Bug B unification).
    """
    return build_decode_spec_grid()


def _count_residual_errors(decoded_bits: np.ndarray, coded: np.ndarray, ilv: InterleaverSpec) -> int:
    """Re-encode decoded bits and compare with the deinterleaved coded stream."""
    if len(decoded_bits) == 0:
        return 0
    reencoded = encode_convolutional(decoded_bits, tail_bits=0)
    short = len(reencoded)
    if short > len(coded):
        reencoded = reencoded[: len(coded)]
        short = len(coded)
    ref = coded[:short]
    if len(ref) != short:
        return 0
    return int(np.count_nonzero(ref != reencoded[:short]))


def _rs_from_bitstream(bits: np.ndarray, params: RSParams):
    """Try RS decode treating the bit stream as packed 8-bit symbols.

    Returns (message_bits, corrected_symbols, ok). The symbol stream is
    truncated to a whole number of RS codewords.
    """
    n = params.n
    symbol_values = bits_to_bytes(bits)
    n_codewords = len(symbol_values) // n
    if n_codewords == 0:
        return bits.copy(), 0, False
    usable = symbol_values[: n_codewords * n]
    corrected = 0
    decoded_symbols = []
    all_ok = True
    for cw_idx in range(n_codewords):
        cw = usable[cw_idx * n:(cw_idx + 1) * n]
        msg, status = rs_decode(cw, params)
        if status < 0:
            all_ok = False
        corrected += max(status, 0)
        decoded_symbols.append(msg)
    out_bytes = np.concatenate(decoded_symbols) if decoded_symbols else np.zeros(0, dtype=np.int64)
    out_bits = bytes_to_bits(out_bytes)
    return out_bits, corrected, all_ok


def decode_bits(
    bits: Union[np.ndarray, Sequence[int]],
    spec: Optional[DecodingSpec] = None,
    llrs: Optional[np.ndarray] = None,
    use_soft_viterbi: bool = True,
) -> DecodingResult:
    """Run one decode chain candidate over a bit/LLR stream.

    Args:
        bits: hard-demodulated bits (0/1) feeding the chain.
        spec: DecodingSpec. Defaults to a plain (none/viterbi, crc16) chain.
        llrs: optional per-bit LLRs (positive -> likely bit 0). When present
              and `use_soft_viterbi` is True, the Viterbi stage uses soft
              decisions.
        use_soft_viterbi: prefer the soft-LLR Viterbi path when llrs available.

    Returns:
        A DecodingResult whose `ok` reflects whether a valid payload with a
        matching frame CRC (or a clean FEC correct-on-first-try result) was
        recovered. `why` always explains the outcome.
    """
    if spec is None:
        spec = make_default_specs()[0]
    bits = np.asarray(bits, dtype=np.int64).ravel()
    n_bits = len(bits)

    # Block interleavers require stream length to be a multiple of N*M.
    if spec.deinterleaver.type == "block" and n_bits % spec.deinterleaver.size != 0:
        return DecodingResult(
            spec_name=spec.name,
            chain="deint(block)",
            ok=False,
            why=(f"Stream length {n_bits} is not a multiple of "
                 f"block size {spec.deinterleaver.size}; candidate not applicable."),
        )

    # 1. De-interleave.
    deint = deinterleave(bits, spec.deinterleaver).astype(np.int64)
    chain_parts = [f"deint({spec.deinterleaver})"]

    # 2. FEC decode.
    curr = deint
    used_soft = False
    num_conv_errors = 0
    viterbi_done = False

    if spec.apply_viterbi or "viterbi" in spec.fec:
        viterbi_done = True
        # Even-align deint to satisfy Viterbi rate-1/2 constraint
        if len(deint) % 2 != 0:
            deint = deint[:len(deint) - 1]
        has_llrs = llrs is not None and len(llrs) >= len(bits)
        if has_llrs:
            deint_llrs = np.asarray(deinterleave(np.asarray(llrs, dtype=np.float64).ravel(),
                                                 spec.deinterleaver), dtype=np.float64)
            if len(deint_llrs) % 2 != 0:
                deint_llrs = deint_llrs[:len(deint_llrs) - 1]
        try:
            if has_llrs and use_soft_viterbi:
                decoded = decode_viterbi(deint_llrs, input_is_llr=True, tail_bits=6)
                used_soft = True
            else:
                decoded = decode_viterbi(deint, input_is_llr=False, tail_bits=6)
        except ValueError as v_exc:
            return DecodingResult(
                spec_name=spec.name,
                chain="deint(block)+viterbi",
                ok=False,
                why=f"Viterbi stream length error: {v_exc}",
            )
        num_conv_errors = _count_residual_errors(decoded, deint, spec.deinterleaver)
        curr = decoded
        chain_parts.append("viterbi-r1/2-k7")
        fec_mode = FEC_VITERBI

    rs_corrected = 0
    if "rs255_223" in spec.fec or "rs255_239" in spec.fec:
        params = RS255_223 if "223" in spec.fec else RS255_239
        curr, rs_corrected, _all_ok = _rs_from_bitstream(curr, params)
        chain_parts.append(f"RS {params.n},{params.k} ({rs_corrected} sym corr)")
    if not viterbi_done and ("rs" not in spec.fec):
        fec_mode = FEC_NONE
    else:
        fec_mode = spec.fec

    # 3. Frame sync + CRC validation.
    frame_res: Optional[FrameSyncResult] = None
    if spec.frame_descriptor is not None:
        lengths = list(DEFAULT_PAYLOAD_LENGTHS)
        hinted = int(spec.frame_descriptor.payload_symbols)
        if hinted and hinted not in lengths:
            lengths.insert(0, hinted)
        frame_res = frame_sync(curr, spec.frame_descriptor, payload_lengths=lengths)
        chain_parts.append(f"frame/{spec.frame_descriptor.name}")
        crc_valid = bool(frame_res.crc_valid) if frame_res.crc_valid is not None else None
        payload_bytes = frame_res.payload_bytes
        payload_bits = list(frame_res.payload_bytes) if frame_res.payload_bytes is not None else list(curr)
    else:
        crc_valid = None
        payload_bytes = None
        payload_bits = list(curr)

    ok = (crc_valid is True) if frame_res is not None else (
        rs_corrected == 0 and num_conv_errors == 0 and n_bits > 0)

    bits_used = len(curr)
    why = (
        f"Chain [{spec.name}]: {bits_used} bits -> {','.join(chain_parts)}. "
        + (frame_res.why if frame_res else "No frame descriptor supplied; chain decoded to FEC layer."
           )
        + (f" CRC={'PASS' if crc_valid else 'FAIL'};" if crc_valid is not None else "")
        + f" viterbi residual hard-errs={num_conv_errors}; RS symbols corrected={rs_corrected}."
    )

    return DecodingResult(
        spec_name=spec.name,
        chain=" -> ".join(chain_parts),
        ok=ok,
        crc_valid=crc_valid,
        payload_bits=payload_bits,
        payload_bytes=payload_bytes,
        num_viterbi_errors=num_conv_errors,
        num_rs_symbols_corrected=rs_corrected,
        preamble_correlation=frame_res.correlation if frame_res else 0.0,
        frame_offset=frame_res.offset if frame_res else None,
        used_soft=used_soft,
        fec=fec_mode,
        why=why,
    )


def search_decode(
    bits: Union[np.ndarray, Sequence[int]],
    specs: Optional[list[DecodingSpec]] = None,
    llrs: Optional[np.ndarray] = None,
    use_soft_viterbi: bool = True,
    top_k: int = 10,
    burst_bits: Optional[int] = None,
) -> list[DecodingResult]:
    """Run the constrained candidate grid and return ranked decodes.

    Each spec runs sequentially through de-interleave -> FEC -> frame (with the
    CRC-assisted payload-length scan handled inside frame sync). Ranking
    prefers, in order: a CRC-valid frame, fewer residual Viterbi errors, more RS
    symbol corrections, higher preamble correlation, and a longer recovered
    payload.

    Args:
        bits            : Hard-demodulated bit stream.
        specs           : Explicit spec list. When None, the canonical grid is
                          built via ``make_default_specs()``.
        llrs            : Soft LLR values (optional).
        use_soft_viterbi: Prefer soft-decision Viterbi when LLRs are available.
        top_k           : Maximum number of results to return.
        burst_bits      : Estimated burst length in bits. When ``specs`` is None
                          (i.e. the default grid is used), this prunes block
                          interleaver specs that are implausibly large for the
                          signal (F9 signal-scoped grid). Ignored when ``specs``
                          is explicitly supplied.
    """
    if specs is None:
        n_bits_hint = burst_bits if burst_bits is not None else (len(bits) if len(bits) > 0 else None)
        specs = build_decode_spec_grid(burst_bits=n_bits_hint)
    if len(bits) == 0:
        return []
    results = [decode_bits(bits, sp, llrs=llrs, use_soft_viterbi=use_soft_viterbi)
               for sp in specs]

    def _rank(r: DecodingResult):
        crc_rank = 0 if r.crc_valid else (1 if r.crc_valid is None else 2)
        return (crc_rank, r.num_viterbi_errors, -r.num_rs_symbols_corrected,
                -r.preamble_correlation, -len(r.payload_bits))

    results.sort(key=_rank)
    return results[:top_k]


# Re-export the Provenance helper so upstream can append a decode step uniformly.
def append_decoding_provenance(signal, spec_name: str, ok: bool, crc_valid: bool, fec: str):
    """Append a provenance step describing a decoding attempt (no mutation of
    the signal bytes is performed)."""
    return ProvenanceStep(
        stage="decoding",
        operation="decode_chain",
        parameters={
            "spec": spec_name,
            "ok": bool(ok),
            "crc_valid": bool(crc_valid) if crc_valid is not None else None,
            "fec": fec,
        },
    )