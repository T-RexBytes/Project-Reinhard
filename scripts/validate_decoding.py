#!/usr/bin/env python3
"""Offline decoding validation harness.

Self-contained synthetic ground truth for the decode stage:

  Regime 1  Viterbi hard-decision FEC corrects isolated bit errors
  Regime 2  Viterbi soft-decision (LLR) beats hard at low SNR
  Regime 3  Reed-Solomon (255,223 / 255,239) corrects t=16/8 symbol errors
  Regime 4  Frame sync finds the preamble in a noisy/garbage bit stream
            and verifies CRC-16 / CRC-32 payload integrity
  Regime 5  End-to-end chain (payload->CRC->convolve->block-interleave
            ->burst errors ->deinterleave->Viterbi->frame/CRC) recovers the
            exact payload, and `search_decode` ranks the true chain #1
  Regime 6  Honest failure: un-correctable error load must NOT produce a
            CRC-valid payload

Exit code 0 = PASS, 1 = FAIL. No network or dataset access required.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from app.decoding.interleaver import none_spec, block_spec, interleave, deinterleave
from app.decoding.convolutional import encode_convolutional, decode_viterbi
from app.decoding.reed_solomon import RS255_223, RS255_239, rs_encode, rs_decode
from app.decoding.frame import (
    FrameDescriptor,
    crc16,
    crc32,
    crc_bytes,
    crc_check,
    frame_sync,
    bytes_to_bits,
    CRC16_IBM,
    CRC32_IEEE,
)
from app.decoding.decoder import (
    DecodingSpec,
    FEC_VITERBI,
    decode_bits,
    search_decode,
)

PASS = 0
FAIL = 1
_results = []

PREAMBLE = [0, 1, 1, 1, 1, 1, 1, 0] * 2


def record(name: str, ok: bool, detail: str) -> None:
    _results.append((name, ok, detail))
    print(f"  [{('PASS' if ok else 'FAIL'):4}] {name}: {detail}")


def noisy_bits(bits: np.ndarray, p_err: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flip = rng.random(len(bits)) < p_err
    out = bits.copy()
    out[flip] = 1 - out[flip]
    return out


def build_coded_frame(payload: bytes, crc_variant=CRC16_IBM, ilv=None):
    """Build the canonical TX stream: preamble|payload|CRC -> conv -> interleave.

    Returns (interleaved_coded, preamble, interleaver_spec).
    """
    frame_bits = np.concatenate([
        np.array(PREAMBLE, dtype=np.int64),
        bytes_to_bits(payload),
        bytes_to_bits(crc_bytes(payload, crc_variant)),
    ])
    coded = encode_convolutional(frame_bits, tail_bits=6)
    ilv = ilv or block_spec(int(len(coded) // 10), 10) if (len(coded) % 10 == 0) and (len(coded) // 10) >= 1 else block_spec(2, 5)
    if len(coded) % ilv.size != 0:
        pad = (-len(coded)) % ilv.size
        coded = np.concatenate([coded, np.zeros(pad, dtype=np.int64)])
    return interleave(coded, ilv), PREAMBLE, ilv


def main() -> int:
    print("=" * 72)
    print("  OFFLINE DECODING VALIDATION (synthetic ground truth)")
    print("=" * 72)
    rng = np.random.default_rng(42)

    # ── Regime 1: Viterbi hard-decision FEC ──────────────────────────────────
    print("\nRegime 1: Viterbi hard-decision corrects isolated bit errors")
    msg = rng.integers(0, 2, 300)
    coded = encode_convolutional(msg)
    for trial, err_pos in enumerate(([5, 300], [11, 12, 500, 501], [40, 41, 42, 500, 501, 502])):
        rx = coded.copy()
        for p in err_pos:
            rx[p] = 1 - rx[p]
        dec = decode_viterbi(rx)
        ok = np.array_equal(dec, msg)
        record(f"trial {trial+1}: {len(err_pos)} flipped coded bits", ok,
               f"len(coded)={len(coded)} | decoded==msg: {np.array_equal(dec, msg)}")

    # ── Regime 2: soft-decision LLR gains at low SNR ─────────────────────────
    print("\nRegime 2: soft-decision Viterbi (LLR) vs hard at low SNR")
    msg2 = rng.integers(0, 2, 400)
    coded2 = encode_convolutional(msg2)
    llr_rx = 1.8 * (1.0 - 2.0 * coded2.astype(float)) + 1.6 * rng.standard_normal(len(coded2))
    hard_rx = (llr_rx > 0).astype(np.int64)
    dec_hard = decode_viterbi(hard_rx)
    dec_soft = decode_viterbi(llr_rx, input_is_llr=True)
    err_hard = int(np.count_nonzero(dec_hard != msg2))
    err_soft = int(np.count_nonzero(dec_soft != msg2))
    record("hard vs soft residual bit errors", err_soft <= err_hard,
           f"hard_errors={err_hard} soft_errors={err_soft} (soft {'' if err_soft <= err_hard else 'NOT '}better)")

    # ── Regime 3: Reed-Solomon t-error correction ────────────────────────────
    print("\nRegime 3: Reed-Solomon symbol-error correction")
    for params in (RS255_223, RS255_239):
        mp = rng.integers(0, 256, params.k)
        cw = rs_encode(mp, params)
        err = cw.copy()
        n_e = params.t
        idx = rng.choice(params.n, size=n_e, replace=False)
        for i, j in enumerate(idx):
            err[j] ^= (i * 5 + 1) & 0xFF
        dec, status = rs_decode(err, params)
        ok = status == n_e and np.array_equal(dec, mp)
        record(f"RS({params.n},{params.k}) correct {n_e} errors", ok,
               f"status={status} | msg==decoded: {np.array_equal(dec, mp)}")

    # ── Regime 4: frame sync + CRC ───────────────────────────────────────────
    print("\nRegime 4: frame sync & CRC integrity in garbage")
    payload_b = b"IQ 40960 samples SAM splice y 50185"
    frame_bits = np.concatenate([
        np.array(PREAMBLE, dtype=np.int64),
        bytes_to_bits(payload_b),
        bytes_to_bits(crc_bytes(payload_b, CRC16_IBM)),
    ])
    noise = rng.integers(0, 2, 37)
    stream = np.concatenate([noise, frame_bits])
    desc16 = FrameDescriptor(name="crc16", preamble=PREAMBLE, payload_symbols=len(payload_b),
                             crc_variant=CRC16_IBM)
    res16 = frame_sync(stream, desc16)
    record("frame_sync offset + CRC-16 validation", res16.found and res16.offset == 37 and res16.crc_valid is True,
           f"offset={res16.offset} corr={res16.correlation:.2f} crc_valid={res16.crc_valid}")

    payload_c = b"crc32 variant payload (61.44 MS/s)"
    frame_bits32 = np.concatenate([
        np.array(PREAMBLE, dtype=np.int64),
        bytes_to_bits(payload_c),
        bytes_to_bits(crc_bytes(payload_c, CRC32_IEEE)),
    ])
    desc32 = FrameDescriptor(name="crc32", preamble=PREAMBLE, payload_symbols=len(payload_c),
                             crc_variant=CRC32_IEEE)
    res32 = frame_sync(frame_bits32, desc32)
    record("frame_sync CRC-32 validation", res32.crc_valid is True and res32.payload_bytes == payload_c,
           f"corr={res32.correlation:.2f} crc_valid={res32.crc_valid}")

    # ── Regime 5: end-to-end chain + search ranking ──────────────────────────
    print("\nRegime 5: end-to-end decode chain and candidate ranking")
    e2e_payload = (b"SIH26147-decode-e2e-" * 3)[:36]
    assert 16 <= len(e2e_payload) <= 248
    coded_frame, pre, ilv = build_coded_frame(e2e_payload, CRC16_IBM, ilv=block_spec(2, 5))
    rx_burst = coded_frame.copy()
    for i in range(12, 30):   # burst of 18 consecutive bit flips
        rx_burst[i] = 1 - rx_burst[i]
    spec_truth = DecodingSpec(
        name="viterbi|block(2,5)|crc16",
        deinterleaver=ilv, fec=FEC_VITERBI,
        frame_descriptor=FrameDescriptor(name="crc16", preamble=pre,
                                         payload_symbols=len(e2e_payload), crc_variant=CRC16_IBM),
    )
    dec = decode_bits(rx_burst, spec_truth)
    record("full chain recovers exact payload", dec.ok and dec.payload_bytes == e2e_payload,
           f"ok={dec.ok} crc_valid={dec.crc_valid} payload==tx: {dec.payload_bytes == e2e_payload}")

    ranked = search_decode(rx_burst, top_k=5)
    top = ranked[0]
    record("search_decode ranks true chain #1", top.crc_valid is True and top.payload_bytes == e2e_payload,
           f"top={top.spec_name} crc_valid={top.crc_valid} corr={top.preamble_correlation:.2f}")

    # ── Regime 6: honest failure under excess error load ─────────────────────
    print("\nRegime 6: un-correctable error load fails honestly")
    heavy_coded, pre6, ilv6 = build_coded_frame(e2e_payload, CRC16_IBM, ilv=block_spec(2, 5))
    # destroy ~40% of coded bits (blew past FEC limit)
    heavy = noisy_bits(heavy_coded, p_err=0.4, seed=5)
    dec_heavy = decode_bits(heavy, spec_truth)
    record("excess errors -> no CRC-valid payload", dec_heavy.crc_valid is not True,
           f"crc_valid={dec_heavy.crc_valid} ok={dec_heavy.ok}")

    # ── Summary ──────────────────────────────────────────────────────────────
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print("\n" + "=" * 72)
    print(f"  SUMMARY: {passed}/{total} checks PASSED")
    print("=" * 72)
    if passed != total:
        for name, ok, detail in _results:
            if not ok:
                print(f"    FAILED: {name} -> {detail}")
        return FAIL
    return PASS


if __name__ == "__main__":
    sys.exit(main())