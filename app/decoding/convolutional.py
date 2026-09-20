"""
app/decoding/convolutional.py
------------------------------
Convolutional FEC: **(k=7, rate 1/2)** encoder + Viterbi decoder.

This is the classic constraint-length-7, rate-1/2 convolutional code
(polynomials 0b1111001 = 171_octal and 0b1011011 = 133_octal), the industry
default for satellite and deep-space links. It is the most common FEC primitive
this reasoning assistant needs to recover.

Pure-NumPy exhaustive-trellis Viterbi decoder that accepts BOTH:
  - **hard decision** bits   (0/1)   -> Hamming branch metric
  - **soft decision** LLRs   (float) -> Euclidean/log branch metric

This mirrors how `DemodulationResult` carries both `bits` (hard) and
`llrs_sample` (soft) so the decoder can exploit channel soft-information.

Scope note (constrained candidate philosophy): only rate 1/2 with k=7 and the
canonical polynomial pair is implemented. Other rates/polynomials would be
added as separate profiles; arbitrary blind FEC identification is out of scope.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# Canonical constraint-length-7 rate-1/2 polynomials (171_octal, 133_octal)
G1 = 0b1111001          # 171 octal
G2 = 0b1011011          # 133 octal
CONSTRAINT_LENGTH = 7
NUM_STATES = 1 << (CONSTRAINT_LENGTH - 1)   # 64 states


def _parity(v: int, poly: int) -> int:
    """Parity of the bitwise AND of v with poly."""
    acc = 0
    p = poly
    while p:
        acc ^= (v & 1) & (p & 1)
        v >>= 1
        p >>= 1
    return acc


def _build_tables():
    """Precompute next-state / output tables for the (7, 1/2) code.

    Encoder: shift register holds the last K-1 info bits. On input bit `b`,
    the register becomes reg = (state << 1) | b (MSB-first convolution), and
    the two parity outputs are the parency of reg against G1 / G2. The new
    state is reg & ((1<<(K-1))-1).
    """
    n_states = NUM_STATES
    reg_mask = (1 << (CONSTRAINT_LENGTH - 1)) - 1
    next_state = np.zeros((n_states, 2), dtype=np.int64)
    output1 = np.zeros((n_states, 2), dtype=np.int64)
    output2 = np.zeros((n_states, 2), dtype=np.int64)
    for state in range(n_states):
        for bit in (0, 1):
            reg = (state << 1) | bit
            next_state[state, bit] = reg & reg_mask
            output1[state, bit] = _parity(reg, G1)
            output2[state, bit] = _parity(reg, G2)
    return next_state, output1, output2, reg_mask


_NEXT_STATE, _OUT1, _OUT2, _REG_MASK = _build_tables()

# Butterfly predecessor tables for the vectorized ACS (add-compare-select).
# Each destination state d is reached from exactly two sources with the SAME
# input bit:
#     pred0 = d >> 1            (upper source half, lower trellis level)
#     pred1 = (d >> 1) | (1 << (K-2))
#     bit = d & 1
# Because pred0 is always visited before pred1 in the scalar sweep, ties on
# equal metric resolve to pred0 — matches the original strictly-less behavior.
_STATE_COUNT = NUM_STATES
_DEST_INDEX = np.arange(_STATE_COUNT, dtype=np.int64)
_PRED0 = _DEST_INDEX >> 1
_PRED1 = _DEST_INDEX >> 1 | (1 << (CONSTRAINT_LENGTH - 2))
_DEST_BIT = _DEST_INDEX & 1


@dataclass
class ConvCodeParams:
    """Describes one convolutional-FEC profile (only the canonical one by default)."""
    name: str = "conv_r12_k7"
    constraint_length: int = CONSTRAINT_LENGTH
    rate_numerator: int = 1
    rate_denominator: int = 2
    g1: int = G1
    g2: int = G2

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "constraint_length": self.constraint_length,
            "rate": f"{self.rate_numerator}/{self.rate_denominator}",
            "g1_octal": f"{self.g1:o}",
            "g2_octal": f"{self.g2:o}",
        }


def encode_convolutional(bits: np.ndarray, tail_bits: int = 6) -> np.ndarray:
    """Convolutionally encode a bit array at rate 1/2.

    Args:
        bits: 1-D uint8/int array of payload bits to encode.
        tail_bits: number of forced-zero tail bits appended to flush the shift
                   register back to the all-zero state (k-1 = 6).

    Returns:
        1-D int array of length 2 * (len(bits) + tail_bits): the two parity
        streams (G1, G2) interleaved per info bit.
    """
    seq = np.asarray(bits, dtype=np.int64).ravel()
    info = np.concatenate([seq, np.zeros(tail_bits, dtype=np.int64)])
    out1 = np.empty(len(info), dtype=np.int64)
    out2 = np.empty(len(info), dtype=np.int64)
    state = 0
    for i, b in enumerate(info):
        b_int = int(b)
        out1[i] = int(_OUT1[state, b_int])
        out2[i] = int(_OUT2[state, b_int])
        state = int(_NEXT_STATE[state, b_int])
    coded = np.empty(len(info) * 2, dtype=np.int64)
    coded[0::2] = out1
    coded[1::2] = out2
    return coded


def decode_viterbi(
    received: np.ndarray,
    input_is_llr: bool = False,
    tail_bits: int = 6,
) -> np.ndarray:
    """Viterbi-decode a rate-1/2 (k=7) convolutional stream.

    Args:
        received: flat array of 2 * msg_len values. If `input_is_llr` is False
                  these are hard 0/1 parity bits; if True they are soft LLRs
                  (positive -> more likely bit 0, matching `compute_soft_llr`).
        input_is_llr: treat `received` as soft LLRs rather than hard bits.
        tail_bits: number of trailing encoder zeros (k-1) used to flush the
                   register to the all-zero state. Their decoded info bits are
                   dropped from the returned sequence.

    Returns:
        1-D decoded message bits (int array), length = len/2 - tail_bits.
    """
    received = np.asarray(received, dtype=np.float64).ravel()
    if received.size % 2 != 0:
        raise ValueError("Convolutional stream length must be a multiple of 2 (rate 1/2).")
    nbits = received.size // 2
    if nbits == 0:
        return np.zeros(0, dtype=np.uint8)

    n_states = NUM_STATES
    state_metric = np.full(n_states, np.inf)
    state_metric[0] = 0.0
    # prev_state[k, dest], prev_bit[k, dest]
    prev_state = np.zeros((nbits, n_states), dtype=np.int64)
    prev_bit = np.zeros((nbits, n_states), dtype=np.int8)

    pred0 = _PRED0
    pred1 = _PRED1
    bit_idx = _DEST_BIT
    out1 = _OUT1
    out2 = _OUT2

    for k in range(nbits):
        p0, p1 = received[2 * k], received[2 * k + 1]
        if not input_is_llr:
            bm = (out1 != p0).astype(np.float64) + (out2 != p1).astype(np.float64)
        else:
            bm = -((1.0 - 2.0 * out1) * p0 + (1.0 - 2.0 * out2) * p1)
        # Add-compare-select (butterfly) — fully vectorized over the 64 states.
        cost0 = state_metric[pred0] + bm[pred0, bit_idx]
        cost1 = state_metric[pred1] + bm[pred1, bit_idx]
        take0 = cost0 <= cost1      # ties resolve to pred0 (scalar-sweep order)
        state_metric = np.where(take0, cost0, cost1)
        prev_state[k] = np.where(take0, pred0, pred1)
        # Both predecessors reach dest d with the SAME input bit d & 1 — the
        # decoded bit does not depend on which predecessor won.
        prev_bit[k] = bit_idx

    # Traceback from end (all-zero state due to tail bits)
    decoded_sequence = np.zeros(nbits, dtype=np.uint8)
    state = 0
    for k in reversed(range(nbits)):
        decoded_sequence[k] = prev_bit[k, state]
        state = prev_state[k, state]
    # drop tail bits
    return decoded_sequence[:nbits - tail_bits]


# Backwards-compatible alias mirroring `demodulation` naming style
viterbi_decode = decode_viterbi