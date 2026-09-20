"""
app/decoding
------------
De-interleaving, FEC decoding, and frame synchronization.

This package implements the constrained candidate decode chain that follows the
demodulation stage:

    deinterleave (none | block N,M)
      -> Viterbi (k=7, rate 1/2) optional
      -> Reed-Solomon (255, 223 | 239) optional
      -> frame sync + CRC-16/CRC-32

Submodules
    interleaver.py    block interleaver / deinterleaver + candidate grid
    convolutional.py  rate-1/2 constraint-7 convolutional encoder + Viterbi
    reed_solomon.py   RS(255,223) / RS(255,239) GF(256) codec
    frame.py          preamble sync, CRC-16/32, frame-boundary detection
    decoder.py        chain orchestration + ranked candidate search
"""

from app.decoding.interleaver import (
    InterleaverSpec,
    InterleaverType,
    block_spec,
    none_spec,
    interleave,
    deinterleave,
    block_candidate_grid,
    default_candidate_specs,
)
from app.decoding.convolutional import (
    encode_convolutional,
    decode_viterbi,
    viterbi_decode,
    G1,
    G2,
    CONSTRAINT_LENGTH,
    NUM_STATES,
    ConvCodeParams,
)
from app.decoding.reed_solomon import (
    RSParams,
    RS255_223,
    RS255_239,
    rs_encode,
    rs_decode,
    encode_rs,
    decode_rs,
    gf_mul,
    gf_inv,
    gf_poly_eval,
)
from app.decoding.frame import (
    FrameDescriptor,
    FrameSyncResult,
    crc16,
    crc32,
    crc_bytes,
    crc_check,
    correlate_preamble,
    frame_sync,
    bits_to_bytes,
    bytes_to_bits,
    CRC16_IBM,
    CRC16_CCITT,
    CRC32_IEEE,
    default_frame_descriptors,
)
from app.decoding.decoder import (
    DecodingSpec,
    DecodingResult,
    FEC_NONE,
    FEC_VITERBI,
    FEC_RS223,
    FEC_RS239,
    FEC_VITERBI_RS223,
    FEC_VITERBI_RS239,
    decode_bits,
    search_decode,
    make_default_specs,
    append_decoding_provenance,
)

__all__ = [
    # interleaver
    "InterleaverSpec", "InterleaverType", "block_spec", "none_spec",
    "interleave", "deinterleave", "block_candidate_grid", "default_candidate_specs",
    # convolutional
    "encode_convolutional", "decode_viterbi", "viterbi_decode",
    "G1", "G2", "CONSTRAINT_LENGTH", "NUM_STATES", "ConvCodeParams",
    # reed-solomon
    "RSParams", "RS255_223", "RS255_239", "rs_encode", "rs_decode",
    "encode_rs", "decode_rs", "gf_mul", "gf_inv", "gf_poly_eval",
    # frame
    "FrameDescriptor", "FrameSyncResult", "crc16", "crc32", "crc_bytes",
    "crc_check", "correlate_preamble", "frame_sync", "bits_to_bytes",
    "bytes_to_bits", "CRC16_IBM", "CRC16_CCITT", "CRC32_IEEE",
    "default_frame_descriptors",
    # decoder orchestration
    "DecodingSpec", "DecodingResult", "FEC_NONE", "FEC_VITERBI",
    "FEC_RS223", "FEC_RS239", "FEC_VITERBI_RS223", "FEC_VITERBI_RS239",
    "decode_bits", "search_decode", "make_default_specs",
    "append_decoding_provenance",
]