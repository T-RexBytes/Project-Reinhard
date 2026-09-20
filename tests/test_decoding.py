"""Tests for the decoding stage: interleaving, Viterbi FEC, Reed-Solomon, frame sync."""

import numpy as np
import pytest

from app.decoding.interleaver import none_spec, block_spec, interleave, deinterleave, InterleaverType
from app.decoding.convolutional import (
    encode_convolutional,
    decode_viterbi,
    viterbi_decode,
    G1,
    G2,
    CONSTRAINT_LENGTH,
)
from app.decoding.reed_solomon import RS255_223, RS255_239, rs_encode, rs_decode
from app.decoding.frame import (
    FrameDescriptor,
    crc16,
    crc32,
    crc_bytes,
    crc_check,
    frame_sync,
    bits_to_bytes,
    bytes_to_bits,
    CRC16_IBM,
    CRC32_IEEE,
)
from app.decoding.decoder import (
    DecodingSpec,
    DecodingResult,
    FEC_NONE,
    FEC_VITERBI,
    decode_bits,
    search_decode,
)


class TestInterleaver:
    def test_none_round_trip(self):
        x = np.random.randint(0, 2, 100)
        assert np.array_equal(deinterleave(interleave(x, none_spec()), none_spec()), x)

    def test_block_round_trip(self):
        for n_rows, n_cols, length in ((4, 6, 24), (2, 5, 30), (4, 6, 96)):
            x = np.random.randint(0, 2, length)
            spec = block_spec(n_rows, n_cols)
            y = interleave(x, spec)
            assert np.array_equal(deinterleave(y, spec), x)

    def test_multi_block_round_trip(self):
        x = np.random.randint(0, 2, 96)  # 4 full blocks of 24
        y = interleave(x, block_spec(4, 6))
        assert np.array_equal(deinterleave(y, block_spec(4, 6)), x)

    def test_block_permutes(self):
        x = np.arange(12)
        spec = block_spec(3, 4)  # 3x4 matrix
        y = interleave(x, spec)
        # Write rows [[0,1,2,3],[4,5,6,7],[8,9,10,11]], read columns:
        assert np.array_equal(y, [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11])

    def test_block_requires_alignment(self):
        x = np.random.randint(0, 2, 25)
        with pytest.raises(ValueError):
            interleave(x, block_spec(4, 6))

    def test_type_constants(self):
        assert InterleaverType.NONE == "none"
        assert InterleaverType.BLOCK == "block"


class TestConvolutional:
    def test_parameters(self):
        assert CONSTRAINT_LENGTH == 7
        assert G1 == 0b1111001
        assert G2 == 0b1011011

    def test_hard_round_trip(self):
        rng = np.random.default_rng(0)
        bits = rng.integers(0, 2, 200)
        coded = encode_convolutional(bits)
        assert len(coded) == 2 * (200 + 6)
        decoded = decode_viterbi(coded)
        assert np.array_equal(decoded, bits)

    def test_soft_round_trip(self):
        rng = np.random.default_rng(1)
        bits = rng.integers(0, 2, 150)
        coded = encode_convolutional(bits)
        llrs = 2.0 * (1.0 - 2.0 * coded) + 0.2 * rng.standard_normal(len(coded))
        decoded = decode_viterbi(llrs, input_is_llr=True)
        assert np.array_equal(decoded, bits)

    def test_corrects_hard_errors(self):
        rng = np.random.default_rng(2)
        bits = rng.integers(0, 2, 200)
        coded = encode_convolutional(bits)
        corrupted = coded.copy()
        for i in (7, 8, 90, 91):
            corrupted[i] = 1 - corrupted[i]
        # k=7 r=1/2 free distance 10 => 4 well-spread errors decode fine
        decoded = viterbi_decode(corrupted)
        assert np.array_equal(decoded, bits)


class TestReedSolomon:
    def _roundtrip(self, params, n_err):
        rng = np.random.default_rng(7)
        msg = rng.integers(0, 256, params.k)
        cw = rs_encode(msg, params)
        assert len(cw) == params.n
        clean, status = rs_decode(cw, params)
        assert status == 0
        assert np.array_equal(clean, msg)
        err = cw.copy()
        idx = rng.choice(params.n, size=n_err, replace=False)
        for i, j in enumerate(idx):
            err[j] ^= (i + 1) & 0xFF
        dec, status = rs_decode(err, params)
        assert status == n_err
        assert np.array_equal(dec, msg)

    def test_rs255_223_clean_and_correct(self):
        self._roundtrip(RS255_223, 0)
        self._roundtrip(RS255_223, 3)
        self._roundtrip(RS255_223, 16)  # full t capability

    def test_rs255_239_clean_and_correct(self):
        self._roundtrip(RS255_239, 0)
        self._roundtrip(RS255_239, 8)

    def test_too_many_errors_fails_honestly(self):
        rng = np.random.default_rng(3)
        msg = rng.integers(0, 256, RS255_223.k)
        cw = rs_encode(msg, RS255_223)
        err = cw.copy()
        for i, j in enumerate(range(0, 200, 10)):
            err[j] ^= (i * 7 + 3) & 0xFF  # 20 errors > t=16
        _, status = rs_decode(err, RS255_223)
        assert status < 0


class TestFrameCRC:
    def test_crc16_known_values(self):
        assert crc16(b"123456789", CRC16_IBM) == 0x4B37  # CRC-16/IBM (MODBUS) check value
        assert crc16(b"123456789", "crc16_ccitt") == 0x29B1  # CRC-16/CCITT-FALSE check value

    def test_crc32_known_value(self):
        assert crc32(b"123456789") == 0xCBF43926  # CRC-32/IEEE check value

    def test_crc_check(self):
        d = b"payload for crc check"
        assert crc_check(d, crc16(d), CRC16_IBM)
        assert not crc_check(d, crc16(d) ^ 1, CRC16_IBM)
        assert crc_check(d, crc32(d), CRC32_IEEE)

    def test_crc_bytes_serialized_value(self):
        d = b"append me"
        # serialized CRC parsed back as a big-endian int must verify
        assert crc_check(d, int.from_bytes(crc_bytes(d, CRC16_IBM), "big"), CRC16_IBM)
        assert crc_check(d, int.from_bytes(crc_bytes(d, CRC32_IEEE), "big"), CRC32_IEEE)


class TestFrameSync:
    def _make_transmission(self, payload, crc_variant=CRC16_IBM, header_bits=0):
        pre = [0, 1, 1, 1, 1, 1, 1, 0] * 2
        bits = bytes_to_bits(payload)
        crc_bits = bytes_to_bits(crc_bytes(payload, crc_variant))
        hdr = np.zeros(header_bits, dtype=np.int64) if header_bits else np.zeros(0, dtype=np.int64)
        return np.concatenate([np.array(pre, dtype=np.int64), hdr, bits, crc_bits]), pre

    def test_sync_finds_offset_and_crc(self):
        payload = b"0123456789abcdef"
        tx, pre = self._make_transmission(payload)
        # embed in garbage
        noise = np.random.randint(0, 2, 40)
        stream = np.concatenate([noise, tx])
        desc = FrameDescriptor(name="test", preamble=pre, payload_symbols=len(payload),
                               crc_variant=CRC16_IBM)
        res = frame_sync(stream, desc)
        assert res.found and res.offset == 40
        assert res.crc_valid is True
        assert res.payload_bytes == payload

    def test_mismatch_reported(self):
        payload = b"0123456789abcdef"
        tx, pre = self._make_transmission(payload)
        bad = tx.copy()
        bad[30] = 1 - bad[30]  # corrupt a payload bit
        desc = FrameDescriptor(name="test", preamble=pre, payload_symbols=len(payload),
                               crc_variant=CRC16_IBM)
        res = frame_sync(bad, desc)
        assert res.found
        assert res.crc_valid is False

    def test_crc32_variant(self):
        payload = b"0123456789abcdef"
        tx, pre = self._make_transmission(payload, crc_variant=CRC32_IEEE)
        desc = FrameDescriptor(name="test32", preamble=pre, payload_symbols=len(payload),
                               crc_variant=CRC32_IEEE)
        res = frame_sync(tx, desc)
        assert res.crc_valid is True


def _build_coded_burst(payload: bytes, crc_variant=CRC16_IBM, n_rows=2, n_cols=5):
    """payload + CRC -> preamble-frame -> convolutional code, padded to the
    interleaver block multiple, then block interleaved.

    Real-world block interleavers operate on codeblock boundaries, so the TX
    pads the coded stream to a whole number of N*M blocks. The trailing pad bits
    decode to near-zero garbage that the frame sync simply ignores.
    """
    pre = [0, 1, 1, 1, 1, 1, 1, 0] * 2
    frame_bits = np.concatenate([
        np.array(pre, dtype=np.int64),
        bytes_to_bits(payload),
        bytes_to_bits(crc_bytes(payload, crc_variant)),
    ])
    coded = encode_convolutional(frame_bits, tail_bits=6)
    ilv = block_spec(n_rows, n_cols)
    pad = (-len(coded)) % ilv.size
    if pad:
        coded = np.concatenate([coded, np.zeros(pad, dtype=np.int64)])
    return interleave(coded, ilv), pre, ilv


class TestDecodeChain:
    def test_end_to_end_viterbi_block_crc(self):
        payload = b"end-to-end decode payload 42"
        recv, pre, ilv = _build_coded_burst(payload)
        # inject a burst (interleaver spreads it before FEC)
        for i in range(10, 26):
            recv[i] = 1 - recv[i]
        spec = DecodingSpec(
            name="vit|block(2,5)|crc16",
            deinterleaver=ilv,
            fec=FEC_VITERBI,
            frame_descriptor=FrameDescriptor(name="crc16", preamble=pre,
                                             payload_symbols=len(payload),
                                             crc_variant=CRC16_IBM),
        )
        res = decode_bits(recv, spec)
        assert isinstance(res, DecodingResult)
        assert res.ok is True
        assert res.crc_valid is True
        assert res.payload_bytes == payload

    def test_wrong_chain_does_not_pass(self):
        payload = b"wrong-chain check payload 99"
        recv, pre, ilv = _build_coded_burst(payload, crc_variant=CRC32_IEEE)
        # decode with a CRC-16 descriptor (mismatch) -> not valid
        spec_wrong = DecodingSpec(
            name="wrong crc16",
            deinterleaver=none_spec(),
            fec=FEC_NONE,
            frame_descriptor=FrameDescriptor(name="crc16", preamble=pre,
                                             payload_symbols=len(payload),
                                             crc_variant=CRC16_IBM),
        )
        res16 = decode_bits(recv, spec_wrong)
        assert res16.crc_valid is not True
        # correct CRC-32 chain must pass
        spec_right = DecodingSpec(
            name="right crc32",
            deinterleaver=ilv,
            fec=FEC_VITERBI,
            frame_descriptor=FrameDescriptor(name="crc32", preamble=pre,
                                             payload_symbols=len(payload),
                                             crc_variant=CRC32_IEEE),
        )
        res32 = decode_bits(recv, spec_right)
        assert res32.crc_valid is True
        assert res32.payload_bytes == payload

    def test_search_ranks_true_chain_first(self):
        payload = (b"search-ranking-payload-" * 3)[:40]  # 40 B, in scan range
        recv, pre, ilv = _build_coded_burst(payload)
        for i in range(5, 15):
            recv[i] = 1 - recv[i]
        results = search_decode(recv, top_k=5)
        assert len(results) >= 1
        assert results[0].crc_valid is True
        # The chain that matched integrates the used interleaver/viterbi pipeline
        assert results[0].payload_bytes == payload

    def test_canonical_grid_covers_block_2_5(self):
        # Bug B: the shared grid must contain the interleaver sizes the true
        # chain needs (block(2,5) is the TX config used across the test suite).
        from app.decoding.decoder import build_decode_spec_grid
        specs = build_decode_spec_grid()
        names = {s.name for s in specs}
        assert "viterbi|block(2, 5)|preamble+crc16" in names
        assert "none|block(4, 8)|preamble+crc32" in names
        # Full FEC table present in the shared grid.
        for fec in ("none", "viterbi", "rs255_223", "rs255_239",
                    "viterbi+rs255_223", "viterbi+rs255_239"):
            assert any(s.name.startswith(f"{fec}|none|") for s in specs)

    def test_make_default_specs_is_shared_grid(self):
        from app.decoding.decoder import build_decode_spec_grid, make_default_specs
        assert make_default_specs() == build_decode_spec_grid()