"""Tests for demodulation, synchronization, and EVM verification."""

import math
import numpy as np
import pytest

from app.models import ComplexSignal, SignalMetadata, DataType, IQOrdering
from app.demodulation.synchronizer import (
    derotate_carrier,
    costas_loop,
    strobe_symbols,
)
from app.demodulation.slicer import (
    slice_symbols,
    compute_evm,
    demodulate_signal,
    get_reference_constellation,
)


class TestSynchronization:
    """Test carrier derotation, Costas loop, and symbol strobing."""

    def test_derotate_carrier(self):
        """Derotation should remove carrier frequency offset."""
        sample_rate = 1e6
        cfo_hz = 15000.0
        n = 4096
        t = np.arange(n) / sample_rate
        # Transmitted baseband DC signal rotated by CFO
        tx = np.exp(1j * 2 * np.pi * cfo_hz * t).astype(np.complex64)

        rx = derotate_carrier(tx, freq_offset_hz=cfo_hz, sample_rate=sample_rate)
        # Result should be constant ~1.0
        assert np.allclose(rx.real, 1.0, atol=1e-3)
        assert np.allclose(rx.imag, 0.0, atol=1e-3)

    def test_costas_loop_qpsk(self):
        """Costas loop should track a slight phase rotation on QPSK."""
        n = 2048
        bits = np.random.randint(0, 4, n)
        syms = np.exp(1j * (bits * np.pi / 2.0 + np.pi / 4.0)).astype(np.complex64)
        # Apply static phase offset of 0.3 rad
        rotated = syms * np.exp(1j * 0.3)

        synced, phase_hist = costas_loop(rotated, modulation="QPSK", loop_bw=0.03)
        # Steady-state phase error should be near zero (last 500 samples)
        err = np.angle(synced[-500:] * np.conj(syms[-500:]))
        # Wrap error to [-pi/4, pi/4] due to 4-fold ambiguity
        err_wrapped = (err + np.pi / 4) % (np.pi / 2) - np.pi / 4
        assert np.mean(np.abs(err_wrapped)) < 0.15

    def test_strobe_symbols(self):
        """Strobing should downsample from oversampled signal to symbol points."""
        sps = 4.0
        num_symbols = 100
        syms = (np.random.choice([-1, 1], num_symbols) + 1j * np.random.choice([-1, 1], num_symbols)) / np.sqrt(2)
        # Repeat each symbol sps times
        oversampled = np.repeat(syms, int(sps))

        strobed = strobe_symbols(oversampled, samples_per_symbol=sps)
        assert len(strobed) == num_symbols
        assert np.allclose(strobed, syms, atol=1e-5)


class TestSlicingAndEVM:
    """Test constellation slicing, EVM calculation, and bit mapping."""

    def test_bpsk_slicing_and_evm(self):
        """Pure BPSK should have EVM < 1% and perfect bit recovery."""
        n = 1000
        bits_tx = np.random.randint(0, 2, n)
        syms = (bits_tx * 2.0 - 1.0).astype(np.complex64)

        sliced, bits_rx, err = slice_symbols(syms, modulation="BPSK")
        evm = compute_evm(syms, sliced)

        assert evm.evm_rms_percent < 0.1
        assert np.array_equal(bits_tx, bits_rx)

    def test_qpsk_slicing_and_evm(self):
        """Noisy QPSK should calculate accurate EVM."""
        n = 2000
        ref = get_reference_constellation("QPSK")
        sym_indices = np.random.randint(0, 4, n)
        tx_syms = ref[sym_indices]

        # Add known noise (SNR ~ 20 dB -> EVM ~ 10%)
        noise = (np.random.randn(n) + 1j * np.random.randn(n)) * 0.0707
        rx_syms = tx_syms + noise

        sliced, bits, err = slice_symbols(rx_syms, modulation="QPSK")
        evm = compute_evm(rx_syms, sliced)

        assert 7.0 <= evm.evm_rms_percent <= 13.0
        assert -23.0 <= evm.evm_db <= -17.0

    def test_16qam_slicing(self):
        """16QAM slicing should produce 4 bits per symbol."""
        ref = get_reference_constellation("16QAM")
        assert len(ref) == 16
        sliced, bits, err = slice_symbols(ref, modulation="16QAM")
        assert len(bits) == 16 * 4

    def test_demodulate_signal_pipeline(self):
        """demodulate_signal should synchronize, slice, and record provenance."""
        n = 8192
        bits = np.random.randint(0, 4, n)
        syms = np.exp(1j * (bits * np.pi / 2.0 + np.pi / 4.0)).astype(np.complex64)
        noise = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64) * 0.05
        data = syms + noise

        meta = SignalMetadata(
            filename="test_qpsk_demod.cf32",
            file_size_bytes=n * 8,
            data_type=DataType.CF32,
            sample_rate=1e6,
            iq_ordering=IQOrdering.INTERLEAVED,
        )
        sig = ComplexSignal(data=data, metadata=meta)

        res = demodulate_signal(sig, modulation="QPSK", samples_per_symbol=1.0, max_symbols=512)
        assert res.modulation == "QPSK"
        assert res.num_symbols > 0
        assert res.evm.evm_rms_percent < 15.0
        assert len(res.bits) > 0

        # Check provenance
        stages = [p.stage for p in sig.provenance]
        assert "demodulation" in stages
