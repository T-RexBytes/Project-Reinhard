"""Tests for modulation classifier and feature parameter alignment."""

import numpy as np
import pytest

from app.characterization.features import extract_features, FeatureVector
from app.modulation.classifier import ModulationClassifier, ModulationCandidate, ModulationClassificationResult


class TestFeatureAlignment:
    """Verify that features align across different dataset sampling rates."""

    def test_instantaneous_freq_norm_invariance(self):
        """Instantaneous frequency norm should be invariant to sample rate scaling."""
        n = 4096
        # Generate a test FM signal with known normalized frequency deviation
        t_norm = np.arange(n)
        cycles_per_sample = 0.05
        phase = 2 * np.pi * cycles_per_sample * np.sin(2 * np.pi * 0.005 * t_norm)
        iq = np.exp(1j * phase).astype(np.complex64)

        # Extract at 25 MHz (data0)
        fv_25 = extract_features(iq, sample_rate=25e6)

        # Extract at 61.44 MHz (zenodo)
        fv_61 = extract_features(iq, sample_rate=61.44e6)

        # In Hz, the values scale with sample_rate
        assert abs(fv_61.instantaneous_freq_std / fv_25.instantaneous_freq_std - (61.44 / 25.0)) < 0.01

        # Dimensionless normalized deviations must match identically!
        assert abs(fv_25.instantaneous_freq_std_norm - fv_61.instantaneous_freq_std_norm) < 1e-6
        assert fv_25.instantaneous_freq_std_norm > 0.0

    def test_physical_parameters_preserved(self):
        """Verify that raw power and DC offsets are preserved on raw complex signals."""
        x = (np.ones(1024, dtype=np.complex64) * (2.0 + 3.0j))
        fv = extract_features(x, sample_rate=25e6)

        assert abs(fv.raw_power - 13.0) < 1e-4  # |2+3j|^2 = 4+9 = 13
        assert abs(fv.dc_offset_real - 2.0) < 1e-4
        assert abs(fv.dc_offset_imag - 3.0) < 1e-4


class TestModulationClassifier:
    """Verify candidate classification on distinct modulation profiles."""

    def test_qpsk_classification(self):
        """Feature vector matching QPSK characteristics should rank QPSK #1."""
        fv = FeatureVector(
            C42_norm=0.88,
            C20=1.0,
            C21_abs=0.01,  # near zero symmetry
            papr_db=4.5,
            amplitude_variance_norm=0.38,
            kurtosis_amplitude=-0.80,  # single-shell
            instantaneous_freq_std_norm=0.03,
            phase_std=1.2,
            spectral_flatness=0.01,
            sample_rate=25e6,
        )
        clf = ModulationClassifier()
        res = clf.classify(fv, snr_db=18.0, symbol_rate_hz=500e3, bandwidth_hz=700e3)

        assert res.top_candidate == "QPSK"
        assert res.confidence > 0.7
        # Verify candidate leaderboard contains all MVP families
        mods = [c.modulation for c in res.candidates]
        assert "QPSK" in mods
        assert "BPSK" in mods
        assert "16QAM" in mods
        assert "2-FSK" in mods

    def test_bpsk_classification(self):
        """Feature vector matching BPSK characteristics should rank BPSK #1."""
        fv = FeatureVector(
            C42_norm=1.95,
            C20=1.0,
            C21_abs=0.92,  # high 2nd-harmonic symmetry
            papr_db=3.2,
            amplitude_variance_norm=0.15,
            kurtosis_amplitude=-0.95,
            instantaneous_freq_std_norm=0.02,
            phase_std=0.8,
            spectral_flatness=0.01,
            sample_rate=25e6,
        )
        clf = ModulationClassifier()
        res = clf.classify(fv, snr_db=20.0, symbol_rate_hz=500e3, bandwidth_hz=650e3)

        assert res.top_candidate == "BPSK"
        assert res.confidence > 0.7

    def test_16qam_classification(self):
        """Feature vector matching 16QAM characteristics should rank 16QAM #1."""
        fv = FeatureVector(
            C42_norm=0.68,
            C20=1.0,
            C21_abs=0.02,
            papr_db=7.2,
            amplitude_variance_norm=0.52,
            kurtosis_amplitude=0.25,  # multi-ring distribution
            instantaneous_freq_std_norm=0.04,
            phase_std=1.4,
            spectral_flatness=0.02,
            sample_rate=25e6,
        )
        clf = ModulationClassifier()
        res = clf.classify(fv, snr_db=22.0, symbol_rate_hz=1e6, bandwidth_hz=1.3e6)

        assert res.top_candidate == "16QAM"
        assert res.confidence > 0.7

    def test_2fsk_classification(self):
        """Constant envelope signal with low C42 and high frequency deviation should rank 2-FSK."""
        fv = FeatureVector(
            C42_norm=0.04,
            C20=1.0,
            C21_abs=0.01,
            papr_db=1.8,
            amplitude_variance_norm=0.10,
            kurtosis_amplitude=-1.1,
            instantaneous_freq_std_norm=0.08,
            phase_std=1.5,
            spectral_flatness=0.08,
            sample_rate=25e6,
        )
        clf = ModulationClassifier()
        res = clf.classify(fv, snr_db=16.0, symbol_rate_hz=200e3, bandwidth_hz=800e3)
        assert "FSK" in res.top_candidate

    def test_low_snr_uncertainty(self):
        """At very low SNR (<3 dB), Noise/CW should have elevated ranking and lower confidence."""
        fv = FeatureVector(
            C42_norm=0.05,
            C20=1.0,
            C21_abs=0.02,
            papr_db=8.0,
            amplitude_variance_norm=0.55,
            kurtosis_amplitude=0.10,
            instantaneous_freq_std_norm=0.15,
            phase_std=1.81,  # uniform noise
            spectral_flatness=0.85,  # white noise
            sample_rate=25e6,
        )
        clf = ModulationClassifier()
        res = clf.classify(fv, snr_db=1.5)

        assert res.top_candidate == "Noise/CW"

    def test_classify_signal_provenance(self):
        """classify_signal should correctly classify ComplexSignal and append provenance."""
        from app.models import ComplexSignal, SignalMetadata, DataType, IQOrdering
        from app.modulation.classifier import classify_signal

        # Generate synthetic QPSK signal
        n = 8192
        bits = np.random.randint(0, 4, n)
        syms = np.exp(1j * (bits * np.pi / 2.0 + np.pi / 4.0)).astype(np.complex64)
        noise = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64) * 0.05
        data = syms + noise

        meta = SignalMetadata(
            filename="synthetic_qpsk.cf32",
            file_size_bytes=n * 8,
            data_type=DataType.CF32,
            sample_rate=1e6,
            iq_ordering=IQOrdering.INTERLEAVED,
        )
        sig = ComplexSignal(data=data, metadata=meta)
        initial_prov_count = len(sig.provenance)

        res = classify_signal(sig)
        assert res.top_candidate == "QPSK"
        # classify_signal now records 2 provenance steps:
        #   1. characterization / estimate_freq_offset  (F2: hint-aware CFO)
        #   2. modulation_classification / classify_candidates
        assert len(sig.provenance) == initial_prov_count + 2
        assert sig.provenance[-1].stage == "modulation_classification"
        assert sig.provenance[-1].operation == "classify_candidates"
        assert sig.provenance[-1].parameters["top_candidate"] == "QPSK"
        # Verify the CFO hint step was recorded correctly (F2)
        assert sig.provenance[-2].stage == "characterization"
        assert sig.provenance[-2].operation == "estimate_freq_offset"
        assert "mod_hint" in sig.provenance[-2].parameters


    def test_classify_iq(self):
        """classify_iq should directly evaluate numpy array."""
        from app.modulation.classifier import classify_iq

        # Generate synthetic BPSK signal
        n = 4096
        bits = np.random.randint(0, 2, n) * 2 - 1  # -1 or +1
        iq = bits.astype(np.complex64)
        res = classify_iq(iq, sample_rate=1e6, snr_db=25.0)

        assert res.top_candidate == "BPSK"
        assert res.confidence > 0.6

