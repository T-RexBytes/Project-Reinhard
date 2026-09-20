"""Tests for signal detection module."""

import numpy as np
import pytest

from app.models import ComplexSignal, DataType, SignalMetadata
from app.detection.fft import compute_psd, compute_fft
from app.detection.spectrogram import compute_spectrogram
from app.detection.noise_floor import estimate_noise_floor
from app.detection.segmentation import build_segments, detect_signals_psd
from app.detection.snr import estimate_snr_from_psd, estimate_snr_m2m4
from app.detection.bandwidth import estimate_bandwidth


def make_test_signal(
    num_samples: int = 8192,
    sample_rate: float = 1e6,
    signal_freq_hz: float = 100e3,
    snr_db: float = 20.0,
    has_signal: bool = True,
) -> ComplexSignal:
    """Create a test signal with known parameters."""
    t = np.arange(num_samples) / sample_rate

    if has_signal:
        signal_power = 1.0
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        i = np.cos(2 * np.pi * signal_freq_hz * t) * np.sqrt(signal_power / 2)
        q = np.sin(2 * np.pi * signal_freq_hz * t) * np.sqrt(signal_power / 2)
        noise = np.random.randn(num_samples) * np.sqrt(noise_power / 2)
        iq = (i + noise + 1j * (q + noise)).astype(np.complex64)
    else:
        iq = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)).astype(np.complex64)
        iq = iq / np.sqrt(np.mean(np.abs(iq) ** 2))

    metadata = SignalMetadata(
        filename="test.iq",
        file_size_bytes=num_samples * 8,
        data_type=DataType.CF32,
        sample_rate=sample_rate,
    )

    return ComplexSignal(data=iq, metadata=metadata)


class TestPSD:
    def test_psd_basic(self):
        signal = make_test_signal()
        psd = compute_psd(signal, fft_size=1024)

        assert psd.frequencies.shape == psd.psd_db.shape
        assert psd.psd_linear.shape == psd.psd_db.shape
        assert len(psd.psd_db) == 1024
        assert psd.sample_rate == 1e6
        assert psd.fft_size == 1024

    def test_psd_with_signal(self):
        signal = make_test_signal(has_signal=True, snr_db=20)
        psd = compute_psd(signal, fft_size=2048)

        peak_idx = np.argmax(psd.psd_db)
        peak_freq = psd.frequencies[peak_idx]

        assert abs(peak_freq - 100e3) < 5000

    def test_psd_noise_only(self):
        signal = make_test_signal(has_signal=False)
        psd = compute_psd(signal, fft_size=1024)

        std_db = np.std(psd.psd_db)
        assert std_db < 5.0

    def test_fft_basic(self):
        signal = make_test_signal()
        freqs, fft = compute_fft(signal, fft_size=1024)

        assert len(freqs) == 1024
        assert len(fft) == 1024
        assert freqs[0] < freqs[-1]


class TestSpectrogram:
    def test_spectrogram_basic(self):
        signal = make_test_signal()
        spec = compute_spectrogram(signal, fft_size=512)

        assert spec.spectrogram_db.ndim == 2
        assert spec.spectrogram_db.shape[0] == len(spec.frequencies)
        assert len(spec.time_bins) > 0
        assert spec.num_time_bins == spec.spectrogram_db.shape[1]

    def test_spectrogram_with_max_time(self):
        signal = make_test_signal(num_samples=32768)
        spec = compute_spectrogram(signal, fft_size=512, max_time_bins=20)

        assert spec.num_time_bins <= 25  # allow rounding


class TestNoiseFloor:
    def test_noise_floor_percentile(self):
        signal = make_test_signal(has_signal=False)
        psd = compute_psd(signal, fft_size=1024)
        nf = estimate_noise_floor(psd, method="percentile", percentile=10)

        assert nf.noise_floor_db < 0
        assert nf.method == "percentile"

    def test_noise_floor_median(self):
        signal = make_test_signal(has_signal=False)
        psd = compute_psd(signal, fft_size=1024)
        nf = estimate_noise_floor(psd, method="median")

        assert nf.noise_floor_db < 0
        assert nf.method == "median"

    def test_noise_floor_mad(self):
        signal = make_test_signal(has_signal=False)
        psd = compute_psd(signal, fft_size=1024)
        nf = estimate_noise_floor(psd, method="mad")

        assert nf.noise_floor_db < 0
        assert nf.method == "mad"


class TestSegmentation:
    def test_detect_signals_with_signal(self):
        signal = make_test_signal(has_signal=True, snr_db=20)
        psd = compute_psd(signal, fft_size=1024)
        nf = estimate_noise_floor(psd, method="mad")

        segments = detect_signals_psd(psd, nf, threshold_db_above_noise=8)
        assert len(segments) > 0

    def test_detect_signals_noise_only(self):
        signal = make_test_signal(has_signal=False)
        psd = compute_psd(signal, fft_size=1024)
        nf = estimate_noise_floor(psd, method="mad")

        segments = detect_signals_psd(psd, nf, threshold_db_above_noise=10)
        assert len(segments) == 0

    def test_build_segments(self):
        signal = make_test_signal(has_signal=True, snr_db=20)
        psd = compute_psd(signal, fft_size=1024)
        nf = estimate_noise_floor(psd, method="mad")

        result = build_segments(psd, nf, signal)
        assert result.num_segments >= 1
        assert result.sample_rate == 1e6
        assert result.total_samples == 8192

    def test_segment_to_dict(self):
        signal = make_test_signal(has_signal=True, snr_db=20)
        psd = compute_psd(signal, fft_size=1024)
        nf = estimate_noise_floor(psd, method="mad")
        result = build_segments(psd, nf, signal)

        d = result.to_dict()
        assert "segments" in d
        assert "noise_floor" in d


class TestSNR:
    def test_snr_from_psd(self):
        signal = make_test_signal(has_signal=True, snr_db=20)
        psd = compute_psd(signal, fft_size=1024)
        nf = estimate_noise_floor(psd, method="mad")
        snr = estimate_snr_from_psd(psd, nf)

        assert snr.snr_db > 5
        assert snr.method == "psd_ratio"

    def test_snr_m2m4(self):
        signal = make_test_signal(has_signal=True, snr_db=20)
        snr = estimate_snr_m2m4(signal.data)

        assert snr.snr_db > 5
        assert snr.method == "m2m4"


class TestBandwidth:
    def test_bandwidth_3db(self):
        signal = make_test_signal(has_signal=True, snr_db=20)
        psd = compute_psd(signal, fft_size=2048)
        bw = estimate_bandwidth(psd, method="3db")

        assert bw.bandwidth_hz > 0
        assert bw.lower_3db_hz < bw.upper_3db_hz
        assert bw.method == "3db"

    def test_bandwidth_fractional(self):
        signal = make_test_signal(has_signal=True, snr_db=20)
        psd = compute_psd(signal, fft_size=2048)
        bw = estimate_bandwidth(psd, method="fractional", fraction=0.99)

        assert bw.bandwidth_hz > 0
        assert bw.method == "fractional_0.99"

    def test_bandwidth_to_dict(self):
        signal = make_test_signal(has_signal=True, snr_db=20)
        psd = compute_psd(signal, fft_size=1024)
        bw = estimate_bandwidth(psd)

        d = bw.to_dict()
        assert "bandwidth_hz" in d
        assert "center_freq_hz" in d
