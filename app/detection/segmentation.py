"""Signal detection and segmentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage

from app.models import ComplexSignal
from app.detection.fft import PSDResult
from app.detection.noise_floor import NoiseFloorResult


@dataclass
class SignalSegment:
    """A detected signal segment."""
    start_sample: int
    end_sample: int
    start_freq_hz: float
    end_freq_hz: float
    peak_freq_hz: float
    peak_power_db: float
    snr_db: float
    bandwidth_hz: float
    center_frequency_hz: Optional[float] = None  # Native RF center frequency

    @property
    def num_samples(self) -> int:
        return self.end_sample - self.start_sample

    @property
    def center_freq_hz(self) -> float:
        return (self.start_freq_hz + self.end_freq_hz) / 2.0

    @property
    def rf_center_freq_hz(self) -> Optional[float]:
        if self.center_frequency_hz is not None:
            return self.center_freq_hz + self.center_frequency_hz
        return None

    @property
    def rf_peak_freq_hz(self) -> Optional[float]:
        if self.center_frequency_hz is not None:
            return self.peak_freq_hz + self.center_frequency_hz
        return None

    def to_dict(self) -> dict:
        return {
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "num_samples": self.num_samples,
            "start_freq_hz": self.start_freq_hz,
            "end_freq_hz": self.end_freq_hz,
            "center_freq_hz": self.center_freq_hz,
            "peak_freq_hz": self.peak_freq_hz,
            "peak_power_db": self.peak_power_db,
            "snr_db": self.snr_db,
            "bandwidth_hz": self.bandwidth_hz,
            "center_frequency_hz": self.center_frequency_hz,
            "rf_center_freq_hz": self.rf_center_freq_hz,
            "rf_peak_freq_hz": self.rf_peak_freq_hz,
        }


@dataclass
class SegmentationResult:
    """Result of signal segmentation."""
    segments: list[SignalSegment]
    noise_floor: NoiseFloorResult
    detection_threshold_db: float
    total_samples: int
    sample_rate: float

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    def to_dict(self) -> dict:
        return {
            "num_segments": self.num_segments,
            "noise_floor": self.noise_floor.to_dict(),
            "detection_threshold_db": self.detection_threshold_db,
            "total_samples": self.total_samples,
            "sample_rate": self.sample_rate,
            "segments": [s.to_dict() for s in self.segments],
        }


def detect_signals_psd(
    psd: PSDResult,
    noise_floor: NoiseFloorResult,
    threshold_db_above_noise: float = 10.0,
    min_segment_bins: int = 3,
    merge_gap_bins: int = 2,
) -> list[tuple[int, int]]:
    """Detect signal regions in PSD.

    Args:
        psd: PSD result
        noise_floor: Noise floor estimation
        threshold_db_above_noise: Detection threshold above noise floor
        min_segment_bins: Minimum bins to consider a signal
        merge_gap_bins: Merge segments closer than this

    Returns:
        List of (start_bin, end_bin) tuples
    """
    threshold_db = noise_floor.noise_floor_db + threshold_db_above_noise
    above_threshold = psd.psd_db > threshold_db

    labeled, num_features = ndimage.label(above_threshold)

    segments = []
    for i in range(1, num_features + 1):
        bins = np.where(labeled == i)[0]
        if len(bins) >= min_segment_bins:
            segments.append((int(bins[0]), int(bins[-1])))

    if not segments:
        return []

    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= merge_gap_bins:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return merged


def segments_from_spectrogram(
    spectrogram_db: np.ndarray,
    frequencies: np.ndarray,
    noise_floor_db: float,
    threshold_db_above_noise: float = 10.0,
    min_duration_bins: int = 3,
    min_bandwidth_bins: int = 3,
) -> list[dict]:
    """Detect signal regions in spectrogram (time-frequency).

    Returns:
        List of dicts with time and frequency boundaries
    """
    threshold_db = noise_floor_db + threshold_db_above_noise
    above = spectrogram_db > threshold_db

    labeled, num = ndimage.label(above)
    regions = []

    for i in range(1, num + 1):
        coords = np.argwhere(labeled == i)
        if len(coords) < min_duration_bins * min_bandwidth_bins:
            continue

        time_bins = coords[:, 0]
        freq_bins = coords[:, 1]

        t_start = int(time_bins.min())
        t_end = int(time_bins.max())
        f_start = int(freq_bins.min())
        f_end = int(freq_bins.max())

        if (t_end - t_start) < min_duration_bins:
            continue
        if (f_end - f_start) < min_bandwidth_bins:
            continue

        regions.append({
            "time_start_bin": t_start,
            "time_end_bin": t_end,
            "freq_start_bin": f_start,
            "freq_end_bin": f_end,
            "freq_start_hz": float(frequencies[f_start]),
            "freq_end_hz": float(frequencies[f_end]),
            "peak_power_db": float(np.max(spectrogram_db[coords[:, 0], coords[:, 1]])),
        })

    return regions


def build_segments(
    psd: PSDResult,
    noise_floor: NoiseFloorResult,
    signal: ComplexSignal,
    threshold_db_above_noise: float = 10.0,
) -> SegmentationResult:
    """Build full segmentation result from PSD.

    Args:
        psd: PSD result
        noise_floor: Noise floor estimation
        signal: Original ComplexSignal for sample-level info
        threshold_db_above_noise: Detection threshold

    Returns:
        SegmentationResult with all detected segments
    """
    threshold_db = noise_floor.noise_floor_db + threshold_db_above_noise
    sample_rate = psd.sample_rate
    fft_size = psd.fft_size
    total_samples = signal.num_samples

    freq_bins = detect_signals_psd(
        psd, noise_floor, threshold_db_above_noise
    )

    segments = []
    samples_per_bin = total_samples / len(psd.frequencies)

    for start_bin, end_bin in freq_bins:
        freq_start = float(psd.frequencies[start_bin])
        freq_end = float(psd.frequencies[end_bin])
        peak_idx = start_bin + np.argmax(psd.psd_db[start_bin:end_bin + 1])
        peak_freq = float(psd.frequencies[peak_idx])
        peak_power = float(psd.psd_db[peak_idx])

        bw_hz = abs(freq_end - freq_start)

        noise_in_band = np.mean(psd.psd_linear[start_bin:end_bin + 1])
        signal_power = peak_power - noise_floor.noise_floor_db

        seg_start = int(start_bin * samples_per_bin)
        seg_end = min(int(end_bin * samples_per_bin), total_samples)

        segments.append(SignalSegment(
            start_sample=seg_start,
            end_sample=seg_end,
            start_freq_hz=freq_start,
            end_freq_hz=freq_end,
            peak_freq_hz=peak_freq,
            peak_power_db=peak_power,
            snr_db=float(signal_power),
            bandwidth_hz=bw_hz,
            center_frequency_hz=signal.metadata.center_frequency,
        ))

    return SegmentationResult(
        segments=segments,
        noise_floor=noise_floor,
        detection_threshold_db=threshold_db,
        total_samples=total_samples,
        sample_rate=sample_rate,
    )
