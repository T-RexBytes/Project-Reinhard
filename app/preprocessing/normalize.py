"""Signal normalization and preprocessing."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import signal as scipy_signal

from app.models import ComplexSignal


def remove_dc(signal: ComplexSignal) -> ComplexSignal:
    """Remove DC offset from the signal without mutating original data."""
    dc_offset = np.mean(signal.data)
    new_data = (signal.data - dc_offset).astype(np.complex64)
    new_sig = ComplexSignal(data=new_data, metadata=signal.metadata, provenance=list(signal.provenance))
    new_sig.add_provenance(
        stage="preprocessing",
        operation="remove_dc",
        parameters={"dc_offset_real": float(dc_offset.real), "dc_offset_imag": float(dc_offset.imag)},
    )
    return new_sig


def normalize_power(signal: ComplexSignal) -> ComplexSignal:
    """Normalize signal to unit average power without mutating original data."""
    power = np.mean(np.abs(signal.data) ** 2)
    if power > 0:
        scale = 1.0 / np.sqrt(power)
        new_data = (signal.data * scale).astype(np.complex64)
    else:
        new_data = signal.data.copy()
    new_sig = ComplexSignal(data=new_data, metadata=signal.metadata, provenance=list(signal.provenance))
    new_sig.add_provenance(
        stage="preprocessing",
        operation="normalize_power",
        parameters={"original_power": float(power)},
    )
    return new_sig


def normalize_peak(signal: ComplexSignal) -> ComplexSignal:
    """Normalize signal to unit peak amplitude without mutating original data."""
    peak = np.max(np.abs(signal.data))
    if peak > 0:
        new_data = (signal.data / peak).astype(np.complex64)
    else:
        new_data = signal.data.copy()
    new_sig = ComplexSignal(data=new_data, metadata=signal.metadata, provenance=list(signal.provenance))
    new_sig.add_provenance(
        stage="preprocessing",
        operation="normalize_peak",
        parameters={"original_peak": float(peak)},
    )
    return new_sig


def lowpass_filter(
    signal: ComplexSignal,
    cutoff_ratio: float = 0.9,
    num_taps: int = 101,
) -> ComplexSignal:
    """Apply a lowpass filter to the signal.

    Args:
        signal: Input ComplexSignal
        cutoff_ratio: Cutoff frequency as ratio of Nyquist (0-1)
        num_taps: Number of filter taps
    """
    taps = scipy_signal.firwin(num_taps, cutoff_ratio)
    filtered = scipy_signal.lfilter(taps, 1.0, signal.data)
    new_data = filtered.astype(np.complex64)
    new_sig = ComplexSignal(data=new_data, metadata=signal.metadata, provenance=list(signal.provenance))
    new_sig.add_provenance(
        stage="preprocessing",
        operation="lowpass_filter",
        parameters={"cutoff_ratio": cutoff_ratio, "num_taps": num_taps},
    )
    return new_sig


def auto_gain_control(
    signal: ComplexSignal,
    target_power_db: float = 0.0,
) -> ComplexSignal:
    """Apply automatic gain control to reach target power level."""
    current_power = np.mean(np.abs(signal.data) ** 2)
    current_power_db = 10 * np.log10(max(current_power, 1e-20))
    gain_db = target_power_db - current_power_db
    gain_linear = 10 ** (gain_db / 20.0)
    new_data = (signal.data * gain_linear).astype(np.complex64)
    new_sig = ComplexSignal(data=new_data, metadata=signal.metadata, provenance=list(signal.provenance))
    new_sig.add_provenance(
        stage="preprocessing",
        operation="auto_gain_control",
        parameters={"target_power_db": target_power_db, "gain_applied_db": float(gain_db)},
    )
    return new_sig


def full_normalize(
    signal: ComplexSignal,
    remove_dc_offset: bool = True,
    normalize_to: Optional[str] = "power",
    apply_lpf: bool = False,
    lpf_cutoff: float = 0.9,
) -> ComplexSignal:
    """Run the complete normalization pipeline.

    Args:
        signal: Input ComplexSignal
        remove_dc_offset: Whether to remove DC offset
        normalize_to: "power", "peak", or None
        apply_lpf: Whether to apply lowpass filter
        lpf_cutoff: Lowpass cutoff ratio (0-1)

    Returns:
        Normalized ComplexSignal
    """
    if remove_dc_offset:
        signal = remove_dc(signal)

    if apply_lpf:
        signal = lowpass_filter(signal, cutoff_ratio=lpf_cutoff)

    if normalize_to == "power":
        signal = normalize_power(signal)
    elif normalize_to == "peak":
        signal = normalize_peak(signal)

    return signal
