"""IQ and WAV file parser with format detection."""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Optional

import numpy as np

from app.models import (
    ComplexSignal,
    DataType,
    IQOrdering,
    SignalMetadata,
)


def detect_data_type(file_path: str) -> DataType:
    """Detect IQ data type from file extension, SigMF sidecar, or content."""
    sigmf_meta = load_sigmf_metadata(file_path)
    if sigmf_meta and "global" in sigmf_meta and "core:datatype" in sigmf_meta["global"]:
        dt_str = str(sigmf_meta["global"]["core:datatype"]).lower()
        if "cf32" in dt_str or "c32" in dt_str:
            return DataType.CF32
        elif "ci16" in dt_str or "i16" in dt_str:
            return DataType.CI16
        elif "ci8" in dt_str or "i8" in dt_str:
            return DataType.CI8
        elif "cu8" in dt_str or "u8" in dt_str:
            return DataType.CU8

    ext = Path(file_path).suffix.lower()

    ext_map = {
        ".iq": DataType.CI16,
        ".cf32": DataType.CF32,
        ".c32": DataType.CF32,
        ".ci16": DataType.CI16,
        ".i16": DataType.CI16,
        ".ci8": DataType.CI8,
        ".i8": DataType.CI8,
        ".cu8": DataType.CU8,
        ".u8": DataType.CU8,
    }

    if ext in ext_map:
        return ext_map[ext]

    if ext == ".wav":
        return DataType.CF32

    file_size = os.path.getsize(file_path)
    if file_size % 4 == 0 and file_size % 8 != 0:
        return DataType.CI16
    if file_size % 2 == 0 and file_size % 4 != 0:
        return DataType.CI8

    return DataType.CF32


def load_sigmf_metadata(file_path: str) -> Optional[dict]:
    """Try to load SigMF metadata if it exists alongside the IQ file."""
    base = Path(file_path).with_suffix("")
    sigmf_path = base.with_suffix(".sigmf-meta")

    if sigmf_path.exists():
        with open(sigmf_path, "r") as f:
            return json.load(f)

    sidecar = Path(file_path).with_suffix(".sigmf-meta")
    if sidecar.exists():
        with open(sidecar, "r") as f:
            return json.load(f)

    return None


def parse_wav_file(file_path: str) -> tuple[np.ndarray, Optional[float], Optional[float]]:
    """Parse a WAV file and extract complex IQ data.

    Returns:
        Complex IQ array (complex64), sample_rate, center_frequency
    """
    import wave

    with wave.open(file_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    if sampwidth == 4 and n_channels == 2:
        data = np.frombuffer(raw_data, dtype=np.float32).reshape(-1, 2)
        iq = data[:, 0] + 1j * data[:, 1]
    elif sampwidth == 2 and n_channels == 2:
        data = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, 2)
        iq = (data[:, 0].astype(np.float32) / 32768.0
              + 1j * data[:, 1].astype(np.float32) / 32768.0)
    elif sampwidth == 1 and n_channels == 2:
        data = np.frombuffer(raw_data, dtype=np.uint8).reshape(-1, 2)
        iq = ((data[:, 0].astype(np.float32) - 128.0) / 128.0
              + 1j * (data[:, 1].astype(np.float32) - 128.0) / 128.0)
    elif n_channels == 1:
        raw = np.frombuffer(raw_data, dtype=np.float32 if sampwidth == 4 else np.int16)
        if sampwidth == 4:
            iq = raw.astype(np.complex64)
        else:
            iq = (raw.astype(np.float32) / 32768.0).astype(np.complex64)
    else:
        raise ValueError(f"Unsupported WAV format: {n_channels} channels, {sampwidth} bytes")

    return iq.astype(np.complex64), float(framerate), None


def parse_raw_iq(
    file_path: str,
    data_type: DataType,
    sample_rate: Optional[float] = None,
    iq_ordering: IQOrdering = IQOrdering.INTERLEAVED,
) -> tuple[np.ndarray, Optional[float]]:
    """Parse a raw IQ file into complex64.

    Returns:
        Complex IQ array (complex64), sample_rate
    """
    raw = np.fromfile(file_path, dtype=np.uint8)

    if data_type == DataType.CF32:
        samples = np.frombuffer(raw.tobytes(), dtype=np.float32)
        if len(samples) % 2 != 0:
            samples = samples[: len(samples) - 1]
        iq = samples.reshape(-1, 2)
        if iq_ordering == IQOrdering.PLANAR:
            result = iq[:, 0] + 1j * iq[:, 1] if len(iq.shape) > 1 else samples.astype(np.complex64)
        else:
            result = iq[:, 0] + 1j * iq[:, 1]

    elif data_type == DataType.CI16:
        samples = np.frombuffer(raw.tobytes(), dtype=np.int16)
        if len(samples) % 2 != 0:
            samples = samples[: len(samples) - 1]
        iq = samples.reshape(-1, 2)
        result = (iq[:, 0].astype(np.float32) / 32768.0
                  + 1j * iq[:, 1].astype(np.float32) / 32768.0)

    elif data_type == DataType.CI8:
        iq = raw.reshape(-1, 2)
        result = (iq[:, 0].astype(np.float32) / 128.0
                  + 1j * iq[:, 1].astype(np.float32) / 128.0)

    elif data_type == DataType.CU8:
        iq = raw.reshape(-1, 2)
        result = ((iq[:, 0].astype(np.float32) - 128.0) / 128.0
                  + 1j * (iq[:, 1].astype(np.float32) - 128.0) / 128.0)

    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    return result.astype(np.complex64), sample_rate


def load_signal(
    file_path: str,
    data_type: Optional[DataType] = None,
    sample_rate: Optional[float] = None,
    iq_ordering: IQOrdering = IQOrdering.INTERLEAVED,
) -> ComplexSignal:
    """Load a signal file and return a ComplexSignal.

    Args:
        file_path: Path to IQ or WAV file
        data_type: Override data type detection
        sample_rate: Override sample rate (from metadata or file)
        iq_ordering: I/Q sample ordering for raw IQ files

    Returns:
        ComplexSignal with metadata and provenance
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = path.stat().st_size
    ext = path.suffix.lower()

    sigmf_meta = load_sigmf_metadata(file_path)

    sigmf_datatype = None
    if sigmf_meta:
        global_meta = sigmf_meta.get("global", {})
        if "core:datatype" in global_meta:
            dt_str = str(global_meta["core:datatype"]).lower()
            if "cf32" in dt_str or "c32" in dt_str:
                sigmf_datatype = DataType.CF32
            elif "ci16" in dt_str or "i16" in dt_str:
                sigmf_datatype = DataType.CI16
            elif "ci8" in dt_str or "i8" in dt_str:
                sigmf_datatype = DataType.CI8
            elif "cu8" in dt_str or "u8" in dt_str:
                sigmf_datatype = DataType.CU8
        if "core:sample_rate" in global_meta:
            sample_rate = float(global_meta["core:sample_rate"])
        captures = sigmf_meta.get("captures", [])
        if "core:frequency" in global_meta:
            center_freq = float(global_meta["core:frequency"])
        elif captures and "core:frequency" in captures[0]:
            center_freq = float(captures[0]["core:frequency"])
        else:
            center_freq = None
    else:
        center_freq = None

    if ext == ".wav":
        iq_data, wav_sr, _ = parse_wav_file(file_path)
        if sample_rate is None:
            sample_rate = wav_sr
        detected_type = DataType.CF32
    else:
        if data_type is None:
            detected_type = sigmf_datatype or detect_data_type(file_path)
        else:
            detected_type = data_type
        iq_data, inferred_sr = parse_raw_iq(
            file_path, detected_type, sample_rate, iq_ordering
        )
        if sample_rate is None:
            sample_rate = inferred_sr

    metadata = SignalMetadata(
        filename=path.name,
        file_size_bytes=file_size,
        data_type=detected_type,
        sample_rate=sample_rate,
        center_frequency=center_freq,
        iq_ordering=iq_ordering,
        sigmf_metadata=sigmf_meta,
        num_samples=len(iq_data),
        duration_seconds=len(iq_data) / sample_rate if sample_rate else 0.0,
    )

    signal = ComplexSignal(data=iq_data, metadata=metadata)
    signal.add_provenance(
        stage="ingestion",
        operation="load_signal",
        parameters={
            "file_path": str(path),
            "data_type": detected_type.value,
            "sample_rate": sample_rate,
        },
    )

    return signal
