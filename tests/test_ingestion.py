"""Tests for signal ingestion and normalization."""

import os
import tempfile

import numpy as np
import pytest

from app.models import ComplexSignal, DataType, SignalMetadata
from app.ingestion.parser import detect_data_type, parse_raw_iq, load_signal
from app.preprocessing.normalize import (
    remove_dc,
    normalize_power,
    normalize_peak,
    full_normalize,
)


@pytest.fixture
def sample_cf32_file(tmp_path):
    """Create a sample CF32 IQ file."""
    n_samples = 1024
    t = np.linspace(0, 1, n_samples, dtype=np.float32)
    i = np.cos(2 * np.pi * 1000 * t)
    q = np.sin(2 * np.pi * 1000 * t)
    iq = np.column_stack([i, q]).flatten()
    file_path = tmp_path / "test.cf32"
    iq.tofile(str(file_path))
    return str(file_path), n_samples


@pytest.fixture
def sample_ci16_file(tmp_path):
    """Create a sample CI16 IQ file."""
    n_samples = 512
    t = np.linspace(0, 1, n_samples, dtype=np.float64)
    i = (np.cos(2 * np.pi * 1000 * t) * 30000).astype(np.int16)
    q = (np.sin(2 * np.pi * 1000 * t) * 30000).astype(np.int16)
    iq = np.column_stack([i, q]).flatten()
    file_path = tmp_path / "test_ci16.iq"
    iq.tofile(str(file_path))
    return str(file_path), n_samples


class TestDataTypeDetection:
    def test_detect_cf32(self, tmp_path):
        file_path = tmp_path / "test.cf32"
        file_path.write_bytes(b"\x00" * 1024)
        assert detect_data_type(str(file_path)) == DataType.CF32

    def test_detect_ci16(self, tmp_path):
        file_path = tmp_path / "test.iq"
        file_path.write_bytes(b"\x00" * 1024)
        assert detect_data_type(str(file_path)) == DataType.CI16


class TestParseRawIQ:
    def test_parse_cf32(self, sample_cf32_file):
        file_path, n_samples = sample_cf32_file
        iq, sr = parse_raw_iq(file_path, DataType.CF32)
        assert iq.dtype == np.complex64
        assert len(iq) == n_samples
        assert sr is None

    def test_parse_ci16(self, sample_ci16_file):
        file_path, n_samples = sample_ci16_file
        iq, sr = parse_raw_iq(file_path, DataType.CI16)
        assert iq.dtype == np.complex64
        assert len(iq) == n_samples


class TestLoadSignal:
    def test_load_cf32(self, sample_cf32_file):
        file_path, n_samples = sample_cf32_file
        signal = load_signal(file_path, sample_rate=2e6)
        assert isinstance(signal, ComplexSignal)
        assert signal.num_samples == n_samples
        assert signal.metadata.sample_rate == 2e6
        assert signal.metadata.data_type == DataType.CF32
        assert len(signal.provenance) == 1

    def test_load_ci16(self, sample_ci16_file):
        file_path, n_samples = sample_ci16_file
        signal = load_signal(file_path, sample_rate=1e6)
        assert signal.num_samples == n_samples
        assert signal.metadata.sample_rate == 1e6


class TestNormalization:
    def test_remove_dc(self):
        data = (np.ones(100) + 1j * np.ones(100)).astype(np.complex64)
        metadata = SignalMetadata(
            filename="test.iq",
            file_size_bytes=800,
            data_type=DataType.CF32,
            sample_rate=1e6,
        )
        signal = ComplexSignal(data=data, metadata=metadata)
        signal = remove_dc(signal)
        assert abs(np.mean(signal.data.real)) < 1e-6
        assert abs(np.mean(signal.data.imag)) < 1e-6

    def test_normalize_power(self):
        data = (np.random.randn(1000) + 1j * np.random.randn(1000)).astype(np.complex64)
        metadata = SignalMetadata(
            filename="test.iq",
            file_size_bytes=8000,
            data_type=DataType.CF32,
            sample_rate=1e6,
        )
        signal = ComplexSignal(data=data, metadata=metadata)
        signal = normalize_power(signal)
        power = np.mean(np.abs(signal.data) ** 2)
        assert abs(power - 1.0) < 0.01

    def test_normalize_peak(self):
        data = (np.random.randn(1000) + 1j * np.random.randn(1000)).astype(np.complex64) * 5
        metadata = SignalMetadata(
            filename="test.iq",
            file_size_bytes=8000,
            data_type=DataType.CF32,
            sample_rate=1e6,
        )
        signal = ComplexSignal(data=data, metadata=metadata)
        signal = normalize_peak(signal)
        assert abs(np.max(np.abs(signal.data)) - 1.0) < 1e-5

    def test_full_normalize(self, sample_cf32_file):
        file_path, _ = sample_cf32_file
        signal = load_signal(file_path, sample_rate=2e6)
        signal = full_normalize(signal)
        assert signal.data.dtype == np.complex64
        assert len(signal.provenance) >= 2


class TestComplexSignal:
    def test_to_dict(self):
        data = np.ones(100, dtype=np.complex64)
        metadata = SignalMetadata(
            filename="test.iq",
            file_size_bytes=800,
            data_type=DataType.CF32,
            sample_rate=1e6,
        )
        signal = ComplexSignal(data=data, metadata=metadata)
        d = signal.to_dict()
        assert d["num_samples"] == 100
        assert d["metadata"]["filename"] == "test.iq"

    def test_provenance(self):
        data = np.ones(100, dtype=np.complex64)
        metadata = SignalMetadata(
            filename="test.iq",
            file_size_bytes=800,
            data_type= DataType.CF32,
        )
        signal = ComplexSignal(data=data, metadata=metadata)
        signal.add_provenance("test", "op1", {"key": "value"})
        assert len(signal.provenance) == 1
        assert signal.provenance[0].stage == "test"
