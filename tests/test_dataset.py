"""Tests for dataset integration module."""

import json

import numpy as np
import pytest

from app.dataset.loader import (
    DatasetScanner,
    DatasetEntry,
    DatasetStats,
    TaskType,
    SignalType,
    SplitType,
)
from app.models import ComplexSignal


@pytest.fixture
def mock_dataset(tmp_path):
    """Create a minimal mock dataset structure for testing."""
    root = tmp_path / "dataset"
    root.mkdir()

    # Create demod_train with one signal type
    demod = root / "demod_train" / "CommSignal2"
    demod.mkdir(parents=True)

    # Generate cf32 IQ data (40960 samples = 327680 bytes)
    n_samples = 40960
    t = np.linspace(0, 1, n_samples, dtype=np.float32)
    i = np.cos(2 * np.pi * 1000 * t) * 0.5
    q = np.sin(2 * np.pi * 1000 * t) * 0.5
    iq = np.column_stack([i, q]).flatten()

    for idx in range(3):
        data_file = demod / f"CommSignal2_demod_train_{idx:04d}.sigmf-data"
        iq.tofile(str(data_file))

        meta = {
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": 25000000.0,
                "core:frequency": 2437000000.0,
                "core:version": "0.0.2",
            },
            "captures": [
                {
                    "core:datetime": "2021-03-25T01:45:21.508515Z",
                    "core:frequency": 2437000000.0,
                    "core:sample_start": 0,
                }
            ],
            "annotations": [],
        }
        meta_file = data_file.with_suffix(".sigmf-meta")
        meta_file.write_text(json.dumps(meta))

    # Create Components directory
    components = root / "demod_train" / "Components" / "CommSignal2"
    for comp_type in ["Interference", "QPSK"]:
        comp_dir = components / comp_type
        comp_dir.mkdir(parents=True)
        for idx in range(3):
            data_file = comp_dir / f"CommSignal2_demod_train_{idx:04d}.sigmf-data"
            iq.tofile(str(data_file))
            meta_file = data_file.with_suffix(".sigmf-meta")
            meta_file.write_text(json.dumps(meta))

    # Create QPSK_Bits directory with pickle files
    import pickle

    qpsk_dir = root / "demod_train" / "QPSK_Bits" / "CommSignal2"
    qpsk_dir.mkdir(parents=True)
    for idx in range(3):
        bits = np.random.randint(0, 2, 100, dtype=np.uint8)
        pkl_file = qpsk_dir / f"CommSignal2_demod_train_QPSK_bits_{idx:04d}.pkl"
        with open(pkl_file, "wb") as f:
            pickle.dump(bits, f)

    return str(root)


class TestDatasetScanner:
    def test_scan_finds_files(self, mock_dataset):
        scanner = DatasetScanner(mock_dataset)
        entries = scanner.scan()
        assert len(entries) > 0

    def test_scan_finds_data_and_components(self, mock_dataset):
        scanner = DatasetScanner(mock_dataset)
        entries = scanner.scan()
        data_entries = [e for e in entries if e.component_type is None]
        component_entries = [e for e in entries if e.component_type is not None]
        assert len(data_entries) > 0
        assert len(component_entries) > 0

    def test_filter_by_task(self, mock_dataset):
        scanner = DatasetScanner(mock_dataset)
        scanner.scan()
        demod_entries = scanner.filter(task=TaskType.DEMOD)
        all_entries = scanner.filter()
        assert len(demod_entries) == len(all_entries)

    def test_filter_by_signal(self, mock_dataset):
        scanner = DatasetScanner(mock_dataset)
        scanner.scan()
        commsignal2 = scanner.filter(signal=SignalType.COMMSIGNAL2)
        assert len(commsignal2) > 0
        commsignal3 = scanner.filter(signal=SignalType.COMMSIGNAL3)
        assert len(commsignal3) == 0

    def test_filter_by_component_type(self, mock_dataset):
        scanner = DatasetScanner(mock_dataset)
        scanner.scan()
        qpsk = scanner.filter(component_type="QPSK")
        interference = scanner.filter(component_type="Interference")
        assert len(qpsk) > 0
        assert len(interference) > 0

    def test_filter_max_entries(self, mock_dataset):
        scanner = DatasetScanner(mock_dataset)
        scanner.scan()
        limited = scanner.filter(max_entries=2)
        assert len(limited) == 2

    def test_get_stats(self, mock_dataset):
        scanner = DatasetScanner(mock_dataset)
        scanner.scan()
        stats = scanner.get_stats()
        assert isinstance(stats, DatasetStats)
        assert stats.total_files > 0
        assert stats.sample_rate == 25e6
        assert "CommSignal2" in stats.by_signal

    def test_ground_truth_bits_linked(self, mock_dataset):
        scanner = DatasetScanner(mock_dataset)
        scanner.scan()
        qpsk_entries = scanner.filter(component_type="QPSK")
        has_bits = [e for e in qpsk_entries if e.ground_truth_bits_path is not None]
        assert len(has_bits) > 0


class TestDatasetEntry:
    def test_load_entry(self, mock_dataset):
        scanner = DatasetScanner(mock_dataset)
        scanner.scan()
        entries = scanner.filter()
        assert len(entries) > 0

        signal = scanner.load_entry(entries[0])
        assert isinstance(signal, ComplexSignal)
        assert signal.data.dtype == np.complex64
        assert signal.metadata.sample_rate == 25e6
        assert signal.num_samples == 40960

    def test_load_ground_truth_bits(self, mock_dataset):
        scanner = DatasetScanner(mock_dataset)
        scanner.scan()
        qpsk_entries = scanner.filter(component_type="QPSK")
        assert len(qpsk_entries) > 0

        for entry in qpsk_entries:
            if entry.ground_truth_bits_path:
                bits = entry.load_ground_truth_bits()
                assert bits is not None
                assert len(bits) == 100
                break

    def test_load_batch(self, mock_dataset):
        scanner = DatasetScanner(mock_dataset)
        scanner.scan()
        entries = scanner.filter(max_entries=3)
        signals = scanner.load_batch(entries)
        assert len(signals) == 3
        assert all(isinstance(s, ComplexSignal) for s in signals)


class TestDatasetNonexistent:
    def test_nonexistent_path(self):
        with pytest.raises(FileNotFoundError):
            DatasetScanner("/nonexistent/path")
