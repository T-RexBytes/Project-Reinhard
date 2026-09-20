"""Dataset scanner and loader for MIT RF Challenge data."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from app.ingestion.parser import load_signal
from app.models import ComplexSignal


class TaskType(str, Enum):
    DEMOD = "demod"
    SEPARATION = "sep"
    FRAME = "frame"


class SignalType(str, Enum):
    COMMSIGNAL2 = "CommSignal2"
    COMMSIGNAL3 = "CommSignal3"
    EMISIGNAL1 = "EMISignal1"


class SplitType(str, Enum):
    TRAIN = "train"
    VAL = "val"


@dataclass
class DatasetEntry:
    """A single file entry from the dataset."""
    data_path: Path
    meta_path: Path
    task: TaskType
    signal: SignalType
    split: SplitType
    index: int
    component_type: Optional[str] = None  # "Interference", "QPSK", "Comm2"
    has_ground_truth: bool = False
    ground_truth_bits_path: Optional[Path] = None

    def load(self, sample_rate: Optional[float] = None) -> ComplexSignal:
        """Load this entry as a ComplexSignal via the existing parser."""
        return load_signal(str(self.data_path), sample_rate=sample_rate)

    def load_ground_truth_bits(self) -> Optional[np.ndarray]:
        """Load QPSK ground truth bits from pickle file."""
        if self.ground_truth_bits_path and self.ground_truth_bits_path.exists():
            with open(self.ground_truth_bits_path, "rb") as f:
                return pickle.load(f)
        return None


@dataclass
class DatasetStats:
    """Summary statistics for a scanned dataset."""
    root_path: str
    total_files: int = 0
    total_size_bytes: int = 0
    by_task: dict[str, int] = field(default_factory=dict)
    by_signal: dict[str, int] = field(default_factory=dict)
    by_split: dict[str, int] = field(default_factory=dict)
    signal_types_found: list[str] = field(default_factory=list)
    sample_rate: float = 25e6
    file_size_bytes: int = 327680
    samples_per_file: int = 40960


class DatasetScanner:
    """Scan and index the MIT RF Challenge dataset structure."""

    # Known signal types in the dataset
    SIGNAL_TYPES = {"CommSignal2", "CommSignal3", "EMISignal1"}

    def __init__(self, root_path: str):
        self.root = Path(root_path)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root_path}")
        self._entries: list[DatasetEntry] = []
        self._scanned = False

    def scan(self) -> list[DatasetEntry]:
        """Scan the entire dataset and build an index of all files."""
        self._entries.clear()

        for task_dir_name, task_type in [
            ("demod_train", TaskType.DEMOD),
            ("demod_val", TaskType.DEMOD),
            ("sep_train", TaskType.SEPARATION),
            ("sep_val", TaskType.SEPARATION),
            ("train_frame", TaskType.FRAME),
        ]:
            task_dir = self.root / task_dir_name
            if not task_dir.exists():
                continue

            split = SplitType.TRAIN if "train" in task_dir_name else SplitType.VAL

            # Scan signal type directories
            for signal_dir in task_dir.iterdir():
                if not signal_dir.is_dir():
                    continue
                if signal_dir.name == "Components":
                    continue
                if signal_dir.name == "QPSK_Bits":
                    continue
                if signal_dir.name not in self.SIGNAL_TYPES:
                    continue

                signal_type = SignalType(signal_dir.name)
                self._scan_signal_dir(signal_dir, task_type, signal_type, split)

            # Scan Components subdirectories
            components_dir = task_dir / "Components"
            if components_dir.exists():
                self._scan_components(components_dir, task_type, split)

        self._scanned = True
        return self._entries

    def _scan_signal_dir(
        self,
        signal_dir: Path,
        task_type: TaskType,
        signal_type: SignalType,
        split: SplitType,
    ) -> None:
        """Scan a single signal type directory for .sigmf-data files."""
        for data_file in sorted(signal_dir.glob("*.sigmf-data")):
            meta_file = data_file.with_suffix(".sigmf-meta")
            if not meta_file.exists():
                continue

            # Extract index from filename
            index = self._extract_index(data_file.stem)

            entry = DatasetEntry(
                data_path=data_file,
                meta_path=meta_file,
                task=task_type,
                signal=signal_type,
                split=split,
                index=index,
            )
            self._entries.append(entry)

    def _scan_components(
        self,
        components_dir: Path,
        task_type: TaskType,
        split: SplitType,
    ) -> None:
        """Scan Components subdirectories for ground truth data."""
        for signal_dir in components_dir.iterdir():
            if not signal_dir.is_dir():
                continue
            if signal_dir.name not in self.SIGNAL_TYPES:
                continue

            signal_type = SignalType(signal_dir.name)

            for component_dir in signal_dir.iterdir():
                if not component_dir.is_dir():
                    continue
                component_type = component_dir.name  # "Interference", "QPSK", "Comm2"

                for data_file in sorted(component_dir.glob("*.sigmf-data")):
                    meta_file = data_file.with_suffix(".sigmf-meta")
                    if not meta_file.exists():
                        continue

                    index = self._extract_index(data_file.stem)
                    entry = DatasetEntry(
                        data_path=data_file,
                        meta_path=meta_file,
                        task=task_type,
                        signal=signal_type,
                        split=split,
                        index=index,
                        component_type=component_type,
                        has_ground_truth=True,
                    )
                    self._entries.append(entry)

        # Scan QPSK_Bits for pickle ground truth
        qpsk_bits_dir = components_dir.parent / "QPSK_Bits"
        if qpsk_bits_dir.exists():
            self._link_qpsk_bits(qpsk_bits_dir)

    def _link_qpsk_bits(self, qpsk_bits_dir: Path) -> None:
        """Link QPSK pickle files to matching entries."""
        for pkl_file in sorted(qpsk_bits_dir.rglob("*.pkl")):
            # Find matching entry by signal type and index
            signal_name = pkl_file.parent.name
            if signal_name not in self.SIGNAL_TYPES:
                continue

            index = self._extract_index(pkl_file.stem)

            for entry in self._entries:
                if (
                    entry.signal.value == signal_name
                    and entry.index == index
                    and entry.component_type == "QPSK"
                ):
                    entry.ground_truth_bits_path = pkl_file
                    break

    @staticmethod
    def _extract_index(filename: str) -> int:
        """Extract numeric index from filename like 'CommSignal2_demod_train_0042'."""
        parts = filename.split("_")
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        return 0

    def get_stats(self) -> DatasetStats:
        """Return summary statistics of the scanned dataset."""
        if not self._scanned:
            self.scan()

        stats = DatasetStats(root_path=str(self.root))
        stats.total_files = len(self._entries)
        stats.signal_types_found = list({e.signal.value for e in self._entries})

        for entry in self._entries:
            try:
                size = entry.data_path.stat().st_size
            except OSError:
                size = 0
            stats.total_size_bytes += size

            task_key = entry.task.value
            stats.by_task[task_key] = stats.by_task.get(task_key, 0) + 1

            signal_key = entry.signal.value
            stats.by_signal[signal_key] = stats.by_signal.get(signal_key, 0) + 1

            split_key = entry.split.value
            stats.by_split[split_key] = stats.by_split.get(split_key, 0) + 1

        if self._entries:
            try:
                stats.file_size_bytes = self._entries[0].data_path.stat().st_size
            except OSError:
                pass
            stats.samples_per_file = stats.file_size_bytes // 8  # cf32 = 8 bytes/sample

        return stats

    def filter(
        self,
        task: Optional[TaskType] = None,
        signal: Optional[SignalType] = None,
        split: Optional[SplitType] = None,
        component_type: Optional[str] = None,
        max_entries: Optional[int] = None,
    ) -> list[DatasetEntry]:
        """Filter entries by criteria."""
        if not self._scanned:
            self.scan()

        results = self._entries

        if task is not None:
            results = [e for e in results if e.task == task]
        if signal is not None:
            results = [e for e in results if e.signal == signal]
        if split is not None:
            results = [e for e in results if e.split == split]
        if component_type is not None:
            results = [e for e in results if e.component_type == component_type]
        if max_entries is not None:
            results = results[:max_entries]

        return results

    def load_entry(
        self,
        entry: DatasetEntry,
        sample_rate: Optional[float] = None,
    ) -> ComplexSignal:
        """Load a single dataset entry into a ComplexSignal preserving native sample rate."""
        return entry.load(sample_rate=sample_rate)

    def load_batch(
        self,
        entries: list[DatasetEntry],
        sample_rate: Optional[float] = None,
    ) -> list[ComplexSignal]:
        """Load multiple entries into ComplexSignals preserving native sample rates."""
        return [entry.load(sample_rate=sample_rate) for entry in entries]
