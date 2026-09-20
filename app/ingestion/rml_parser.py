"""
app/ingestion/rml_parser.py
----------------------------
Memory-safe ingestion parser for the RadioML 2016.10a dataset (RML2016.10a_dict.pkl).

Key Invariants:
1. Streaming / lazy frame access: Never instantiates 220,000 ComplexSignal objects in RAM simultaneously.
2. Metadata Preservation: Retains ground-truth modulation, labeled SNR (dB), frame index,
   planar float32 structure, and origin ("rml2016.10a").
3. SigMF Export: Ability to slice a calibrated benchmark set to disk with full SigMF metadata.
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import numpy as np

from app.models import (
    ComplexSignal,
    DataType,
    IQOrdering,
    ProvenanceStep,
    SignalMetadata,
)

DEFAULT_RML_PATH = Path("D:/dataset/rml/RML2016.10a_dict.pkl")
DEFAULT_RML_SAMPLE_RATE = 1_000_000.0  # 1.0 MHz normalized baseband


class RMLDatasetReader:
    """Memory-safe reader for RML2016.10a dictionary pickle files."""

    def __init__(self, pkl_path: Union[str, Path] = DEFAULT_RML_PATH):
        self.pkl_path = Path(pkl_path)
        self._data: Optional[dict[tuple[str, int], np.ndarray]] = None
        self._keys: Optional[list[tuple[str, int]]] = None
        self._modulations: Optional[list[str]] = None
        self._snrs: Optional[list[int]] = None

    def _ensure_loaded(self):
        """Lazy load the pickle file if not already in memory."""
        if self._data is None:
            if not self.pkl_path.exists():
                raise FileNotFoundError(f"RML pickle dataset not found at {self.pkl_path}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open(self.pkl_path, "rb") as f:
                    self._data = pickle.load(f, encoding="latin1")

            self._keys = sorted(list(self._data.keys()))
            self._modulations = sorted(list(set(k[0] for k in self._keys)))
            self._snrs = sorted(list(set(k[1] for k in self._keys)))

    @property
    def modulations(self) -> list[str]:
        self._ensure_loaded()
        return list(self._modulations)

    @property
    def snrs(self) -> list[int]:
        self._ensure_loaded()
        return list(self._snrs)

    @property
    def keys(self) -> list[tuple[str, int]]:
        self._ensure_loaded()
        return list(self._keys)

    def get_frame_count(self, mod: str, snr_db: int) -> int:
        self._ensure_loaded()
        key = (mod, snr_db)
        if key not in self._data:
            raise KeyError(f"Key {key} not found in RML dataset")
        return len(self._data[key])

    def get_frame(
        self,
        mod: str,
        snr_db: int,
        frame_idx: int = 0,
        sample_rate: float = DEFAULT_RML_SAMPLE_RATE,
    ) -> ComplexSignal:
        """Load a single frame as a ComplexSignal with complete native metadata."""
        self._ensure_loaded()
        key = (mod, snr_db)
        if key not in self._data:
            raise KeyError(f"Key {key} not found in RML dataset")

        frames = self._data[key]
        if frame_idx < 0 or frame_idx >= len(frames):
            raise IndexError(f"Frame index {frame_idx} out of range [0, {len(frames)})")

        raw_iq = frames[frame_idx]  # shape: (2, 128), float32
        # Channel 0 is I, Channel 1 is Q
        i_chan = raw_iq[0].astype(np.float32)
        q_chan = raw_iq[1].astype(np.float32)
        complex_samples = (i_chan + 1j * q_chan).astype(np.complex64)

        snr_str = f"{snr_db:+03d}dB" if snr_db != 0 else "+00dB"
        fname = f"rml_{mod}_{snr_str}_frame{frame_idx:04d}"

        meta = SignalMetadata(
            filename=fname,
            file_size_bytes=complex_samples.nbytes,
            data_type=DataType.PLANAR_FLOAT32,
            sample_rate=sample_rate,
            center_frequency=0.0,
            iq_ordering=IQOrdering.PLANAR,
            sigmf_metadata={
                "global": {
                    "core:datatype": "cf32_le",
                    "core:sample_rate": sample_rate,
                    "core:author": "DeepSig Inc. / RadioML 2016.10a",
                    "core:description": f"RadioML synthetic transmission: {mod} @ {snr_db} dB SNR",
                },
                "captures": [
                    {
                        "core:frequency": 0.0,
                        "core:sample_start": 0,
                    }
                ],
            },
            num_samples=len(complex_samples),
            duration_seconds=len(complex_samples) / sample_rate,
            dataset_origin="rml2016.10a",
            native_parameters={
                "ground_truth_mod": mod,
                "labeled_snr_db": snr_db,
                "frame_index": frame_idx,
                "source_shape": list(raw_iq.shape),
                "source_dtype": str(raw_iq.dtype),
                "channel_simulated": True,
            },
        )

        sig = ComplexSignal(data=complex_samples, metadata=meta)
        sig.add_provenance(
            stage="ingestion",
            operation="load_rml_frame",
            parameters={
                "modulation": mod,
                "snr_db": snr_db,
                "frame_index": frame_idx,
                "samples": len(complex_samples),
            },
        )
        return sig

    def iter_frames(
        self,
        mods: Optional[list[str]] = None,
        snrs: Optional[list[int]] = None,
        max_per_key: int = 10,
        sample_rate: float = DEFAULT_RML_SAMPLE_RATE,
    ) -> Iterator[ComplexSignal]:
        """Yield frames one-by-one to preserve memory."""
        self._ensure_loaded()
        target_mods = mods or self.modulations
        target_snrs = snrs or self.snrs

        for mod in target_mods:
            for snr in target_snrs:
                key = (mod, snr)
                if key not in self._data:
                    continue
                total = len(self._data[key])
                limit = min(max_per_key, total)
                for idx in range(limit):
                    yield self.get_frame(mod, snr, idx, sample_rate=sample_rate)


def load_single_rml_frame(
    pkl_path: Union[str, Path] = DEFAULT_RML_PATH,
    mod: str = "QPSK",
    snr_db: int = 10,
    frame_idx: int = 0,
) -> ComplexSignal:
    """Convenience helper to load one frame without managing reader state."""
    reader = RMLDatasetReader(pkl_path)
    return reader.get_frame(mod, snr_db, frame_idx)


def slice_rml_dataset(
    pkl_path: Union[str, Path] = DEFAULT_RML_PATH,
    output_dir: Union[str, Path] = "D:/dataset/rml_frames",
    frames_per_key: int = 5,
    mods: Optional[list[str]] = None,
    snrs: Optional[list[int]] = None,
) -> list[Path]:
    """Slice a balanced benchmark set from the RML pickle into standard SigMF files.
    
    Each frame is written as:
      <out_dir>/rml_<mod>_<snr>dB_<idx>.sigmf-data (interleaved complex64)
      <out_dir>/rml_<mod>_<snr>dB_<idx>.sigmf-meta (JSON sidecar)
    """
    reader = RMLDatasetReader(pkl_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    created_files: list[Path] = []
    for sig in reader.iter_frames(mods=mods, snrs=snrs, max_per_key=frames_per_key):
        base_name = sig.metadata.filename
        data_file = out_path / f"{base_name}.sigmf-data"
        meta_file = out_path / f"{base_name}.sigmf-meta"

        # Interleaved complex64 binary
        sig.data.tofile(str(data_file))

        # SigMF metadata sidecar preserving all native parameters
        sigmf_dict = {
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": sig.metadata.sample_rate,
                "core:version": "0.0.2",
                "core:author": "DeepSig Inc. / RadioML 2016.10a",
                "core:description": f"RML2016.10a {sig.metadata.native_parameters.get('ground_truth_mod')} @ {sig.metadata.native_parameters.get('labeled_snr_db')}dB",
                "sih:dataset_origin": "rml2016.10a",
                "sih:native_parameters": sig.metadata.native_parameters,
            },
            "captures": [
                {
                    "core:sample_start": 0,
                    "core:frequency": sig.metadata.center_frequency,
                }
            ],
            "annotations": [],
        }

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(sigmf_dict, f, indent=2)

        created_files.append(data_file)

    return created_files
