"""Core data models for the RF Analysis pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class DataType(str, Enum):
    """Supported IQ data types."""
    CF32 = "cf32"
    CI16 = "ci16"
    CI8 = "ci8"
    CU8 = "cu8"
    PLANAR_FLOAT32 = "planar_float32"


class IQOrdering(str, Enum):
    """IQ sample ordering."""
    INTERLEAVED = "interleaved"
    PLANAR = "planar"


class ProvenanceType:
    """String constants for ProvenanceStep.provenance_type.

    Use these constants when tagging how a pipeline measurement was obtained:
      - SIGNAL_INFERRED  : derived algorithmically from the IQ data (default)
      - METADATA_PROVIDED: value taken verbatim from the signal metadata/SigMF sidecar
      - SYNTHETIC_GROUND_TRUTH: known synthetic reference (e.g. validate_*.py ground truth)
      - USER_OVERRIDE    : analyst manually corrected or confirmed the estimate
    """
    SIGNAL_INFERRED = "signal_inferred"
    METADATA_PROVIDED = "metadata_provided"
    SYNTHETIC_GROUND_TRUTH = "synthetic_ground_truth"
    USER_OVERRIDE = "user_override"


@dataclass
class ProvenanceStep:
    """A single processing step in the provenance chain.

    Attributes:
        stage              : Pipeline stage name (e.g. "detection", "classification").
        operation          : Specific operation performed (e.g. "estimate_snr").
        parameters         : Key/value map of inputs / outputs / hyper-parameters.
        timestamp          : Unix timestamp of the step.
        provenance_type    : How the measurement was obtained (see ProvenanceType).
        measurement_trust  : False when the step's own internal consistency checks
                             indicate the result is unreliable (e.g. symbol rate > BW).
                             Downstream stages MUST surface this flag; they may weight
                             evidence from untrusted steps at zero.
    """
    stage: str
    operation: str
    parameters: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    provenance_type: str = ProvenanceType.SIGNAL_INFERRED
    measurement_trust: bool = True

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "operation": self.operation,
            "parameters": self.parameters,
            "timestamp": self.timestamp,
            "provenance_type": self.provenance_type,
            "measurement_trust": self.measurement_trust,
        }


@dataclass
class SignalMetadata:
    """Metadata about the raw signal file and its parameters."""
    filename: str
    file_size_bytes: int
    data_type: DataType
    sample_rate: Optional[float] = None
    center_frequency: Optional[float] = None
    iq_ordering: IQOrdering = IQOrdering.INTERLEAVED
    sigmf_metadata: Optional[dict] = None
    num_samples: int = 0
    duration_seconds: float = 0.0
    dataset_origin: str = "unknown"
    native_parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "file_size_bytes": self.file_size_bytes,
            "data_type": self.data_type.value,
            "sample_rate": self.sample_rate,
            "center_frequency": self.center_frequency,
            "iq_ordering": self.iq_ordering.value,
            "sigmf_metadata": self.sigmf_metadata,
            "num_samples": self.num_samples,
            "duration_seconds": self.duration_seconds,
            "dataset_origin": self.dataset_origin,
            "native_parameters": self.native_parameters,
        }


@dataclass
class ComplexSignal:
    """Core representation of a complex IQ signal.
    
    All signals are internally stored as complex64 (np.complex64)
    regardless of original format.
    """
    data: np.ndarray  # complex64 numpy array
    metadata: SignalMetadata
    provenance: list[ProvenanceStep] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.data, np.ndarray):
            raise TypeError("data must be a numpy array")
        if self.data.dtype != np.complex64:
            raise ValueError(f"data must be complex64, got {self.data.dtype}")

    @property
    def num_samples(self) -> int:
        return len(self.data)

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.metadata.sample_rate and self.metadata.sample_rate > 0:
            return self.num_samples / self.metadata.sample_rate
        return None

    def add_provenance(
        self,
        stage: str,
        operation: str,
        parameters: Optional[dict] = None,
        trusted: bool = True,
        provenance_type: str = ProvenanceType.SIGNAL_INFERRED,
    ):
        """Record a processing step in the provenance chain.

        Args:
            stage           : Pipeline stage name.
            operation       : Specific operation label.
            parameters      : Dict of relevant key/value pairs to record.
            trusted         : Set False when the estimate failed internal sanity
                              checks (e.g. symbol_rate > bandwidth) so downstream
                              consumers can discount this measurement.
            provenance_type : One of ProvenanceType.* constants.
        """
        self.provenance.append(
            ProvenanceStep(
                stage=stage,
                operation=operation,
                parameters=parameters or {},
                provenance_type=provenance_type,
                measurement_trust=trusted,
            )
        )

    def to_dict(self) -> dict:
        return {
            "num_samples": self.num_samples,
            "dtype": str(self.data.dtype),
            "metadata": self.metadata.to_dict(),
            "provenance": [p.to_dict() for p in self.provenance],
        }
