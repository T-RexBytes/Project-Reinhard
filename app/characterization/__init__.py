"""Signal characterization layer — features, symbol rate, frequency offset."""

from app.characterization.features import extract_features, FeatureVector
from app.characterization.symbol_rate import estimate_symbol_rate, SymbolRateResult
from app.characterization.freq_offset import estimate_freq_offset, FreqOffsetResult

__all__ = [
    "extract_features", "FeatureVector",
    "estimate_symbol_rate", "SymbolRateResult",
    "estimate_freq_offset", "FreqOffsetResult",
]
