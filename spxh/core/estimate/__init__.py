"""参数估计层：载频、符号速率、盲 SNR（M1）."""
from spxh.core.estimate.analyze import AnalyzeResult, analyze_signal, compare_with_truth
from spxh.core.estimate.cfo import cfo_centroid, cfo_edge_midpoint, cfo_symmetry, estimate_cfo, residual_spectrum
from spxh.core.estimate.snr import estimate_snr, estimate_snr_m2m4, estimate_snr_spectral
from spxh.core.estimate.symbol_rate import (
    cyclic_profile,
    envelope_profile,
    estimate_symbol_rate,
    harmonic_score,
)
from spxh.core.estimate.types import Estimate

__all__ = [
    "Estimate",
    "estimate_cfo",
    "cfo_centroid",
    "cfo_symmetry",
    "cfo_edge_midpoint",
    "residual_spectrum",
    "estimate_symbol_rate",
    "cyclic_profile",
    "envelope_profile",
    "harmonic_score",
    "estimate_snr",
    "estimate_snr_spectral",
    "estimate_snr_m2m4",
    "analyze_signal",
    "compare_with_truth",
    "AnalyzeResult",
]
