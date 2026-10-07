"""信号合成层：调制映射、成型滤波、损伤注入、信号生成器."""
from spxh.core.synth.generator import SynthConfig, synthesize
from spxh.core.synth.impairments import (
    add_awgn,
    apply_cfo,
    apply_dc_offset,
    apply_iq_imbalance,
    apply_phase_noise,
    complex_noise_variance,
    measure_snr_db,
)
from spxh.core.synth.mapping import (
    bits_per_symbol,
    bits_to_indices,
    constellation,
    indices_to_bits,
    indices_to_points,
    num_points,
    points_to_indices,
)
from spxh.core.synth.modulators import ModParams, modulate_bits, modulate_indices
from spxh.core.synth.pulse import matched_filter, pulse_shape, rrc_taps, upsample

__all__ = [
    "SynthConfig",
    "synthesize",
    "ModParams",
    "modulate_bits",
    "modulate_indices",
    "constellation",
    "num_points",
    "bits_per_symbol",
    "bits_to_indices",
    "indices_to_bits",
    "indices_to_points",
    "points_to_indices",
    "rrc_taps",
    "upsample",
    "pulse_shape",
    "matched_filter",
    "add_awgn",
    "complex_noise_variance",
    "apply_cfo",
    "apply_phase_noise",
    "apply_iq_imbalance",
    "apply_dc_offset",
    "measure_snr_db",
]
