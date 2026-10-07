"""STFT 时频域特征：包络动态、突发性、频率游走."""
from __future__ import annotations

import numpy as np

from spxh.core.dsp.timefreq import Waterfall

__all__ = ["timefreq_features"]


def envelope_cv(samples) -> float:
    """原始复基带包络的变异系数 std(|x|)/mean(|x|).

    这是区分"恒模"与"非恒模"最鲁棒的物理量：
      FSK（恒模，无成型）在高信噪比下约 0.19（残余只来自噪声），
      RRC 成型的 PSK/QAM 在 0.34~0.44；信噪比越低两者越趋同（噪声主导，都趋向 0.52）。
    因此只能作为**多条证据之一**，不能单独定案。
    """
    envelope = np.abs(np.asarray(samples).reshape(-1))
    if envelope.size == 0:
        return 0.0
    mean = float(np.mean(envelope))
    if mean <= 0:
        return 0.0
    return float(np.std(envelope) / mean)


def timefreq_features(waterfall: Waterfall, symbol_rate: float, samples=None) -> dict[str, float]:
    magnitude = np.asarray(waterfall.magnitude_db, dtype=np.float64)
    env_cv = envelope_cv(samples) if samples is not None else 0.0
    freqs = np.asarray(waterfall.freqs, dtype=np.float64)
    if magnitude.size == 0:
        return {"tf_env_var": 0.0, "tf_env_cv": env_cv, "tf_centroid_std": 0.0}

    linear = 10.0 ** (magnitude / 10.0)
    frame_power = np.sum(linear, axis=0)
    mean_power = float(np.mean(frame_power))
    if mean_power <= 0:
        return {"tf_env_var": 0.0, "tf_env_cv": env_cv, "tf_centroid_std": 0.0}

    env_var = float(np.var(frame_power) / (mean_power ** 2))

    weight_sum = np.sum(linear, axis=0)
    safe = np.maximum(weight_sum, 1e-300)
    centroids = (linear * freqs[:, None]).sum(axis=0) / safe
    centroid_std = float(np.std(centroids) / symbol_rate) if symbol_rate > 0 else 0.0

    return {
        "tf_env_var": env_var,
        "tf_env_cv": env_cv,
        "tf_centroid_std": centroid_std,
    }
