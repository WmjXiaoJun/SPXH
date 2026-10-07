"""M2 特征工程：五域 23 维特征.

域划分（顺序固定，训练与推理必须一致）：
  cumulant (4)          原始 I/Q 高阶累积量
  spectral (5)          频谱形状与带宽
  timefreq (3)          STFT 时频动态
  constellation (6)     星座几何（用估计的载频/符号速率盲恢复）
  cyclostationary (5)   循环平稳强度与谱线结构
"""
from spxh.core.features.extractor import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    FeatureConfig,
    FeatureSet,
    extract_features,
)

__all__ = ["FeatureSet", "FeatureConfig", "extract_features", "FEATURE_NAMES", "FEATURE_GROUPS"]
