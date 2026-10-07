"""M2 全链路：信号 -> M1 估计 -> 23 维特征 -> 分类（门控 + 物理回退）."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from spxh.core.classify.model import DEFAULT_MODEL_DIR, ModulationClassifier
from spxh.core.classify.rules import physical_classify
from spxh.core.estimate import AnalyzeResult
from spxh.core.features import FeatureConfig, FeatureSet, extract_features
from spxh.core.types import Signal

__all__ = ["classify_signal"]


def classify_signal(
    signal: Signal,
    classifier: Optional[ModulationClassifier] = None,
    model_dir=DEFAULT_MODEL_DIR,
    feature_config: Optional[FeatureConfig] = None,
    estimates: Optional[AnalyzeResult] = None,
    threshold: Optional[float] = None,
) -> dict[str, Any]:
    """完整分类链路，返回可直接 JSON 序列化的字典（含特征与证据）."""
    features: FeatureSet = extract_features(signal, estimates=estimates, config=feature_config)
    if classifier is None:
        classifier = ModulationClassifier.load(model_dir)

    if classifier is None:
        physical = physical_classify(signal, features)
        return {
            "modulation": physical.modulation,
            "probabilities": physical.scores,
            "confidence": physical.confidence,
            "gate": "fallback" if physical.modulation else "reject",
            "threshold": None,
            "calibrated": False,
            "ood": {"score": 0.0, "is_outlier": False, "method": "none", "threshold": 0.0},
            "physical": physical.to_dict(),
            "features": {
                "symbol_rate": float(features.evidence.get("symbol_rate", 0.0)),
                "cfo_hz": float(features.evidence.get("cfo_hz", 0.0)),
                "snr_db": float(features.evidence.get("es_n0_db", 0.0)),
            },
            "warnings": list(features.warnings) + ["未找到训练好的模型，仅使用物理规则回退"],
            "feature_set": features.as_dict(),
        }

    payload = classifier.predict(features, signal=signal, threshold=threshold)
    payload["feature_set"] = features.as_dict()
    return payload
