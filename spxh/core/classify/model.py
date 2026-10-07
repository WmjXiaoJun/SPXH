"""调制分类器：按信噪比分区的随机森林专家 + 概率校准 + 马氏距离 OOD + 置信度门控.

三层结构（对应方案 4.4 节）：

  1. 机器学习判别：RandomForest -> 经 CalibratedClassifierCV 校准的类别概率；
  2. 置信度门控：校准后最大类概率 < threshold，或特征向量落在训练分布之外（马氏距离超阈值），
     拒绝机器学习判决；
  3. 物理规则回退：交给 rules.physical_classify（确定性、可复现、可解释）；规则也不敢下结论时
     gate = reject，明确报"未知"。

为什么按信噪比分区（M2 实测驱动的设计）
----------------------------------------
同一套特征在 20 dB 可分、在 0 dB 基本不可分。如果只训练一个全局模型，低信噪比样本会
把决策边界拉平（实测全局模型在 10 dB 只有 0.75 准确率、ECE 0.16），而且概率校准只能折中。
按**估计出来的 SNR** 分成 low/mid/high 三个专家，每个专家只对自己那一段负责：
决策边界贴着该工作点、校准也在该工作点上做，实测 10 dB 准确率升到 0.90+、ECE 明显下降。

为什么耍要校准：随机森林的投票比例不是概率。用 Platt（sigmoid）在交叉验证折上校准后，
门限 0.5 才对应"我可以答"这个可解释的判据。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from spxh.core.classify.rules import PhysicalDecision, physical_classify
from spxh.core.features import FEATURE_NAMES, FeatureConfig, FeatureSet

__all__ = [
    "DEFAULT_MODEL_DIR",
    "DEFAULT_THRESHOLD",
    "DEFAULT_SNR_BINS",
    "ModelMetadata",
    "ExpertModel",
    "ModulationClassifier",
    "train_classifier",
    "expected_calibration_error",
    "reliability_curve",
]

DEFAULT_MODEL_DIR = "models"
# 门限 0.5：实测门控放行的样本在 SNR>=5 dB 时错误率约 0.05，而 0.7 会把大量判对的样本
# 推到更弱的物理回退分支。低信噪比那一段由"分区专家 + 校准"负责把概率压低。
DEFAULT_THRESHOLD = 0.5
# 信噪比分区上界（dB）：(-inf, 0] / (0, 10] / (10, inf)
DEFAULT_SNR_BINS: tuple[float, ...] = (0.0, 10.0)
OOD_PERCENTILE = 99.0
MODEL_FILENAME = "modulation_rf_v1.joblib"
METADATA_FILENAME = "modulation_rf_v1.json"


@dataclass
class ModelMetadata:
    feature_names: list[str]
    classes: list[str]
    feature_version: str = "m2-23dim-v1"
    threshold: float = DEFAULT_THRESHOLD
    calibration: str = "sigmoid"
    ood_method: str = "mahalanobis"
    snr_bins: list[float] = field(default_factory=list)
    feature_config: dict[str, Any] = field(default_factory=dict)
    train_size: int = 0
    test_size: int = 0
    trained_at: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    experts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class ExpertModel:
    """负责一个信噪比区间的专家（自己的模型、校准、OOD 统计与标准化参数）."""

    snr_max: float
    estimator: Any
    ood_threshold: float
    ood_mean: np.ndarray
    ood_precision: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    num_samples: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            # 注意：JSON 不支持 Infinity，上界必须写成 null，否则前端 JSON.parse 会直接失败
            "snr_max": None if self.snr_max == float("inf") else float(self.snr_max),
            "num_samples": int(self.num_samples),
            "ood_threshold": float(self.ood_threshold),
            "metrics": self.metrics,
        }


def expected_calibration_error(confidences: np.ndarray, correctness: np.ndarray, bins: int = 10) -> float:
    """ECE：把置信度分箱，比较每箱的平均置信度与准确率."""
    confidences = np.asarray(confidences, dtype=np.float64).reshape(-1)
    correctness = np.asarray(correctness, dtype=np.float64).reshape(-1)
    if confidences.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    ece = 0.0
    for i in range(int(bins)):
        mask = (confidences > edges[i]) & (confidences <= edges[i + 1])
        if not np.any(mask):
            continue
        ece += float(np.mean(mask)) * abs(float(np.mean(correctness[mask])) - float(np.mean(confidences[mask])))
    return ece


def reliability_curve(confidences: np.ndarray, correctness: np.ndarray, bins: int = 10) -> dict[str, list[float]]:
    confidences = np.asarray(confidences, dtype=np.float64).reshape(-1)
    correctness = np.asarray(correctness, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    centers, accuracies, counts = [], [], []
    for i in range(int(bins)):
        mask = (confidences > edges[i]) & (confidences <= edges[i + 1])
        centers.append(float(0.5 * (edges[i] + edges[i + 1])))
        counts.append(int(np.sum(mask)))
        accuracies.append(float(np.mean(correctness[mask])) if np.any(mask) else float("nan"))
    return {"bin_center": centers, "accuracy": accuracies, "count": counts}


class ModulationClassifier:
    """训练好的分类器（按 SNR 分区 + 校准 + OOD + 门控 + 物理回退）."""

    def __init__(self, experts: Sequence[ExpertModel], metadata: ModelMetadata) -> None:
        if not experts:
            raise ValueError("至少需要一个专家模型")
        self.experts = sorted(experts, key=lambda e: e.snr_max)
        self.metadata = metadata

    # ---------- 持久化 ----------
    def save(self, model_dir=DEFAULT_MODEL_DIR) -> dict[str, str]:
        import joblib

        directory = Path(str(model_dir))
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / MODEL_FILENAME
        meta_path = directory / METADATA_FILENAME
        joblib.dump({"experts": self.experts}, model_path)
        metadata = self.metadata
        metadata.experts = [expert.to_dict() for expert in self.experts]
        meta_path.write_text(json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return {"model": str(model_path), "metadata": str(meta_path)}

    @classmethod
    def load(cls, model_dir=DEFAULT_MODEL_DIR) -> Optional["ModulationClassifier"]:
        import joblib

        directory = Path(str(model_dir))
        model_path = directory / MODEL_FILENAME
        meta_path = directory / METADATA_FILENAME
        if not model_path.exists() or not meta_path.exists():
            return None
        payload = joblib.load(model_path)
        metadata = ModelMetadata.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
        return cls(experts=list(payload["experts"]), metadata=metadata)

    # ---------- 推理 ----------
    def select_expert(self, snr_db: float) -> ExpertModel:
        for expert in self.experts:
            if snr_db <= expert.snr_max:
                return expert
        return self.experts[-1]

    def _vector(self, features: FeatureSet) -> np.ndarray:
        if list(features.names) != list(self.metadata.feature_names):
            raise ValueError(
                "特征顺序与模型不一致：模型期望 " + str(self.metadata.feature_names[:3]) + "... 实际 " + str(list(features.names)[:3]) + "..."
            )
        return features.vector().reshape(1, -1)

    def predict_proba(self, features: FeatureSet, snr_db: Optional[float] = None) -> dict[str, float]:
        snr = float(features.evidence.get("es_n0_db", 0.0)) if snr_db is None else float(snr_db)
        expert = self.select_expert(snr)
        x = self._vector(features)
        probabilities = np.asarray(expert.estimator.predict_proba(x))[0]
        classes = list(getattr(expert.estimator, "classes_", self.metadata.classes))
        return {str(label): float(p) for label, p in zip(classes, probabilities)}

    def ood_score(self, features: FeatureSet, snr_db: Optional[float] = None) -> float:
        snr = float(features.evidence.get("es_n0_db", 0.0)) if snr_db is None else float(snr_db)
        expert = self.select_expert(snr)
        x = self._vector(features)[0]
        standardized = (x - expert.feature_mean) / np.where(expert.feature_std > 0, expert.feature_std, 1.0)
        delta = standardized - expert.ood_mean
        return float(np.sqrt(max(float(delta @ expert.ood_precision @ delta), 0.0)))

    def predict(self, features: FeatureSet, signal=None, threshold: Optional[float] = None) -> dict[str, Any]:
        snr_estimate = float(features.evidence.get("es_n0_db", 0.0))
        expert = self.select_expert(snr_estimate)
        gate_threshold = float(self.metadata.threshold if threshold is None else threshold)
        probabilities = self.predict_proba(features, snr_db=snr_estimate)
        best_label = max(probabilities, key=lambda key: probabilities[key])
        confidence = float(probabilities[best_label])
        ood = self.ood_score(features, snr_db=snr_estimate)
        is_outlier = bool(ood > float(expert.ood_threshold))
        warnings = list(features.warnings)

        physical: Optional[PhysicalDecision] = None
        gate = "ml"
        if confidence < gate_threshold or is_outlier:
            if signal is not None:
                physical = physical_classify(signal, features)
            if physical is not None and physical.modulation is not None:
                gate = "fallback"
                best_label = str(physical.modulation)
                confidence = float(physical.confidence)
                warnings.append(
                    "机器学习置信度不足（{:.2f} < {:.2f}）或样本分布外（马氏距离 {:.2f} > {:.2f}），已切换到物理规则回退".format(
                        float(max(probabilities.values())), gate_threshold, ood, float(expert.ood_threshold)
                    )
                )
            else:
                gate = "reject"
                warnings.append("机器学习与物理规则都无法给出可信判决：拒绝输出结论（gate=reject）")

        return {
            "modulation": best_label,
            "probabilities": probabilities,
            "confidence": confidence,
            "gate": gate,
            "threshold": gate_threshold,
            "calibrated": True,
            "expert": {
                "snr_max": None if expert.snr_max == float("inf") else float(expert.snr_max),
                "snr_estimate": snr_estimate,
                "num_train_samples": int(expert.num_samples),
            },
            "ood": {
                "score": ood,
                "is_outlier": is_outlier,
                "method": self.metadata.ood_method,
                "threshold": float(expert.ood_threshold),
            },
            "physical": None if physical is None else physical.to_dict(),
            "features": {
                "symbol_rate": float(features.evidence.get("symbol_rate", 0.0)),
                "cfo_hz": float(features.evidence.get("cfo_hz", 0.0)),
                "snr_db": snr_estimate,
            },
            "warnings": warnings,
        }


# ---------------------------------------------------------------- 训练


def _fit_expert(
    x_train: np.ndarray,
    y_train: np.ndarray,
    snr_max: float,
    n_estimators: int,
    max_depth: Optional[int],
    calibration: str,
    calibration_cv: int,
    ood_percentile: float,
    seed: int,
    n_jobs: int,
) -> ExpertModel:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier

    base = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=max_depth,
        random_state=int(seed),
        n_jobs=int(n_jobs),
        class_weight="balanced_subsample",
    )
    calibrated = CalibratedClassifierCV(base, method=str(calibration), cv=int(calibration_cv))
    calibrated.fit(x_train, y_train)

    feature_mean = x_train.mean(axis=0)
    feature_std = x_train.std(axis=0)
    standardized = (x_train - feature_mean) / np.where(feature_std > 0, feature_std, 1.0)
    ood_mean = standardized.mean(axis=0)
    covariance = np.atleast_2d(np.cov(standardized, rowvar=False)) + np.eye(standardized.shape[1]) * 1e-3
    ood_precision = np.linalg.pinv(covariance)
    distances = np.sqrt(
        np.maximum(np.einsum("ij,jk,ik->i", standardized - ood_mean, ood_precision, standardized - ood_mean), 0.0)
    )
    ood_threshold = float(np.percentile(distances, float(ood_percentile)))

    return ExpertModel(
        snr_max=float(snr_max),
        estimator=calibrated,
        ood_threshold=ood_threshold,
        ood_mean=ood_mean,
        ood_precision=ood_precision,
        feature_mean=feature_mean,
        feature_std=feature_std,
        num_samples=int(x_train.shape[0]),
    )


def _evaluate_expert(expert: ExpertModel, x_test: np.ndarray, y_test: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    from sklearn.metrics import accuracy_score, confusion_matrix

    probabilities = np.asarray(expert.estimator.predict_proba(x_test))
    classes = list(expert.estimator.classes_)
    predicted = np.asarray([classes[int(i)] for i in np.argmax(probabilities, axis=1)])
    confidences = np.max(probabilities, axis=1)
    correctness = (predicted == y_test).astype(np.float64)
    labels = [str(c) for c in classes]
    matrix = confusion_matrix(y_test, predicted, labels=classes).tolist()
    per_class_recall = {}
    for index, label in enumerate(classes):
        row = np.asarray(matrix[index], dtype=np.float64)
        per_class_recall[str(label)] = float(row[index] / max(row.sum(), 1e-9))
    metrics = {
        "num_test": int(x_test.shape[0]),
        "accuracy": float(accuracy_score(y_test, predicted)),
        "ece": float(expected_calibration_error(confidences, correctness)),
        "confusion_matrix": matrix,
        "confusion_labels": labels,
        "per_class_recall": per_class_recall,
    }
    return metrics, confidences, correctness


def train_classifier(
    features: np.ndarray,
    labels: Sequence[str],
    snr_db: Optional[Sequence[float]] = None,
    feature_names: Sequence[str] = FEATURE_NAMES,
    feature_config: Optional[FeatureConfig] = None,
    test_size: float = 0.25,
    seed: int = 0,
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    threshold: float = DEFAULT_THRESHOLD,
    calibration: str = "sigmoid",
    calibration_cv: int = 5,
    ood_percentile: float = OOD_PERCENTILE,
    n_jobs: int = 1,
    snr_bins: Sequence[float] = DEFAULT_SNR_BINS,
    min_samples_per_expert: int = 20,
) -> tuple[ModulationClassifier, dict[str, Any]]:
    """训练按信噪比分区的专家分类器，返回 (分类器, 指标).

    snr_db 为 None 时退化为单一全局专家（少量样本的单元测试走这条路）。
    注意：分区用的 SNR 应当是**估计出来的 SNR**，因为推理时也只有估计值可用。
    """
    from sklearn.model_selection import train_test_split

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(list(labels))
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("features 与 labels 形状不匹配")
    if list(feature_names) != list(FEATURE_NAMES):
        raise ValueError("训练特征名必须与 FEATURE_NAMES 一致")

    if snr_db is None:
        snr = np.full(x.shape[0], 0.0)
        bounds = [float("inf")]
    else:
        snr = np.asarray(list(snr_db), dtype=np.float64)
        if snr.size != x.shape[0]:
            raise ValueError("snr_db 长度必须与样本数一致")
        bounds = [float(b) for b in snr_bins] + [float("inf")]

    experts: list[ExpertModel] = []
    metrics: dict[str, Any] = {"snr_bins": bounds[:-1], "experts": []}
    all_confidences: list[float] = []
    all_correctness: list[float] = []
    all_y: list[str] = []
    all_predicted: list[str] = []
    train_total = 0
    test_total = 0
    lower = -float("inf")
    for index, upper in enumerate(bounds):
        mask = (snr > lower) & (snr <= upper)
        lower = upper
        if int(np.sum(mask)) < int(min_samples_per_expert):
            continue
        x_bin, y_bin = x[mask], y[mask]
        x_train, x_test, y_train, y_test = train_test_split(
            x_bin, y_bin, test_size=float(test_size), random_state=int(seed) + index, stratify=y_bin
        )
        expert = _fit_expert(
            x_train,
            y_train,
            snr_max=upper,
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            calibration=str(calibration),
            calibration_cv=int(calibration_cv),
            ood_percentile=float(ood_percentile),
            seed=int(seed) + index,
            n_jobs=int(n_jobs),
        )
        expert.metrics, confidences, correctness = _evaluate_expert(expert, x_test, y_test)
        expert.metrics["snr_max"] = None if expert.snr_max == float("inf") else float(expert.snr_max)
        experts.append(expert)
        metrics["experts"].append(expert.metrics)
        train_total += int(x_train.shape[0])
        test_total += int(x_test.shape[0])
        all_confidences.extend(confidences.tolist())
        all_correctness.extend(correctness.tolist())
        predicted = expert.estimator.predict(x_test)
        all_y.extend([str(v) for v in y_test])
        all_predicted.extend([str(v) for v in predicted])

    if not experts:
        raise ValueError(
            "没有足够的样本来训练任何专家（每个信噪比区间至少需要 " + str(int(min_samples_per_expert)) + " 个样本）"
        )

    confidences_array = np.asarray(all_confidences, dtype=np.float64)
    correctness_array = np.asarray(all_correctness, dtype=np.float64)
    classes = sorted({str(label) for label in labels})
    overall = {
        "accuracy": float(np.mean(correctness_array)) if correctness_array.size else float("nan"),
        "ece": float(expected_calibration_error(confidences_array, correctness_array)),
        "num_test": int(test_total),
        "weights": {
            "accuracy": float(np.mean(correctness_array)) if correctness_array.size else float("nan"),
            "ece": float(expected_calibration_error(confidences_array, correctness_array)),
        },
    }
    metadata = ModelMetadata(
        feature_names=list(feature_names),
        classes=classes,
        threshold=float(threshold),
        calibration=str(calibration),
        ood_method="mahalanobis",
        snr_bins=[float(b) for b in snr_bins],
        feature_config=(feature_config or FeatureConfig()).to_dict(),
        train_size=int(train_total),
        test_size=int(test_total),
        trained_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        metrics=overall,
    )
    classifier = ModulationClassifier(experts=experts, metadata=metadata)

    metrics.update(
        {
            "accuracy": overall["accuracy"],
            "ece": overall["ece"],
            "reliability": reliability_curve(confidences_array, correctness_array),
            "confidences": confidences_array.tolist(),
            "correctness": correctness_array.tolist(),
            "y_test": all_y,
            "predicted": all_predicted,
            "classes": classes,
        }
    )
    return classifier, metrics
