"""M2 调制识别：随机森林 + 概率校准 + 置信度门控 + 物理规则回退."""
from spxh.core.classify.model import ModulationClassifier, ModelMetadata, train_classifier
from spxh.core.classify.pipeline import classify_signal
from spxh.core.classify.rules import PhysicalDecision, count_if_tones, physical_classify

__all__ = [
    "ModulationClassifier",
    "ModelMetadata",
    "train_classifier",
    "classify_signal",
    "physical_classify",
    "PhysicalDecision",
    "count_if_tones",
]
