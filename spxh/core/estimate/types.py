"""估计结果容器：值 + 单位 + 置信度 + 方法 + 证据.

设计原则（对应方案中"每一步结果都要可解释、可回退"）：
每个估计值都必须带**方法名**与**证据字典**，下游可以据此判断是否采信，
分析员也可以覆盖参数后重跑。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from spxh.core.types import jsonable

__all__ = ["Estimate"]


@dataclass
class Estimate:
    name: str
    value: float
    unit: str = ""
    confidence: float = 0.0
    method: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.value = float(self.value)
        self.confidence = float(min(max(self.confidence, 0.0), 1.0))

    @property
    def valid(self) -> bool:
        return math.isfinite(self.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "evidence": jsonable(self.evidence),
        }

    def describe(self) -> str:
        return "{:s} = {:.4f} {:s} [方法={:s}, 置信度={:.2f}]".format(
            self.name, self.value, self.unit, self.method, self.confidence
        )
