"""特征容器：名称 + 数值 + 域分组 + 证据."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from spxh.core.types import jsonable

__all__ = ["FeatureSet"]


@dataclass
class FeatureSet:
    names: list[str]
    values: np.ndarray
    groups: dict[str, list[int]]
    evidence: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64).reshape(-1)
        if self.values.size != len(self.names):
            raise ValueError("values size " + str(self.values.size) + " != names " + str(len(self.names)))
        if not np.all(np.isfinite(self.values)):
            bad = [self.names[i] for i in range(self.values.size) if not np.isfinite(self.values[i])]
            self.warnings.append("非有限特征值已置零: " + ", ".join(bad))
            self.values = np.nan_to_num(self.values, nan=0.0, posinf=0.0, neginf=0.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "values": [float(v) for v in self.values],
            "groups": {k: list(v) for k, v in self.groups.items()},
            "values_by_name": {name: float(self.values[i]) for i, name in enumerate(self.names)},
            "evidence": jsonable(self.evidence),
            "warnings": list(self.warnings),
        }

    def vector(self) -> np.ndarray:
        return self.values.copy()

    def __getitem__(self, name: str) -> float:
        return float(self.values[self.names.index(name)])
