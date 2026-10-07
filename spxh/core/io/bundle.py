"""npz bundle：把波形 + 元数据 + 真值数组打进单个文件，便于快速加载与回放."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from spxh.core.types import GroundTruth, Signal, jsonable

__all__ = ["write_bundle", "read_bundle", "bundle_path"]


def bundle_path(prefix) -> Path:
    path = Path(str(prefix))
    if path.suffix.lower() == ".npz":
        return path
    return Path(str(path) + ".spxh.npz")


def write_bundle(prefix, signal: Signal, truth: Optional[GroundTruth] = None) -> str:
    target = bundle_path(prefix)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "samples": signal.samples.astype(np.complex128),
        "sample_rate": np.asarray([signal.sample_rate], dtype=np.float64),
        "center_freq": np.asarray([np.nan if signal.center_freq is None else signal.center_freq], dtype=np.float64),
        "meta_json": np.asarray(json.dumps(jsonable(signal.meta), ensure_ascii=False)),
    }
    if truth is not None:
        arrays["truth_json"] = np.asarray(json.dumps(truth.to_dict(), ensure_ascii=False))
        for key, value in (
            ("bits", truth.bits),
            ("symbol_indices", truth.symbol_indices),
            ("symbols", truth.symbols),
            ("tone_hz", truth.tone_hz),
            ("clean", truth.clean),
        ):
            if value is not None:
                arrays["truth_" + key] = np.asarray(value)
    np.savez_compressed(target, **arrays)
    return str(target)


def read_bundle(path) -> tuple[Signal, Optional[GroundTruth]]:
    with np.load(str(path), allow_pickle=False) as data:
        samples = data["samples"]
        fs = float(data["sample_rate"][0]) if "sample_rate" in data.files else None
        fc = float(data["center_freq"][0]) if "center_freq" in data.files else np.nan
        meta = json.loads(str(data["meta_json"])) if "meta_json" in data.files else {}
        if fs is None:
            raise ValueError("bundle has no sample_rate")
        signal = Signal(
            samples=samples,
            sample_rate=fs,
            center_freq=None if (fc is None or np.isnan(fc)) else fc,
            meta=meta,
        )
        truth = None
        if "truth_json" in data.files:
            truth_dict = json.loads(str(data["truth_json"]))
            arrays = {}
            for key in ("bits", "symbol_indices", "symbols", "tone_hz", "clean"):
                name = "truth_" + key
                if name in data.files:
                    arrays[key] = data[name]
            truth = GroundTruth.from_dict(truth_dict, arrays)
    return signal, truth
