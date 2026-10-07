"""SigMF（Signal Metadata Format）读写，v1.0.0.

参考：https://sigmf.org/ 。本模块支持 cf32/cf64/cf64/ci16/ci8/rf32/rf64/ri16 的
little-endian 变体，并把 SPXH 的真值摘要写入扩展键 spxh:ground_truth
（完整真值在 <prefix>.truth.json/.truth.npz）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from spxh.core.types import GroundTruth, Signal, jsonable

__all__ = [
    "SIGMF_VERSION",
    "SIGMF_DATATYPES",
    "sigmf_paths",
    "write_sigmf",
    "read_sigmf",
    "make_annotations",
]

SIGMF_VERSION = "1.0.0"

SIGMF_DATATYPES: dict[str, dict[str, Any]] = {
    "cf32_le": {"numpy": "<c8", "kind": "complex", "itemsize": 8},
    "cf64_le": {"numpy": "<c16", "kind": "complex", "itemsize": 16},
    "ci16_le": {"numpy": "<i2", "kind": "complex_int", "itemsize": 4, "full_scale": 32768.0},
    "ci8_le": {"numpy": "i1", "kind": "complex_int", "itemsize": 2, "full_scale": 128.0},
    "rf32_le": {"numpy": "<f4", "kind": "real", "itemsize": 4},
    "rf64_le": {"numpy": "<f8", "kind": "real", "itemsize": 8},
    "ri16_le": {"numpy": "<i2", "kind": "real_int", "itemsize": 2, "full_scale": 32768.0},
}


def sigmf_paths(prefix) -> tuple[Path, Path]:
    """由前缀或任一分片路径得到 (meta, data) 两个路径."""
    path = Path(str(prefix))
    name = path.name
    for suffix in (".sigmf-meta", ".sigmf-data"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    base = path.with_name(name)
    return Path(str(base) + ".sigmf-meta"), Path(str(base) + ".sigmf-data")


def make_annotations(truth: GroundTruth) -> list[dict[str, Any]]:
    """由真值生成 SigMF annotations（帧边界）."""
    annotations = []
    for frame in truth.frames:
        annotations.append(
            {
                "core:sample_start": int(frame.start_sample),
                "core:sample_count": int(frame.num_samples),
                "core:label": "frame{}:{}:payload{}B".format(
                    frame.index, frame.header.modulation, frame.header.payload_len
                ),
                "spxh:frame_index": int(frame.index),
                "spxh:start_bit": int(frame.start_bit),
                "spxh:payload_hex": bytes(frame.payload).hex(),
                "spxh:crc": int(frame.crc),
            }
        )
    return annotations


def _encode(samples: np.ndarray, datatype: str) -> np.ndarray:
    spec = SIGMF_DATATYPES[datatype]
    kind = str(spec["kind"])
    dtype = np.dtype(str(spec["numpy"]))
    if kind == "complex":
        return samples.astype(dtype)
    if kind == "complex_int":
        full_scale = float(spec["full_scale"])
        bound = 1.0 - 1.0 / full_scale
        real = np.round(np.clip(samples.real, -1.0, bound) * full_scale).astype(np.int64)
        imag = np.round(np.clip(samples.imag, -1.0, bound) * full_scale).astype(np.int64)
        out = np.empty(real.size * 2, dtype=dtype)
        out[0::2] = real
        out[1::2] = imag
        return out
    if kind == "real_int":
        full_scale = float(spec["full_scale"])
        clipped = np.clip(samples.real, -1.0, 1.0 - 1.0 / full_scale)
        return np.round(clipped * full_scale).astype(dtype)
    return samples.real.astype(dtype)


def write_sigmf(
    prefix,
    signal: Signal,
    datatype: str = "cf32_le",
    annotations: Optional[list[dict[str, Any]]] = None,
    description: str = "",
    author: str = "SPXH",
    truth: Optional[GroundTruth] = None,
    extra_global: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    """写出 .sigmf-meta / .sigmf-data，返回路径字典."""
    if datatype not in SIGMF_DATATYPES:
        raise ValueError("unsupported SigMF datatype: " + str(datatype) + " (supported: " + ", ".join(sorted(SIGMF_DATATYPES)) + ")")
    meta_path, data_path = sigmf_paths(prefix)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _encode(signal.samples, datatype)
    payload.tofile(str(data_path))

    if annotations is None and truth is not None:
        annotations = make_annotations(truth)

    global_section: dict[str, Any] = {
        "core:datatype": datatype,
        "core:sample_rate": float(signal.sample_rate),
        "core:version": SIGMF_VERSION,
        "core:description": description or "SPXH synthetic signal",
        "core:author": author,
        "core:hw": "SPXH synth",
        "core:license": "MIT",
        "core:num_channels": 1,
    }
    if extra_global:
        global_section.update(jsonable(extra_global))

    captures = [{"core:sample_start": 0, "core:frequency": float(signal.center_freq or 0.0)}]

    meta: dict[str, Any] = {
        "global": global_section,
        "captures": captures,
        "annotations": jsonable(annotations or []),
    }
    if truth is not None:
        meta["spxh:ground_truth"] = jsonable(truth.summary())

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"meta": str(meta_path), "data": str(data_path)}


def read_sigmf(meta_path, sample_rate: Optional[float] = None, center_freq: Optional[float] = None) -> Signal:
    """读取 .sigmf-meta（自动定位同目录的 .sigmf-data）."""
    meta_path = Path(str(meta_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    global_section = meta.get("global", {})
    datatype = str(global_section.get("core:datatype", "cf32_le"))
    if datatype not in SIGMF_DATATYPES:
        raise ValueError("unsupported SigMF datatype in file: " + datatype)
    spec = SIGMF_DATATYPES[datatype]

    dataset = global_section.get("core:dataset")
    if dataset:
        data_path = Path(str(dataset))
        if not data_path.is_absolute():
            data_path = meta_path.parent / data_path
    else:
        data_path = Path(str(sigmf_paths(meta_path)[1]))
    if not data_path.exists():
        raise FileNotFoundError("SigMF data file not found: " + str(data_path))

    raw = np.fromfile(str(data_path), dtype=np.dtype(str(spec["numpy"])))
    kind = str(spec["kind"])
    if kind == "complex":
        samples = raw.astype(np.complex128)
    elif kind == "complex_int":
        full_scale = float(spec["full_scale"])
        pairs = raw.reshape(-1, 2).astype(np.float64)
        samples = (pairs[:, 0] + 1j * pairs[:, 1]) / full_scale
    elif kind == "real_int":
        full_scale = float(spec["full_scale"])
        samples = (raw.astype(np.float64) / full_scale).astype(np.complex128)
    else:
        samples = raw.astype(np.float64).astype(np.complex128)

    fs = sample_rate if sample_rate is not None else global_section.get("core:sample_rate")
    if fs is None:
        raise ValueError("SigMF meta has no core:sample_rate; pass sample_rate explicitly")

    captures = meta.get("captures") or []
    fc = center_freq
    if fc is None and captures:
        fc = captures[0].get("core:frequency")

    return Signal(
        samples=samples,
        sample_rate=float(fs),
        center_freq=float(fc) if fc is not None else None,
        meta={
            "source": str(meta_path),
            "datatype": datatype,
            "is_real": kind in ("real", "real_int"),
            "sigmf_global": jsonable(global_section),
            "sigmf_annotations": jsonable(meta.get("annotations", [])),
        },
    )
