"""原生 I/Q 二进制流读写.

支持的数据类型
--------------
cf32 / cf64  复数浮点（I,Q 交错）
ci16 / ci8 / ci32  复数整型（I,Q 交错，按满量程归一化到 [-1, 1)）
rf32 / rf64 / ri16 实信号（返回时虚部为 0，meta.is_real = True）

约定：文件名后缀可用于推断类型（.cf32/.cs16/.c16/.ci16/...）；
采样率**无法**从裸数据推断，必须显式提供（SigMF/WAV 中才有元数据）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from spxh.core.types import Signal

__all__ = ["RAW_DTYPES", "SUFFIX_DATATYPE", "infer_datatype", "read_raw", "write_raw"]

RAW_DTYPES: dict[str, dict[str, object]] = {
    "cf32": {"dtype": "<c8", "kind": "complex", "itemsize": 8},
    "cf64": {"dtype": "<c16", "kind": "complex", "itemsize": 16},
    "ci32": {"dtype": "<i4", "kind": "complex_int", "itemsize": 8, "full_scale": 2 ** 31},
    "ci16": {"dtype": "<i2", "kind": "complex_int", "itemsize": 4, "full_scale": 2 ** 15},
    "ci8": {"dtype": "i1", "kind": "complex_int", "itemsize": 2, "full_scale": 2 ** 7},
    "rf32": {"dtype": "<f4", "kind": "real", "itemsize": 4},
    "rf64": {"dtype": "<f8", "kind": "real", "itemsize": 8},
    "ri16": {"dtype": "<i2", "kind": "real_int", "itemsize": 2, "full_scale": 2 ** 15},
}

SUFFIX_DATATYPE: dict[str, str] = {
    ".cf32": "cf32", ".fc32": "cf32", ".c32": "cf32", ".complex32": "cf32",
    ".cf64": "cf64", ".fc64": "cf64", ".c64": "cf64",
    ".ci16": "ci16", ".cs16": "ci16", ".c16": "ci16", ".sc16": "ci16",
    ".ci8": "ci8", ".cs8": "ci8", ".c8": "ci8", ".sc8": "ci8",
    ".ci32": "ci32", ".cs32": "ci32",
    ".rf32": "rf32", ".f32": "rf32",
    ".rf64": "rf64", ".f64": "rf64",
    ".ri16": "ri16", ".s16": "ri16",
}


def infer_datatype(path) -> Optional[str]:
    """按后缀推断数据类型，未知返回 None."""
    suffix = Path(str(path)).suffix.lower()
    return SUFFIX_DATATYPE.get(suffix)


def _resolve(datatype: Optional[str], path) -> str:
    resolved = datatype or infer_datatype(path)
    if resolved is None:
        raise ValueError(
            "cannot infer datatype from '" + str(path) + "'; pass datatype explicitly ("
            + ", ".join(sorted(RAW_DTYPES)) + ")"
        )
    resolved = str(resolved).lower()
    if resolved not in RAW_DTYPES:
        raise ValueError("unknown datatype: " + str(datatype) + " (supported: " + ", ".join(sorted(RAW_DTYPES)) + ")")
    return resolved


def read_raw(
    path,
    datatype: Optional[str] = None,
    sample_rate: Optional[float] = None,
    offset: int = 0,
    count: Optional[int] = None,
    center_freq: Optional[float] = None,
) -> Signal:
    """读取裸 I/Q 文件.

    offset/count 以**复数样本**为单位（count=None 表示读到结尾）。
    """
    kind_info = RAW_DTYPES[_resolve(datatype, path)]
    kind = str(kind_info["kind"])
    dtype = np.dtype(str(kind_info["dtype"]))

    if kind.startswith("complex") and kind != "complex":
        raw = np.fromfile(str(path), dtype=dtype)
        complex_view = raw.reshape(-1, 2).astype(np.float64)
        samples = (complex_view[:, 0] + 1j * complex_view[:, 1]) / float(kind_info["full_scale"])
    elif kind == "complex":
        samples = np.fromfile(str(path), dtype=dtype).astype(np.complex128)
    elif kind == "real_int":
        samples = np.fromfile(str(path), dtype=dtype).astype(np.float64) / float(kind_info["full_scale"])
        samples = samples.astype(np.complex128)
    else:
        samples = np.fromfile(str(path), dtype=dtype).astype(np.complex128)

    if offset:
        samples = samples[offset:]
    if count is not None:
        samples = samples[: int(count)]
    if samples.size == 0:
        raise ValueError("no samples read from " + str(path))

    if sample_rate is None:
        raise ValueError(
            "raw I/Q has no sample-rate metadata; pass sample_rate explicitly "
            "(blind sample-rate identification from samples alone is not reliable)"
        )
    is_real = kind in ("real", "real_int")
    return Signal(
        samples=samples,
        sample_rate=float(sample_rate),
        center_freq=center_freq,
        meta={"source": str(path), "datatype": _resolve(datatype, path), "is_real": is_real},
    )


def write_raw(path, signal: Signal, datatype: str = "cf32") -> str:
    """写出裸 I/Q 文件，返回实际写入路径."""
    resolved = _resolve(datatype, path)
    kind_info = RAW_DTYPES[resolved]
    kind = str(kind_info["kind"])
    dtype = np.dtype(str(kind_info["dtype"]))
    target = Path(str(path))

    if kind == "complex":
        data = signal.samples.astype(dtype)
    elif kind == "complex_int":
        full_scale = float(kind_info["full_scale"])
        bound = 1.0 - 1.0 / full_scale
        real = np.round(np.clip(signal.samples.real, -1.0, bound) * full_scale).astype(np.int64)
        imag = np.round(np.clip(signal.samples.imag, -1.0, bound) * full_scale).astype(np.int64)
        interleaved = np.empty(real.size * 2, dtype=dtype)
        interleaved[0::2] = real
        interleaved[1::2] = imag
        data = interleaved
    elif kind == "real_int":
        full_scale = float(kind_info["full_scale"])
        clipped = np.clip(signal.samples.real, -1.0, 1.0 - 1.0 / full_scale)
        data = np.round(clipped * full_scale).astype(dtype)
    else:
        data = signal.samples.real.astype(dtype)

    target.parent.mkdir(parents=True, exist_ok=True)
    data.tofile(str(target))
    return str(target)
