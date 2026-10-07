"""WAV 读写.

约定
----
mode="iq"    双声道 WAV，左声道 I、右声道 Q（保留完整复基带，推荐）
mode="real"  单声道 WAV，只保留实部（有损；用于与只支持实采样的工具交换）

读入时：双声道 -> I/Q 复信号；单声道 -> 默认做 Hilbert 变换取解析信号
（analytic=True），从而得到复基带。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert

from spxh.core.types import Signal

__all__ = ["read_wav", "write_wav"]


def read_wav(path, analytic: bool = True, sample_rate: Optional[float] = None) -> Signal:
    fs, data = wavfile.read(str(path))
    arr = np.asarray(data)
    if arr.dtype.kind in ("i", "u"):
        info = np.iinfo(arr.dtype)
        scale = float(max(abs(info.min), info.max))
        arr = arr.astype(np.float64) / scale
    else:
        arr = arr.astype(np.float64)

    if arr.ndim == 2:
        if arr.shape[1] < 2:
            raise ValueError("WAV with 2-D data must have at least 2 channels (I/Q)")
        samples = arr[:, 0] + 1j * arr[:, 1]
        is_real = False
    else:
        if analytic:
            samples = hilbert(arr).astype(np.complex128)
            is_real = False
        else:
            samples = arr.astype(np.complex128)
            is_real = True

    return Signal(
        samples=samples,
        sample_rate=float(sample_rate) if sample_rate is not None else float(fs),
        center_freq=None,
        meta={"source": str(path), "datatype": "wav", "is_real": is_real, "wav_channels": int(arr.shape[1]) if arr.ndim == 2 else 1},
    )


def write_wav(path, signal: Signal, mode: str = "iq") -> str:
    target = Path(str(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "iq":
        data = np.stack([signal.samples.real, signal.samples.imag], axis=1).astype(np.float32)
    elif mode == "real":
        data = signal.samples.real.astype(np.float32)
    else:
        raise ValueError("mode must be 'iq' or 'real', got " + str(mode))
    wavfile.write(str(target), int(round(signal.sample_rate)), data)
    return str(target)
