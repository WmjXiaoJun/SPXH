"""交织与解交织.

* 矩阵块交织：按行写入、按列读出（把突发错误在时间上打散）。
* 卷积交织：I 个支路、相邻支路延迟差 step（工程上更常用，延迟小、实现简单）。

两种都提供**精确可逆**的逆运算：卷积交织这里不做"移位寄存器推理"，而是直接算出
下标映射 perm（out[i] = x[perm[i]]，perm[i] = -1 表示填充零），解交织把值放回原位。
这样不会出现延迟/符号约定写错导致的"看似能跑但错位一格"的问题。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "block_interleave",
    "block_deinterleave",
    "convolutional_interleave_map",
    "convolutional_interleave",
    "convolutional_deinterleave",
]


def _as_array(bits) -> np.ndarray:
    arr = np.asarray(bits)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def block_interleave(bits, rows: int, cols: int) -> np.ndarray:
    """矩阵块交织：按行写入、按列读出（长度必须是 rows*cols）."""
    arr = _as_array(bits)
    rows = int(rows)
    cols = int(cols)
    if arr.size != rows * cols:
        raise ValueError("长度 {} 与交织矩阵 {}x{} 不匹配".format(arr.size, rows, cols))
    return arr.reshape(rows, cols).T.reshape(-1).copy()


def block_deinterleave(bits, rows: int, cols: int) -> np.ndarray:
    """矩阵块交织的逆：按列写入、按行读出."""
    arr = _as_array(bits)
    rows = int(rows)
    cols = int(cols)
    if arr.size != rows * cols:
        raise ValueError("长度 {} 与交织矩阵 {}x{} 不匹配".format(arr.size, rows, cols))
    return arr.reshape(cols, rows).T.reshape(-1).copy()


def convolutional_interleave_map(n: int, depth: int, step: int = 1) -> np.ndarray:
    """卷积交织的下标映射（含冲刷段）.

    经典卷积交织是**流式**的：I 个支路，第 k 个支路延迟 k*step 个符号。
    收尾时要把还留在支路里的符号冲出来，因此输出长度 =
    n + step*depth*(depth-1)/2（例如 depth=5, step=1 时多出 10 个符号）。
    这里不做"移位寄存器推理"，而是直接把每个输出位置对应的输入下标记录成 perm
    （perm[j] = -1 表示该位置是预热零），解交织按 perm 放回即可精确可逆 ——
    避免延迟/符号约定写错导致的"看似能跑但错位一格"。
    """
    n = int(n)
    depth = int(depth)
    step = int(step)
    if depth < 1 or step < 1:
        raise ValueError("depth/step 必须为正整数")
    perm: list[int] = []
    banks: list[list[int]] = [[] for _ in range(depth)]
    for i in range(n):
        bank = i % depth
        delay = bank * step
        if delay == 0:
            perm.append(i)
        else:
            banks[bank].append(i)
            if len(banks[bank]) > delay:
                perm.append(banks[bank].pop(0))
            else:
                perm.append(-1)
    # 冲刷：轮流从各支路取出剩余符号，直到取空
    while any(banks):
        for bank in range(depth):
            if banks[bank]:
                perm.append(banks[bank].pop(0))
    return np.asarray(perm, dtype=np.int64)


def convolutional_interleave(bits, depth: int, step: int = 1) -> np.ndarray:
    arr = _as_array(bits)
    perm = convolutional_interleave_map(arr.size, depth, step)
    out = np.zeros(perm.size, dtype=arr.dtype)
    valid = perm >= 0
    out[valid] = arr[perm[valid]]
    return out


def convolutional_deinterleave(bits, depth: int, step: int = 1, length: int | None = None) -> np.ndarray:
    arr = _as_array(bits)
    if length is None:
        raise ValueError("解交织必须给出原始长度 length（卷积交织输出比输入长 (depth-1)*step）")
    perm = convolutional_interleave_map(int(length), depth, step)
    if arr.size != perm.size:
        raise ValueError("输入长度 {} 与期望 {} 不符".format(arr.size, perm.size))
    out = np.zeros(int(length), dtype=arr.dtype)
    valid = perm >= 0
    out[perm[valid]] = arr[valid]
    return out
