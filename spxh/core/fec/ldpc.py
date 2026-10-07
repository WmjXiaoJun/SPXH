"""LDPC 编译码：Gallager 正则构造 + 对数域归一化最小和（NMS）迭代译码.

* 构造：dv、dc 正则。H = [H1; pi1(H1); ...; pi_{dv-1}(H1)]，H1 是 (M/dv) x n 的子矩阵，
  每行 dc 个连续 1、每列恰好 1 个 1，于是整块 H 的行重为 dc、列重为 dv。
* **秩**：Gallager 构造有一个已知的秩亏：每个子块的行异或都等于全 1 向量，因此
  dv 个子块之间存在 dv-1 个线性相关（实测 1152/3/6 时秩 = 574 = M-2）。这里不做假装，
  而是显式做秩判定、丢掉相关行，并把真实维度 k = n - rank 报出来（此时 k = 578，
  码率 0.5017，而不是硬凑的 0.5）。
* 编码：系统码。对 H 做高斯消元选主元列（不要求是最后 M 列），得到
  x_c = sum_{info j} H'[r][j] x_j，用 Python 大整数按位运算实现，编码是 O(M) 次整数运算。
* 译码：对数域归一化最小和（alpha=0.8），flooding 调度，最多 max_iter 轮；
  收敛判据是 H*hard = 0（大整数奇偶校验，快且无歧义）。

NR BG / DVB-S2 的基图只是不同的 H，可按同一接口接入；速率匹配、缩短/打孔等标准细节
不在本阶段范围内（见 M4 报告"已知局限"）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["LDPCCode"]


@dataclass
class LDPCCode:
    n: int = 1152
    dv: int = 3
    dc: int = 6
    seed: int = 2026
    alpha: float = 0.8
    max_iter: int = 50
    m: int = field(init=False, default=0)
    k: int = field(init=False, default=0)
    _all_rows: list[int] = field(default_factory=list, repr=False)
    _h_rows: list[int] = field(default_factory=list, repr=False)
    _info_cols: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64), repr=False)
    _check_cols: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64), repr=False)
    _row_info_mask: list[int] = field(default_factory=list, repr=False)
    check_vars: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int64), repr=False)

    def __post_init__(self) -> None:
        if self.n % self.dc != 0:
            raise ValueError("n 必须是 dc 的整数倍")
        nominal_m = self.n * self.dv // self.dc
        if nominal_m * self.dc != self.n * self.dv:
            raise ValueError("dv/dc 与 n 不匹配")
        if nominal_m % self.dv != 0:
            raise ValueError("校验行数必须能被 dv 整除")
        self._build_matrix(nominal_m)
        self._build_encoder()

    # ------------------------------------------------------------ 构造
    def _build_matrix(self, nominal_m: int) -> None:
        rng = np.random.default_rng(self.seed)
        block = nominal_m // self.dv
        h1 = np.zeros((block, self.n), dtype=np.uint8)
        for r in range(block):
            h1[r, r * self.dc : (r + 1) * self.dc] = 1
        blocks = [h1]
        for _ in range(self.dv - 1):
            blocks.append(h1[:, rng.permutation(self.n)])
        h = np.vstack(blocks)
        self._all_rows = [_bits_to_int(row) for row in h]

    # ------------------------------------------------------------ 编码器
    def _build_encoder(self) -> None:
        """高斯消元定秩 -> 选主元列（= 校验列）-> 系统编码器."""
        rows = list(self._all_rows)
        origin = list(range(len(rows)))
        pivot_row_for_col: dict[int, int] = {}
        next_row = 0
        total_rows = len(rows)
        for column in sorted(range(self.n), reverse=True):
            if next_row >= total_rows:
                break
            candidate = None
            for r in range(next_row, total_rows):
                if (rows[r] >> column) & 1:
                    candidate = r
                    break
            if candidate is None:
                continue
            rows[next_row], rows[candidate] = rows[candidate], rows[next_row]
            origin[next_row], origin[candidate] = origin[candidate], origin[next_row]
            pivot_row_for_col[column] = next_row
            pivot = rows[next_row]
            for r in range(total_rows):
                if r != next_row and ((rows[r] >> column) & 1):
                    rows[r] ^= pivot
            next_row += 1

        self.m = next_row                       # 真实秩
        self.k = self.n - self.m
        independent = sorted(origin[: self.m])
        self._h_rows = [self._all_rows[i] for i in independent]
        self.check_vars = np.stack(
            [np.flatnonzero(_int_to_bits(row, self.n)) for row in self._h_rows]
        ).astype(np.int64)
        self._build_edge_map()

        info_cols = [c for c in range(self.n) if c not in pivot_row_for_col]
        if len(info_cols) != self.k:
            raise RuntimeError("信息列数不为 k")
        self._info_cols = np.asarray(info_cols, dtype=np.int64)
        row_info_mask: list[int] = []
        check_cols: list[int] = []
        for column, r in sorted(pivot_row_for_col.items(), key=lambda item: item[1]):
            row = rows[r]
            mask = 0
            for c in info_cols:
                if (row >> c) & 1:
                    mask |= 1 << c
            row_info_mask.append(mask)
            check_cols.append(column)
        self._row_info_mask = row_info_mask
        self._check_cols = np.asarray(check_cols, dtype=np.int64)

    def _build_edge_map(self) -> None:
        """建立 (校验节点, 槽位) <-> (变量节点, 槽位) 的双向映射.

        flooding 调度的变量节点更新必须"排除该校验节点自己的消息"，
        所以每条边都要有稳定的槽位索引。
        """
        m, dc = self.check_vars.shape
        lists: list[list[int]] = [[] for _ in range(self.n)]
        edge_slot = np.zeros((m, dc), dtype=np.int64)
        for i in range(m):
            for k in range(dc):
                j = int(self.check_vars[i, k])
                edge_slot[i, k] = len(lists[j])
                lists[j].append(i)
        self.max_degree = max(len(item) for item in lists)
        var_checks = np.full((self.n, self.max_degree), -1, dtype=np.int64)
        degrees = np.zeros(self.n, dtype=np.int64)
        for j, item in enumerate(lists):
            var_checks[j, : len(item)] = item
            degrees[j] = len(item)
        self.var_checks = var_checks
        self._edge_slot = edge_slot
        # 秩亏丢掉相关行后列重不再严格等于 dv，这里如实记录度分布
        self.degree_histogram = {int(d): int(np.sum(degrees == d)) for d in np.unique(degrees)}

    # ------------------------------------------------------------ 编码
    def encode(self, info_bits) -> np.ndarray:
        info = np.asarray(info_bits, dtype=np.uint8).reshape(-1)
        if info.size != self.k:
            raise ValueError("信息比特数必须是 {}，当前 {}".format(self.k, info.size))
        info_int = 0
        for bit, column in zip(info, self._info_cols):
            if bit:
                info_int |= 1 << int(column)
        codeword = info_int
        for mask, column in zip(self._row_info_mask, self._check_cols):
            if bin(info_int & mask).count("1") & 1:
                codeword |= 1 << int(column)
        return _int_to_bits(codeword, self.n)

    def parity_ok(self, bits) -> bool:
        word = np.asarray(bits, dtype=np.uint8).reshape(-1)
        if word.size != self.n:
            return False
        packed = _bits_to_int(word)
        for row in self._all_rows:
            if bin(packed & row).count("1") & 1:
                return False
        return True

    # ------------------------------------------------------------ 译码
    def decode(self, llr, max_iter: int | None = None) -> dict:
        """对数域归一化最小和译码。LLR 约定：正数表示比特 0 更可能."""
        channel = np.asarray(llr, dtype=np.float64).reshape(-1)
        if channel.size != self.n:
            raise ValueError("LLR 长度必须是 {}，当前 {}".format(self.n, channel.size))
        iterations = int(max_iter or self.max_iter)
        check_vars = self.check_vars
        m, dc = check_vars.shape
        messages = np.zeros((m, dc), dtype=np.float64)
        posterior = channel.copy()
        converged = False
        used = 0
        # 变量节点 -> 校验节点 的消息，初值就是信道 LLR。
        # 注意不能用"后验"当变量节点消息：后验里已经含了该校验节点自己的消息，
        # 会让迭代互相污染（实测各种信噪比都收敛不了）。
        variable_to_check = np.tile(channel[:, None], (1, self.max_degree))
        for step in range(iterations):
            used = step + 1
            excluded = variable_to_check[check_vars, self._edge_slot]   # (m, dc)
            signs = np.where(excluded >= 0, 1.0, -1.0)
            sign_product = np.prod(signs, axis=1, keepdims=True)
            magnitude = np.abs(excluded)
            argmin = np.argmin(magnitude, axis=1)
            rows = np.arange(m)
            first = magnitude[rows, argmin]
            temp = magnitude.copy()
            temp[rows, argmin] = np.inf
            second = temp.min(axis=1)
            minimum = np.repeat(first[:, None], dc, axis=1)
            minimum[rows, argmin] = second
            messages = self.alpha * sign_product * signs * minimum
            # 后验 = 信道 LLR + 所有相连校验节点消息之和
            posterior = channel + np.bincount(
                check_vars.reshape(-1), weights=messages.reshape(-1), minlength=self.n
            )
            # 变量节点更新：排除该校验节点自己的贡献
            variable_to_check[check_vars, self._edge_slot] = posterior[check_vars] - messages
            hard = (posterior < 0).astype(np.uint8)
            if self.parity_ok(hard):
                converged = True
                break
        hard = (posterior < 0).astype(np.uint8)
        return {
            "bits": hard,
            "info": hard[self._info_cols],
            "iterations": used,
            "converged": converged,
            "parity_ok": bool(self.parity_ok(hard)),
        }


def _bits_to_int(bits) -> int:
    value = 0
    for index, bit in enumerate(np.asarray(bits).reshape(-1)):
        if int(bit):
            value |= 1 << index
    return value


def _int_to_bits(value: int, length: int) -> np.ndarray:
    bits = np.zeros(int(length), dtype=np.uint8)
    for index in range(int(length)):
        bits[index] = (int(value) >> index) & 1
    return bits
