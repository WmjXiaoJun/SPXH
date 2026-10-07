"""卷积码与 Viterbi 译码（K=7、码率 1/2、生成多项式 133/171 八进制，NASA/CCSDS 标准）.

* 硬判决：分支度量为 Hamming 距离（输入是解调后的 0/1）。
* 软判决：分支度量为 LLR 相关（输入是 M3 输出的 LLR，正数表示比特 0 更可能）。
  软判决把"有多可靠"也带进网格搜索，典型增益约 2 dB —— 这也是 M3 要认真做 LLR 标定的原因。

实现说明：状态定义为最近 K-1 个输入比特；编码器是标准的前馈卷积编码器
（抽头由八进制生成多项式给出），结尾补 K-1 个零比特（terminated）以保证网格回到零状态。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["ConvolutionalCode", "viterbi_decode_hard", "viterbi_decode_soft"]


@dataclass(frozen=True)
class ConvolutionalCode:
    constraint: int = 7
    polys: tuple[int, ...] = (0o133, 0o171)

    def __post_init__(self) -> None:
        if self.constraint < 3:
            raise ValueError("约束长度至少为 3")
        if len(self.polys) < 1:
            raise ValueError("至少需要一个生成多项式")
        for poly in self.polys:
            if poly <= 0 or poly >= (1 << self.constraint):
                raise ValueError("生成多项式超出约束长度范围: {:o}".format(poly))

    @property
    def num_states(self) -> int:
        return 1 << (self.constraint - 1)

    @property
    def rate(self) -> float:
        return 1.0 / float(len(self.polys))

    def encode(self, bits: Sequence[int], terminate: bool = True) -> np.ndarray:
        """编码；terminate=True 时在末尾补 K-1 个零比特，使网格回到零状态."""
        arr = np.asarray(bits, dtype=np.uint8).reshape(-1)
        if np.any((arr != 0) & (arr != 1)):
            raise ValueError("输入必须是 0/1 比特")
        tail = np.zeros(self.constraint - 1, dtype=np.uint8) if terminate else np.zeros(0, dtype=np.uint8)
        data = np.concatenate([arr, tail]) if tail.size else arr
        out = np.empty(data.size * len(self.polys), dtype=np.uint8)
        register = 0
        mask = (1 << self.constraint) - 1
        shift = self.constraint - 1
        for index, bit in enumerate(data):
            register = ((register << 1) | int(bit)) & mask
            for j, poly in enumerate(self.polys):
                out[index * len(self.polys) + j] = bin(register & poly).count("1") & 1
        return out

    # ------------------------------------------------------------- 译码
    def _branch_table(self) -> tuple[np.ndarray, np.ndarray]:
        """预计算 (next_state, outputs) 两张表，形状均为 (num_states, 2)."""
        states = self.num_states
        shift = self.constraint - 1
        next_state = np.zeros((states, 2), dtype=np.int64)
        outputs = np.zeros((states, 2, len(self.polys)), dtype=np.uint8)
        for state in range(states):
            for bit in (0, 1):
                register = (state << 1) | bit
                next_state[state, bit] = register & (states - 1)
                for j, poly in enumerate(self.polys):
                    outputs[state, bit, j] = bin(register & poly).count("1") & 1
        return next_state, outputs

    def decode_hard(self, bits: Sequence[int], terminate: bool = True) -> np.ndarray:
        """硬判决 Viterbi（Hamming 距离度量）."""
        arr = np.asarray(bits, dtype=np.uint8).reshape(-1)
        n = len(self.polys)
        if arr.size % n != 0:
            raise ValueError("接收比特数必须是每符号输出比特数的整数倍")
        received = arr.reshape(-1, n)
        return self._decode(received.astype(np.float64), soft=False, terminate=terminate)

    def decode_soft(self, llr: Sequence[float], terminate: bool = True) -> np.ndarray:
        """软判决 Viterbi（LLR 相关度量）；LLR 约定：正数表示比特 0 更可能."""
        arr = np.asarray(llr, dtype=np.float64).reshape(-1)
        n = len(self.polys)
        if arr.size % n != 0:
            raise ValueError("LLR 个数必须是每符号输出比特数的整数倍")
        received = arr.reshape(-1, n)
        return self._decode(received, soft=True, terminate=terminate)

    def _decode(self, received: np.ndarray, soft: bool, terminate: bool) -> np.ndarray:
        next_state, outputs = self._branch_table()
        steps = received.shape[0]
        states = self.num_states
        neg_inf = -1e18
        metric = np.full(states, neg_inf, dtype=np.float64)
        metric[0] = 0.0
        predecessors = np.zeros((steps, states), dtype=np.int64)
        decisions = np.zeros((steps, states), dtype=np.uint8)
        candidates_by_bit: list[tuple[np.ndarray, np.ndarray]] = []

        for step in range(steps):
            symbol = received[step]
            new_metric = np.full(states, neg_inf, dtype=np.float64)
            candidates_by_bit.clear()
            for bit in (0, 1):
                # 每条分支：(前驱状态) -> (目标状态)，输出比特由生成多项式决定
                branch_outputs = outputs[:, bit, :]                # (states, n)
                if soft:
                    # 最大化 sum_j (1-2*out_j) * llr_j：out=0 且 llr>0 时贡献 +llr
                    score = np.sum((1.0 - 2.0 * branch_outputs) * symbol[None, :], axis=1)
                else:
                    score = -np.sum(np.abs(branch_outputs.astype(np.float64) - symbol[None, :]), axis=1)
                candidate_metric = metric + score
                target = next_state[:, bit]
                # 关键：一个目标状态有两条前驱分支，必须用 maximum 归并。
                # 直接 new[target] = candidate 会让"后写的那条"覆盖更好的那条
                # （fancy indexing 重复下标时最后一次赋值生效），实测会导致
                # 无噪声译码都译错。
                np.maximum.at(new_metric, target, candidate_metric)
                candidates_by_bit.append((target, candidate_metric, bit))
            # 回填回溯指针：谁达到了该目标状态的最优度量，就记谁
            for target, candidate_metric, bit in candidates_by_bit:
                wins = candidate_metric >= new_metric[target] - 1e-9
                if np.any(wins):
                    rows = np.flatnonzero(wins)
                    predecessors[step, target[rows]] = rows
                    decisions[step, target[rows]] = bit
            metric = new_metric

        # 回溯：terminated 时终点固定为状态 0；否则取度量最大的状态
        if terminate:
            state = 0
        else:
            state = int(np.argmax(metric))
        decoded = np.zeros(steps, dtype=np.uint8)
        for step in range(steps - 1, -1, -1):
            decoded[step] = decisions[step, state]
            state = int(predecessors[step, state])
        if terminate:
            decoded = decoded[: max(0, decoded.size - (self.constraint - 1))]
        return decoded


_DEFAULT = ConvolutionalCode()


def viterbi_decode_hard(bits: Sequence[int], code: ConvolutionalCode | None = None, terminate: bool = True) -> np.ndarray:
    return (code or _DEFAULT).decode_hard(bits, terminate=terminate)


def viterbi_decode_soft(llr: Sequence[float], code: ConvolutionalCode | None = None, terminate: bool = True) -> np.ndarray:
    return (code or _DEFAULT).decode_soft(llr, terminate=terminate)
