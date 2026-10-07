"""Reed-Solomon 编译码（GF(2^8)，标准 CCSDS/DVB 参数：RS(255,223)、fcr=0、本原多项式 0x11D）.

* 编码：系统码。信息多项式乘 x^nsym 后对生成多项式取余，余式就是校验符号。
* 译码：伴随式 -> Berlekamp-Massey 求错误定位多项式 -> Chien 搜索找错误位置 ->
  Forney 公式求错误幅值 -> 纠正后再验证伴随式是否全零。
* 纠错能力：t = (n-k)/2 个**符号**错误；超出能力时不猜测，返回 uncorrectable=True
  （用"纠正后伴随式必须全零"作为硬判据，避免静默错纠）。

约定：多项式用**低位在前**（poly[i] 是 x^i 的系数），与 BM/Chien/Forney 的公式一致。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

__all__ = ["GF256", "ReedSolomon", "rs_correctable_errors"]


class GF256:
    """GF(2^8) 运算（exp/log 查表）."""

    def __init__(self, primitive: int = 0x11D, generator: int = 2) -> None:
        self.primitive = int(primitive)
        self.generator = int(generator)
        self.exp = np.zeros(512, dtype=np.int64)
        self.log = np.zeros(256, dtype=np.int64)
        value = 1
        for i in range(255):
            self.exp[i] = value
            self.log[value] = i
            value <<= 1
            if value & 0x100:
                value ^= self.primitive
        for i in range(255, 512):
            self.exp[i] = self.exp[i - 255]
        self.log[0] = 0  # 未定义，调用方保证不会用到

    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return int(self.exp[self.log[a] + self.log[b]])

    def div(self, a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError("GF(256) 除以零")
        if a == 0:
            return 0
        return int(self.exp[(self.log[a] - self.log[b]) % 255])

    def pow(self, a: int, e: int) -> int:
        if a == 0:
            return 0
        return int(self.exp[(self.log[a] * e) % 255])

    def inverse(self, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("GF(256) 零元没有逆元")
        return int(self.exp[(255 - self.log[a]) % 255])


def _poly_mul(p: list[int], q: list[int], gf: GF256) -> list[int]:
    out = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a == 0:
            continue
        for j, b in enumerate(q):
            if b:
                out[i + j] ^= gf.mul(a, b)
    return out


def _poly_eval(p: list[int], x: int, gf: GF256) -> int:
    """Horner 求值（低位在前）."""
    y = 0
    for coefficient in reversed(p):
        y = gf.mul(y, x) ^ coefficient
    return y


def _poly_add(p: list[int], q: list[int]) -> list[int]:
    size = max(len(p), len(q))
    out = [0] * size
    for i, value in enumerate(p):
        out[i] ^= value
    for i, value in enumerate(q):
        out[i] ^= value
    return out


def _poly_divmod(dividend: list[int], divisor: list[int], gf: GF256) -> tuple[list[int], list[int]]:
    """低位在前的多项式除法，返回 (商, 余)."""
    out = list(dividend)
    divisor = list(divisor)
    while len(divisor) > 1 and divisor[-1] == 0:
        divisor.pop()
    if len(divisor) == 1 and divisor[0] == 0:
        raise ZeroDivisionError("除数为零多项式")
    if len(out) < len(divisor):
        return [0], list(out)
    quotient = [0] * (len(out) - len(divisor) + 1)
    inv_lead = gf.inverse(divisor[-1])
    for i in range(len(quotient) - 1, -1, -1):
        coefficient = gf.mul(out[i + len(divisor) - 1], inv_lead)
        quotient[i] = coefficient
        if coefficient:
            for j in range(len(divisor)):
                out[i + j] ^= gf.mul(coefficient, divisor[j])
    remainder = out[: len(divisor) - 1]
    return quotient, remainder


def rs_correctable_errors(n: int, k: int) -> int:
    return (int(n) - int(k)) // 2


@dataclass
class ReedSolomon:
    """RS(n, k) over GF(2^8)。n 最大 255；可用缩短码（n < 255）。"""

    n: int = 255
    k: int = 223
    fcr: int = 0
    primitive: int = 0x11D

    def __post_init__(self) -> None:
        if not (0 < self.k < self.n <= 255):
            raise ValueError("要求 0 < k < n <= 255")
        self.gf = GF256(self.primitive)
        self.nsym = self.n - self.k
        self.t = self.nsym // 2
        self.generator_poly = self._build_generator()

    def _build_generator(self) -> list[int]:
        gen = [1]
        for i in range(self.nsym):
            root = self.gf.pow(self.gf.generator, self.fcr + i)
            gen = _poly_mul(gen, [root, 1], self.gf)  # (x - root)，特征 2 下减法即加法
        return gen

    # --------------------------------------------------------------- 编码
    def encode(self, message: Iterable[int]) -> np.ndarray:
        """系统编码：输出 消息字节 + 校验字节，第 0 个字节对应多项式的最高次项."""
        data = [int(b) & 0xFF for b in message]
        if len(data) != self.k:
            raise ValueError("信息长度必须是 {} 字节，当前 {}".format(self.k, len(data)))
        # 多项式系数低位在前：字节序 index 0 = 最高次 -> 取反。
        # 乘 x^nsym 在低位在前的表示里是**在高位补零**（把信息项整体抬高 nsym 次），
        # 所以是前插零而不是后补零 —— 写成后补零会让余式完全错（实测伴随式非零）。
        coefficients = [0] * self.nsym + list(reversed(data))
        _, remainder = _poly_divmod(coefficients, self.generator_poly, self.gf)
        remainder = remainder + [0] * (self.nsym - len(remainder))
        parity = list(reversed(remainder))                      # 转回字节序（高次在前）
        return np.asarray(data + parity, dtype=np.uint8)

    # --------------------------------------------------------------- 译码
    def syndromes(self, codeword: Iterable[int]) -> list[int]:
        word = [int(b) & 0xFF for b in codeword]
        coefficients = list(reversed(word))                     # 低位在前
        return [
            _poly_eval(coefficients, self.gf.pow(self.gf.generator, self.fcr + i), self.gf)
            for i in range(self.nsym)
        ]

    def decode(self, received: Iterable[int], erasures: Iterable[int] | None = None) -> tuple[np.ndarray, int, bool]:
        """返回 (译码后的 n 字节码字, 纠正的符号数, uncorrectable).

        erasures 给出**已知出错位置**（例如"这段突发里所有符号都不可信"）时走擦除译码：
        位置已知 -> 只需要解幅值，能力从 t 个符号翻倍到 2t = nsym 个符号。
        这是"帧同步/置信度能指出可疑位置"时的标准做法。
        """
        word = [int(b) & 0xFF for b in received]
        if len(word) != self.n:
            raise ValueError("码字长度必须是 {}，当前 {}".format(self.n, len(word)))
        if erasures is not None:
            return self._decode_with_erasures(word, erasures)
        syndrome = self.syndromes(word)
        if not any(syndrome):
            return np.asarray(word, dtype=np.uint8), 0, False

        locator = self._berlekamp_massey(syndrome)
        degree = len(locator) - 1
        if degree == 0 or degree > self.t:
            return np.asarray(word, dtype=np.uint8), 0, True

        positions = self._chien_search(locator)
        if len(positions) != degree:
            return np.asarray(word, dtype=np.uint8), 0, True

        magnitudes = self._forney(syndrome, locator, positions)
        corrected = list(word)
        for position, magnitude in zip(positions, magnitudes):
            corrected[position] ^= magnitude
        if any(self.syndromes(corrected)):
            # 纠正后伴随式不为零 -> 明确报告"不可纠"，绝不返回可疑结果
            return np.asarray(word, dtype=np.uint8), 0, True
        return np.asarray(corrected, dtype=np.uint8), len(positions), False

    # ------------------------------------------------------- 擦除译码
    def _locator_for(self, position: int) -> int:
        """位置 p（字节序下标）对应的错误定位数 X = alpha^(n-1-p)."""
        exponent = (self.n - 1 - int(position)) % 255
        return self.gf.pow(self.gf.generator, exponent)

    def _solve_erasures(self, syndrome: list[int], positions: list[int]) -> list[int] | None:
        """已知位置求幅值：解范德蒙德线性方程组 S_i = sum_j e_j * X_j^i (i=0..e-1)."""
        gf = self.gf
        count = len(positions)
        locators = [self._locator_for(p) for p in positions]
        # 构造增广矩阵
        matrix = [[0] * (count + 1) for _ in range(count)]
        for i in range(count):
            for j in range(count):
                matrix[i][j] = gf.pow(locators[j], i)
            matrix[i][count] = syndrome[i]
        # 高斯消元（列主元）
        for col in range(count):
            pivot = None
            for row in range(col, count):
                if matrix[row][col] != 0:
                    pivot = row
                    break
            if pivot is None:
                return None
            matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
            inv = gf.inverse(matrix[col][col])
            for j in range(col, count + 1):
                matrix[col][j] = gf.mul(matrix[col][j], inv)
            for row in range(count):
                if row == col or matrix[row][col] == 0:
                    continue
                factor = matrix[row][col]
                for j in range(col, count + 1):
                    matrix[row][j] ^= gf.mul(factor, matrix[col][j])
        return [matrix[i][count] for i in range(count)]

    def _correct_pure_erasures(self, word: list[int], positions: list[int]) -> tuple[np.ndarray, int, bool]:
        """纯擦除（位置全部已知、无未知错误）：解范德蒙德线性方程组求幅值."""
        syndrome = self.syndromes(word)
        if not any(syndrome):
            return np.asarray(word, dtype=np.uint8), 0, False
        magnitudes = self._solve_erasures(syndrome, positions)
        if magnitudes is None:
            return np.asarray(word, dtype=np.uint8), 0, True
        corrected = list(word)
        for position, magnitude in zip(positions, magnitudes):
            corrected[position] ^= magnitude
        if any(self.syndromes(corrected)):
            return np.asarray(word, dtype=np.uint8), 0, True
        return np.asarray(corrected, dtype=np.uint8), len(positions), False

    def _decode_with_erasures(self, word: list[int], erasures: Iterable[int]) -> tuple[np.ndarray, int, bool]:
        """误差 + 擦除**联合**译码（Forney 伴随式 + 修正 BM）.

        能力：2 x 未知错误数 + 擦除数 <= nsym = 2t。例如 t=8 时可以
        (16 擦除) 或 (8 错误) 或 (4 擦除 + 6 错误) —— 这样"擦除标得不全、
        还剩几个未知错误"的场景也能救回来，而不是像纯擦除译码那样直接放弃。
        """
        positions = sorted({int(p) for p in erasures})
        if not positions:
            return self.decode(word)
        if len(positions) > self.nsym or any(p < 0 or p >= self.n for p in positions):
            return np.asarray(word, dtype=np.uint8), 0, True
        syndrome = self.syndromes(word)
        if not any(syndrome):
            return np.asarray(word, dtype=np.uint8), 0, False

        # 1) 擦除定位多项式 Gamma(x) = prod (1 - X_p x)
        # 注意系数顺序（低位在前）：擦除定位多项式是 prod (1 + X_p x)，
        # 对应的系数数组是 [1, X_p]，写成 [X_p, 1] 会变成 prod (x + X_p) ——
        # 那是个不同的多项式，Chien 搜索一个根都找不到（踩过）。
        gamma = [1]
        for position in positions:
            gamma = _poly_mul(gamma, [1, self._locator_for(position)], self.gf)
        # 2) Forney 伴随式：T(x) = [S(x)Gamma(x)] mod x^nsym，但 BM 只能喂
        #    **从 Gamma 次数开始**的那一段（i = e .. nsym-1）—— 低位那几项里擦除的
        #    贡献还没抵消干净，喂进去 BM 会解出一个虚假的"错误定位多项式"
        #    （实测 16 个擦除 + 0 个错误时 BM 给出 8 次多项式）。
        gamma_degree = len(gamma) - 1
        product = _poly_mul(syndrome, gamma, self.gf)
        modified = product[gamma_degree : self.nsym]
        if not modified:
            # 擦除数已经用满 nsym：没有剩余伴随式去解未知错误，
            # 这时退化成"纯擦除"（位置已知、只解幅值），能力正好 2t
            return self._correct_pure_erasures(word, positions)
        # 3) BM 只解剩下的未知错误
        locator_errors = self._berlekamp_massey(modified)
        locator = _poly_mul(locator_errors, gamma, self.gf)
        # 4) Chien 搜根：根的个数必须等于定位多项式的次数，否则不可纠
        bad_positions = self._chien_search(locator)
        if len(bad_positions) != len(locator) - 1:
            return np.asarray(word, dtype=np.uint8), 0, True
        if len(bad_positions) > self.nsym:
            return np.asarray(word, dtype=np.uint8), 0, True
        # 5) Forney 求幅值并纠正，最后用"伴随式必须全零"做硬判据
        magnitudes = self._forney(syndrome, locator, bad_positions)
        corrected = list(word)
        for position, magnitude in zip(bad_positions, magnitudes):
            corrected[position] ^= magnitude
        if any(self.syndromes(corrected)):
            return np.asarray(word, dtype=np.uint8), 0, True
        return np.asarray(corrected, dtype=np.uint8), len(bad_positions), False

    def _berlekamp_massey(self, syndrome: list[int]) -> list[int]:
        gf = self.gf
        locator = [1]
        previous = [1]
        shift = 1
        scale = 1
        for step in range(len(syndrome)):
            discrepancy = syndrome[step]
            for i in range(1, len(locator)):
                if locator[i] and step - i >= 0:
                    discrepancy ^= gf.mul(locator[i], syndrome[step - i])
            if discrepancy == 0:
                shift += 1
                continue
            factor = gf.div(discrepancy, scale)
            candidate = _poly_add(locator, [0] * shift + [gf.mul(factor, c) for c in previous])
            if 2 * (len(locator) - 1) <= step:
                old = list(locator)
                locator = candidate
                previous = old
                scale = discrepancy
                shift = 1
            else:
                locator = candidate
                shift += 1
        return locator

    def _chien_search(self, locator: list[int]) -> list[int]:
        """找错误位置：错误在码字下标 p 处 <=> locator(alpha^-(n-1-p)) = 0."""
        gf = self.gf
        positions: list[int] = []
        for p in range(self.n):
            exponent = (self.n - 1 - p) % 255
            x_inverse = gf.pow(gf.generator, (-exponent) % 255)
            if _poly_eval(locator, x_inverse, gf) == 0:
                positions.append(p)
        return positions

    def _forney(self, syndrome: list[int], locator: list[int], positions: list[int]) -> list[int]:
        gf = self.gf
        omega = _poly_mul(syndrome, locator, gf)[: self.nsym]
        derivative = [0] * max(1, len(locator) - 1)
        for i in range(1, len(locator)):
            if i % 2 == 1:
                derivative[i - 1] = locator[i]
        magnitudes: list[int] = []
        for p in positions:
            exponent = (self.n - 1 - p) % 255
            x = gf.pow(gf.generator, exponent)
            x_inverse = gf.inverse(x)
            numerator = _poly_eval(omega, x_inverse, gf)
            denominator = _poly_eval(derivative, x_inverse, gf)
            if denominator == 0:
                magnitudes.append(0)
                continue
            magnitude = gf.div(numerator, denominator)
            magnitude = gf.mul(magnitude, gf.pow(x, 1 - self.fcr))
            magnitudes.append(magnitude)
        return magnitudes
