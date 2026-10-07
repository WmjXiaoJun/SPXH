"""统一编码接口：把 M4 的四种编译码器接到帧结构上（M6 端到端链路的中间层）.

约定的帧布局（Barker 在编码之外，帧头与 CRC 也在编码之外）：

    Barker-13 (13b, 未编码, 用于帧同步/解相位模糊)
    + 帧头 (32b, 未编码, 含 fec / interleave / payload_len -> 接收端据此译码)
    + 已编码载荷 (长度由 payload_len 与所选编码唯一决定)
    + CRC-16 (16b, 未编码, 覆盖 帧头 + **信息字节**)

之所以让帧头留在编码之外：接收端必须先知道"用哪种码、交织没交织"才能译码；
而 CRC 放在编码之外、覆盖**译完后**的信息字节，正好对应方案里的
"解调 -> 解交织 -> Viterbi/RS -> 帧同步 -> 载荷 -> CRC"这条链的最终校验点。

交织采用固定 16 列的块交织（行数 = ceil(n/16)，不足补零、接收端按长度裁掉），
参数只与长度有关，收发两端都能独立算出来，不需要额外传参。
"""
from __future__ import annotations

import numpy as np

from spxh.core.fec.ccsds import CCSDSCode, symbol_deinterleave, symbol_interleave
from spxh.core.fec.interleave import block_deinterleave, block_interleave
from spxh.core.fec.ldpc import LDPCCode
from spxh.core.fec.reed_solomon import ReedSolomon
from spxh.core.fec.viterbi import ConvolutionalCode

__all__ = [
    "FEC_MODES",
    "normalize_fec",
    "coded_length_bits",
    "encode_payload",
    "decode_payload",
    "decode_fec",
]

FEC_MODES = ("none", "conv", "rs", "ccsds", "ldpc")
_INTERLEAVE_COLS = 16
_LDPC_CACHE: dict[int, LDPCCode] = {}


def normalize_fec(fec: str) -> str:
    value = str(fec or "none").strip().lower()
    if value not in FEC_MODES:
        raise ValueError("不支持的编码方式：{}（可选 {}）".format(fec, "/".join(FEC_MODES)))
    return value


def _ldpc_code() -> LDPCCode:
    key = 1152
    if key not in _LDPC_CACHE:
        _LDPC_CACHE[key] = LDPCCode(n=1152, dv=3, dc=6, seed=2026)
    return _LDPC_CACHE[key]


def _rs_params(info_bytes: int) -> tuple[int, int]:
    """自适应缩短 RS：n = k + 16（t=8），k 就是信息字节数."""
    k = int(info_bytes)
    if k < 1:
        raise ValueError("RS 需要至少 1 字节信息")
    n = k + 16
    if n > 255:
        raise ValueError("RS 码长超限：k={} -> n={} > 255（请减小载荷或分段）".format(k, n))
    return n, k


def _ccsds_params(info_bytes: int, interleave: int = 5) -> tuple[int, int, int, int]:
    """返回 (交织深度, 每路 RS 码长 n, 每路信息字节 k, **补齐后**的总信息字节数).

    深度为 I 的级联要求信息字节数能被 I 整除；不足时在**信息尾部补零**
    （不是补在编码后），这样任意载荷长度都能用同一套参数，收发两端算法一致。
    调用方按真实长度截断即可。
    """
    depth = int(interleave)
    if depth < 1:
        raise ValueError("交织深度必须为正")
    pad = (-int(info_bytes)) % depth
    total = int(info_bytes) + pad
    k = total // depth
    n = k + 16
    if n > 255:
        raise ValueError("CCSDS 级联的 RS 码长超限：n={}（信息太长，请减小载荷或提高交织深度）".format(n))
    return depth, n, k, total


def _ldpc_info_bits(info_bytes: int) -> tuple[int, int]:
    code = _ldpc_code()
    bits = int(info_bytes) * 8
    if bits > code.k:
        raise ValueError("LDPC 单块信息上限 {} 比特（{} 字节），当前 {} 比特".format(code.k, code.k // 8, bits))
    return code.k, bits


def _inner_length_bits(info_bytes: int, fec: str) -> int:
    """编码器**自然输出**的比特数（未交织、未补零）."""
    mode = normalize_fec(fec)
    count = int(info_bytes)
    if count < 0:
        raise ValueError("payload_len 不能为负")
    if mode == "none":
        return count * 8
    if mode == "conv":
        return (count * 8 + (ConvolutionalCode().constraint - 1)) * 2
    if mode == "rs":
        n, _k = _rs_params(count)
        return n * 8
    if mode == "ccsds":
        depth, n, _k, _total = _ccsds_params(count)
        return ((depth * n) * 8 + (ConvolutionalCode().constraint - 1)) * 2
    return _ldpc_code().n


def _interleave_total(length: int, interleave: bool, unit_bits: int = 1) -> int:
    if not interleave or length <= 0:
        return int(length)
    unit, rows, _groups = _interleave_layout(length, unit_bits)
    return rows * _INTERLEAVE_COLS * unit


def coded_length_bits(info_bytes: int, fec: str, interleave: bool = False) -> int:
    """给定信息字节数，返回**已编码载荷**的比特数（不含前导/帧头/CRC）.

    注意：开了交织时这里返回的是**补零后**的长度（16 列块交织会把尾部补齐），
    收发两端都能由信息长度独立算出同一个数，因此不需要额外传参。
    """
    return _interleave_total(_inner_length_bits(info_bytes, fec), interleave, interleave_unit_bits(fec))


def interleave_unit_bits(fec: str) -> int:
    """交织的**单位**：卷积/LDPC 是比特级码（1 bit），RS/CCSDS 是符号级码（8 bit）.

    这一点很关键：对 RS 这种"按符号纠错"的码，如果把突发在**比特**上打散，
    一个字节里翻 1 个比特也算一个符号错误 —— 32 比特突发会被摊成 32 个符号错误，
    直接超过 t=8，比不交织还糟（实测 RS+比特交织 8 次全败）。
    按**字节**打散才对：32 比特 = 4 个字节错误，稳稳落在 t=8 之内。
    """
    mode = normalize_fec(fec)
    return 8 if mode in ("rs", "ccsds") else 1


def _interleave_layout(length: int, unit_bits: int = 1) -> tuple[int, int, int]:
    unit = max(1, int(unit_bits))
    groups = int(np.ceil(int(length) / float(unit))) if length else 0
    rows = int(np.ceil(groups / float(_INTERLEAVE_COLS))) if groups else 0
    return unit, rows, groups


def _apply_interleave(bits: np.ndarray, interleave: bool, unit_bits: int = 1) -> np.ndarray:
    if not interleave or bits.size == 0:
        return np.asarray(bits, dtype=np.uint8)
    unit, rows, groups = _interleave_layout(bits.size, unit_bits)
    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    padded = np.zeros(rows * _INTERLEAVE_COLS * unit, dtype=np.uint8)
    padded[: data.size] = data
    blocks = padded.reshape(rows, _INTERLEAVE_COLS, unit)
    return blocks.transpose(1, 0, 2).reshape(-1).copy()


def _undo_interleave(bits: np.ndarray, interleave: bool, length: int, unit_bits: int = 1) -> np.ndarray:
    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if not interleave or length == 0:
        return data[: int(length)]
    unit, rows, _groups = _interleave_layout(int(length), unit_bits)
    expected = rows * _INTERLEAVE_COLS * unit
    if data.size != expected:
        raise ValueError("交织长度不符：{} != {}".format(data.size, expected))
    blocks = data.reshape(_INTERLEAVE_COLS, rows, unit)
    return blocks.transpose(1, 0, 2).reshape(-1)[: int(length)]


def _apply_interleave_llr(llr: np.ndarray, interleave: bool, unit_bits: int = 1) -> np.ndarray:
    values = np.asarray(llr, dtype=np.float64).reshape(-1)
    if not interleave or values.size == 0:
        return values
    unit, rows, _groups = _interleave_layout(values.size, unit_bits)
    padded = np.zeros(rows * _INTERLEAVE_COLS * unit, dtype=np.float64)
    padded[: values.size] = values
    blocks = padded.reshape(rows, _INTERLEAVE_COLS, unit)
    return blocks.transpose(1, 0, 2).reshape(-1).copy()


def _undo_interleave_llr(llr: np.ndarray, interleave: bool, length: int, unit_bits: int = 1) -> np.ndarray:
    values = np.asarray(llr, dtype=np.float64).reshape(-1)
    if not interleave or length == 0:
        return values[: int(length)]
    unit, rows, _groups = _interleave_layout(int(length), unit_bits)
    expected = rows * _INTERLEAVE_COLS * unit
    if values.size != expected:
        return values[: int(length)]
    blocks = values.reshape(_INTERLEAVE_COLS, rows, unit)
    return blocks.transpose(1, 0, 2).reshape(-1)[: int(length)]


def encode_payload(info_bytes: bytes, fec: str, interleave: bool = False) -> np.ndarray:
    """信息字节 -> 已编码载荷比特（含可选交织）."""
    mode = normalize_fec(fec)
    data = bytes(info_bytes)
    if mode == "none":
        return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big").astype(np.uint8)
    unit = interleave_unit_bits(mode)
    if mode == "conv":
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big").astype(np.uint8)
        return _apply_interleave(ConvolutionalCode().encode(bits), interleave, unit)
    if mode == "rs":
        n, k = _rs_params(len(data))
        codeword = ReedSolomon(n=n, k=k).encode(data)
        return _apply_interleave(np.unpackbits(codeword.astype(np.uint8), bitorder="big"), interleave, unit)
    if mode == "ccsds":
        depth, n, k, total = _ccsds_params(len(data))
        padded = data + b"\x00" * (total - len(data))
        coded = CCSDSCode(interleave=depth, n=n, k=k).encode(padded)
        return _apply_interleave(coded, interleave, unit)
    k_bits, used_bits = _ldpc_info_bits(len(data))
    info = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big").astype(np.uint8)
    info = np.concatenate([info, np.zeros(k_bits - info.size, dtype=np.uint8)])
    return _apply_interleave(_ldpc_code().encode(info), interleave, unit)


def erasures_from_llr(
    llr: np.ndarray,
    info_bytes: int,
    interleave: bool = False,
    max_erasures: int | None = None,
    threshold: float = 1.0,
    unit_bits: int = 8,
    min_run: int = 3,
) -> list[int]:
    """按软信息挑"已知不可信"的符号位置（擦除），交给 RS 联合译码.

    思路：把编码后的比特流按 RS 的符号宽度（字节）分组，取每组 |LLR| 的均值作为该符号的
    可信度；均值低于门限的符号标为擦除。这样"解调器知道哪里不可信"这一信息就被用上了，
    联合译码的能力从 t 个符号扩到 2t（含少量未知错误时按 2f + e <= 2t 分配）。
    """
    values = np.asarray(llr, dtype=np.float64).reshape(-1)
    coded_bits = coded_length_bits(int(info_bytes), "rs", bool(interleave))
    if values.size < coded_bits or unit_bits <= 0:
        return []
    usable = values[:coded_bits]
    inner = _inner_length_bits(int(info_bytes), "rs")
    inner = inner - (inner % unit_bits)
    if inner <= 0:
        return []
    usable = usable[:inner] if usable.size >= inner else usable
    symbols = usable[: (usable.size // unit_bits) * unit_bits].reshape(-1, unit_bits)
    if symbols.size == 0:
        return []
    reliability = np.mean(np.abs(symbols), axis=1)
    n_symbols = reliability.size
    limit = int(max_erasures) if max_erasures is not None else n_symbols
    weak = reliability < float(threshold)

    # 只把**连续成段**的弱区当擦除（突发/遮挡的形态），不把 AWGN 下零散的低置信度符号当擦除。
    # 实测教训：按"|LLR| 低于门限就标擦除"在纯 AWGN 下会把**本来正确**的符号标掉，
    # 而每个错误擦除要吃掉半个纠错能力（2f + e <= 2t），端到端门限反而从 8.78 dB 升到 9.20 dB。
    # 突发场景则相反：位置真实可信，擦除能救回超出 t 的损伤（见 check_m7 M7-C）。
    chosen: list[int] = []
    index = 0
    while index < n_symbols:
        if not weak[index]:
            index += 1
            continue
        start = index
        while index < n_symbols and weak[index]:
            index += 1
        if index - start >= max(1, int(min_run)):
            chosen.extend(range(start, index))
    return chosen[: max(0, min(limit, n_symbols))]


def decode_payload(
    bits: np.ndarray,
    fec: str,
    info_bytes: int,
    interleave: bool = False,
    llr: np.ndarray | None = None,
    erasures: list[int] | None = None,
    erasure_from_llr: bool = False,
    max_erasures: int | None = None,
) -> tuple[bytes, dict]:
    """已编码载荷 -> 信息字节；llr 给出时走软判决（Viterbi / LDPC 都用得上）."""
    mode = normalize_fec(fec)
    expected = coded_length_bits(info_bytes, mode, interleave)
    raw = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if raw.size < expected:
        raise ValueError("编码段长度不足：{} < {}".format(raw.size, expected))
    raw = raw[:expected]
    unit = interleave_unit_bits(mode)
    raw_llr = None
    if llr is not None:
        values = np.asarray(llr, dtype=np.float64).reshape(-1)
        if values.size >= expected:
            raw_llr = _undo_interleave_llr(values[:expected], interleave, expected, unit)
    info = {"fec": mode, "soft": bool(raw_llr is not None), "info_bytes": int(info_bytes)}
    if mode == "none":
        data = np.packbits(raw, bitorder="big").tobytes()
        return data[: int(info_bytes)], info
    inner = _inner_length_bits(int(info_bytes), mode)
    payload = _undo_interleave(raw, interleave, expected, unit)[:inner]
    if raw_llr is not None:
        raw_llr = raw_llr[:inner]

    if mode == "conv":
        code = ConvolutionalCode()
        decoded = code.decode_soft(raw_llr) if raw_llr is not None else code.decode_hard(payload)
        data = np.packbits(decoded, bitorder="big").tobytes()
        info["decoded_bits"] = int(decoded.size)
        return data[: int(info_bytes)], info
    if mode == "rs":
        n, k = _rs_params(int(info_bytes))
        received = np.packbits(payload, bitorder="big").tobytes()[:n]
        if erasures is None and erasure_from_llr and llr is not None:
            # 联合译码接进链路的关键一步：擦除位置由**软信息**给出，而不是靠人工标注
            erasures = erasures_from_llr(
                llr, int(info_bytes), interleave=bool(interleave), max_erasures=max_erasures
            )
        # erasures：已知不可信的**符号（字节）**位置。帧同步/置信度能指出可疑位置时，
        # 纠错能力从 t 个符号翻倍到 nsym = 2t 个符号。
        decoded, corrected, failed = ReedSolomon(n=n, k=k).decode(received, erasures=erasures)
        info.update({
            "corrected_symbols": int(corrected),
            "uncorrectable": bool(failed),
            "erasures": len(erasures) if erasures else 0,
        })
        return bytes(decoded[:k]), info
    if mode == "ccsds":
        depth, n, k, _total = _ccsds_params(int(info_bytes))
        code = CCSDSCode(interleave=depth, n=n, k=k)
        flat_llr = raw_llr if raw_llr is not None else np.where(payload == 0, 1.0, -1.0)
        result = code.decode(flat_llr, soft=raw_llr is not None)
        info.update({
            "corrected_symbols": int(result["corrected_symbols"]),
            "uncorrectable_codewords": int(result["uncorrectable_codewords"]),
        })
        return bytes(result["info"][: int(info_bytes)]), info
    k_bits, _ = _ldpc_info_bits(int(info_bytes))
    code = _ldpc_code()
    if raw_llr is not None:
        llr_values = raw_llr
    else:
        llr_values = np.where(payload == 0, 4.0, -4.0)
    result = code.decode(llr_values)
    info.update({"converged": bool(result["converged"]), "iterations": int(result["iterations"])})
    decoded = result["info"][:k_bits]
    data = np.packbits(decoded, bitorder="big").tobytes()
    return data[: int(info_bytes)], info


def _INTERLEAVE_PAD(length: int) -> int:
    if length <= 0:
        return 0
    cols = _INTERLEAVE_COLS
    rows = int(np.ceil(length / float(cols)))
    return rows * cols - int(length)


def decode_fec(*args, **kwargs):
    """decode_payload 的别名（语义更直白）."""
    return decode_payload(*args, **kwargs)
