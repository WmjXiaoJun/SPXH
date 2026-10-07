"""帧结构定义与打包/解析.

帧格式（MSB-first）::

    [0 : 13]                 Barker-13 前导           13 bit
    [13 : 45]                32 位帧头                 4 byte
    [45 : 45 + 8L]           载荷                      L byte
    [45 + 8L : 49 + 8L]      CRC-16-CCITT（大端）      2 byte

帧头 32 位布局::

    byte0  0xA5                       包标识 MAGIC
    byte1  bit7-4 调制码 (MOD_CODES)
           bit3-1 编码码 (FEC_CODES，5 种：none/conv/rs/ccsds/ldpc)
           bit0   交织标志
           （M6 起原"保留位"并入 fec 字段；fec=none 时字节与旧布局完全一致）
    byte2-3  载荷长度（uint16 大端，单位字节）

CRC 覆盖范围：帧头 4 字节 + 载荷字节（不含前导）。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from spxh.core.bits import bits_to_bytes, bits_to_int, bytes_to_bits
from spxh.core.fec.codec import coded_length_bits, decode_payload, encode_payload, normalize_fec
from spxh.core.framing.barker import BARKER_LEN, barker_bits
from spxh.core.framing.crc import crc16_ccitt, crc8

#: 软前导判据：最多容忍多少位不符、软相关至少要多大
#: （13 位 Barker；容忍 2 位、软相关门限 0 = 支持该前导的证据占优。
#:  宁可放过、由帧头 CRC-8 与帧 CRC-16 两级校验兜底，也不要因为 1~2 位错就整帧丢掉。）
_SOFT_PREAMBLE_MAX_ERRORS = 2
_SOFT_PREAMBLE_MIN_SCORE = 0.0
from spxh.core.types import FRAME_MAGIC, MOD_BY_CODE, MOD_CODES, Header, normalize_modulation

__all__ = [
    "FEC_CODES",
    "FEC_BY_CODE",
    "HEADER_BYTES",
    "HEADER_BITS",
    "PREAMBLE_BITS",
    "CRC_BITS",
    "CRC_BYTES",
    "MAX_PAYLOAD_BYTES",
    "header_bytes",
    "parse_header_bytes",
    "frame_length_bits",
    "build_frame",
    "parse_frame",
    "verify_frame",
]

# 帧头 byte1 布局（M6 起把原来的保留位并入 fec 字段，腾出 3 bit 以容纳 LDPC）：
#   bit7-4 调制码 | bit3-1 fec 码 | bit0 交织标志
# fec="none" 时新旧布局字节完全相同（0<<2 == 0<<1），因此已有数据与测试不受影响。
FEC_CODES: dict[str, int] = {"none": 0, "conv": 1, "rs": 2, "ccsds": 3, "ldpc": 4}
FEC_BY_CODE: dict[int, str] = {v: k for k, v in FEC_CODES.items()}

# 帧头 4 字节 + 1 字节 CRC-8（M9 起）。第 5 字节是前 4 字节的 CRC-8，用来
# 1) 立刻发现帧头被打错；2) 让"载荷长度也被打错"的情形可以**盲恢复**：
#    对 (编码, 交织, 长度) 的假设重写帧头、用 CRC-8 过滤，再做帧 CRC 校验。
HEADER_BYTES = 5
HEADER_PAYLOAD_BYTES = 4
HEADER_BITS = HEADER_BYTES * 8
HEADER_CRC_BYTES = 1
PREAMBLE_BITS = BARKER_LEN
CRC_BYTES = 2
CRC_BITS = CRC_BYTES * 8
MAX_PAYLOAD_BYTES = 0xFFFF


def header_bytes(header: Header) -> bytes:
    """结构化帧头 -> 4 字节（大端），同时完成范围校验."""
    modulation = normalize_modulation(header.modulation)
    mod_code = MOD_CODES[modulation] & 0x0F

    fec = str(header.fec).strip().lower()
    if fec not in FEC_CODES:
        raise ValueError("unsupported fec: " + str(header.fec) + " (supported: " + ", ".join(FEC_CODES) + ")")
    fec_code = FEC_CODES[fec] & 0x07

    if int(header.reserved) not in (0, 1):
        raise ValueError("reserved field must be 0 or 1")
    if int(header.reserved) != 0:
        # 保留位在 M6 起并入 fec 字段（腾出 3 bit 以容纳 LDPC），不再单独传输
        raise ValueError("reserved 位已并入 fec 字段，必须为 0")

    length = int(header.payload_len)
    if length < 0 or length > MAX_PAYLOAD_BYTES:
        raise ValueError("payload_len out of range: " + str(length))

    byte1 = (mod_code << 4) | ((fec_code & 0x07) << 1) | (0x01 if header.interleave else 0x00)
    body = bytes([FRAME_MAGIC, byte1, (length >> 8) & 0xFF, length & 0xFF])
    return body + bytes([crc8(body)])


def header_crc_ok(data: bytes) -> bool:
    """帧头 CRC-8 自校验（数据不足 5 字节时视为不通过）."""
    if len(data) < HEADER_BYTES:
        return False
    return crc8(bytes(data[:HEADER_PAYLOAD_BYTES])) == int(data[HEADER_PAYLOAD_BYTES])


def parse_header_bytes(data: bytes) -> Header:
    """5 字节 -> 结构化帧头；MAGIC 或调制码非法时抛 ValueError.

    注意：CRC-8 不通过**不抛异常**（帧头被打错时仍要尽力解析，
    由调用方用 header_crc_ok() 判断并把状态报出去）。
    """
    if len(data) != HEADER_BYTES:
        raise ValueError("header must be exactly " + str(HEADER_BYTES) + " bytes, got " + str(len(data)))
    if data[0] != FRAME_MAGIC:
        raise ValueError("bad frame magic: 0x{:02x} (expected 0x{:02x})".format(data[0], FRAME_MAGIC))
    byte1 = data[1]
    mod_code = (byte1 >> 4) & 0x0F
    if mod_code not in MOD_BY_CODE:
        raise ValueError("unknown modulation code: " + str(mod_code))
    fec_code = (byte1 >> 1) & 0x07
    if fec_code not in FEC_BY_CODE:
        raise ValueError("unknown fec code: " + str(fec_code))
    return Header(
        modulation=MOD_BY_CODE[mod_code],
        fec=FEC_BY_CODE[fec_code],
        interleave=bool(byte1 & 0x01),
        payload_len=(data[2] << 8) | data[3],
        reserved=0,
    )


def frame_length_bits(payload_len: int, fec: str = "none", interleave: bool = False) -> int:
    """给定载荷字节数与编码方式，返回整帧比特数.

    两种布局（M7 起）：

    * fec = none：Barker + 帧头 + 信息字节 + CRC-16（信息与 CRC 都不编码，与 M0 完全一致）
    * fec != none：Barker + 帧头 + **编码(信息字节 || CRC-16)**
      即把 CRC **放进编码里**：不然 CRC 字段自己是暴露的，低信噪比下"载荷完全正确但
      CRC 被打错"会被误判为失败（M6 实测 8 dB 时约 15% 的帧这样丢）。
      代价是接收端必须先译码才知道 CRC —— 这没问题，因为帧头已经告诉了它用哪种码。
      帧头本身仍在编码之外（接收端必须先知道码参数），它的保护靠"CRC 覆盖帧头"来间接保证。
    """
    if payload_len < 0 or payload_len > MAX_PAYLOAD_BYTES:
        raise ValueError("payload_len out of range: " + str(payload_len))
    mode = normalize_fec(fec)
    if mode == "none":
        return PREAMBLE_BITS + HEADER_BITS + int(payload_len) * 8 + CRC_BITS
    body = coded_length_bits(int(payload_len) + CRC_BYTES, mode, bool(interleave))
    return PREAMBLE_BITS + HEADER_BITS + body


def build_frame(
    payload: bytes,
    modulation: str,
    fec: str = "none",
    interleave: bool = False,
    include_preamble: bool = True,
) -> tuple[np.ndarray, Header, int]:
    """打包一帧，返回 (比特数组, 帧头, CRC 值)."""
    payload = bytes(payload)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("payload too long: " + str(len(payload)) + " bytes")
    header = Header(
        modulation=normalize_modulation(modulation),
        fec=fec,
        interleave=bool(interleave),
        payload_len=len(payload),
    )
    hb = header_bytes(header)
    # CRC 覆盖 帧头 + **信息字节**：接收端译码之后再校验，这样 CRC 才真正说明
    # "解出来的载荷是对的"。
    crc = crc16_ccitt(hb + payload)
    crc_bytes = crc.to_bytes(CRC_BYTES, "big")
    parts = []
    if include_preamble:
        parts.append(barker_bits(BARKER_LEN))
    parts.append(bytes_to_bits(hb))
    if normalize_fec(header.fec) == "none":
        parts.append(bytes_to_bits(payload))
        parts.append(bytes_to_bits(crc_bytes))
    else:
        # CRC 也进编码：校验字段自己受保护
        parts.append(encode_payload(payload + crc_bytes, header.fec, header.interleave))
    return np.concatenate(parts).astype(np.uint8), header, crc


def parse_frame(
    bits,
    llr=None,
    force_fec: str | None = None,
    force_interleave: bool | None = None,
    erasures: list[int] | None = None,
    force_payload_len: int | None = None,
    erasure_decoding: bool = False,
    max_erasures: int | None = None,
) -> dict[str, Any]:
    """从帧起始处解析一帧.

    返回 dict：ok / reason / header / payload / crc_expected / crc_actual /
    crc_ok / preamble_ok / bits_consumed。
    """
    arr = np.asarray(bits).reshape(-1).astype(np.uint8)
    result: dict[str, Any] = {
        "ok": False,
        "reason": "",
        "header": None,
        "payload": b"",
        "crc_expected": None,
        "crc_actual": None,
        "crc_ok": False,
        "preamble_ok": False,
        "bits_consumed": 0,
    }
    minimum = PREAMBLE_BITS + HEADER_BITS + CRC_BITS
    if arr.size < minimum:
        result["reason"] = "too short: " + str(arr.size) + " < " + str(minimum) + " bits"
        return result

    # 前导校验：**软判决**优先。硬比对要求 13 位全对，在 5~8 dB（BER 约 2~5%）下
    # 单帧通过率只有 50%~75%，这是低信噪比帧召回损失的主要来源（实测）。
    # 有 LLR 时改用"软相关 + 容错门限"：允许少量比特出错，但要求整体证据足够强，
    # 再用后面的帧头 CRC-8 与帧 CRC 把误判兜住 —— 不是放宽判据，而是换成更充分利用信息量的判据。
    expected = barker_bits(BARKER_LEN)
    preamble = arr[:PREAMBLE_BITS]
    mismatches = int(np.sum(preamble != expected))
    hard_ok = mismatches == 0
    soft_score = None
    if not hard_ok and llr is not None:
        values = np.asarray(llr, dtype=np.float64).reshape(-1)
        if values.size >= PREAMBLE_BITS:
            head = values[:PREAMBLE_BITS]
            # LLR 约定：正 => 更可能是 0。把期望位映射成 +1/-1 后做软相关。
            signed = np.where(expected == 0, 1.0, -1.0)
            soft_score = float(np.sum(head * signed))
    allow_soft = (
        not hard_ok
        and soft_score is not None
        and mismatches <= int(_SOFT_PREAMBLE_MAX_ERRORS)
        and soft_score >= float(_SOFT_PREAMBLE_MIN_SCORE)
    )
    result["preamble_ok"] = bool(hard_ok or allow_soft)
    result["preamble_mismatches"] = int(mismatches)
    if soft_score is not None:
        result["preamble_soft_score"] = round(soft_score, 3)
    result["preamble_soft_accept"] = bool(allow_soft)
    if not result["preamble_ok"]:
        result["reason"] = "preamble mismatch（{} 位不符）".format(mismatches)
        return result

    hb_rx = bits_to_bytes(arr[PREAMBLE_BITS:PREAMBLE_BITS + HEADER_BITS])
    try:
        header = parse_header_bytes(hb_rx)
    except ValueError as exc:
        result["reason"] = "header error: " + str(exc)
        return result
    if force_payload_len is not None:
        header = Header(
            modulation=header.modulation, fec=header.fec, interleave=header.interleave,
            payload_len=int(force_payload_len), reserved=header.reserved,
        )
    result["header"] = header
    result["header_crc_ok"] = bool(header_crc_ok(hb_rx)) if force_payload_len is None else None
    result["header_crc8"] = int(hb_rx[HEADER_PAYLOAD_BYTES]) if len(hb_rx) >= HEADER_BYTES else None
    result["crc8_expected"] = int(crc8(hb_rx[:HEADER_PAYLOAD_BYTES])) if len(hb_rx) >= HEADER_BYTES else None

    # 盲编码识别：按假设**重写帧头里的 fec/interleave 位**再算 CRC。
    # 这样帧头字段本身被打错时也能被"假设 + CRC 验证"修正回来 —— 否则 CRC 用的还是
    # 被打错的帧头字节，正确的码假设也永远对不上。
    if force_fec is None and force_interleave is None and force_payload_len is None:
        fec_used = header.fec
        inter_used = header.interleave
        hb = hb_rx
    else:
        fec_used = normalize_fec(header.fec if force_fec is None else force_fec)
        inter_used = header.interleave if force_interleave is None else bool(force_interleave)
        hb = header_bytes(
            Header(
                modulation=header.modulation,
                fec=fec_used,
                interleave=inter_used,
                payload_len=header.payload_len,
            )
        )
    result["assumed"] = {"fec": fec_used, "interleave": bool(inter_used)}

    try:
        total = frame_length_bits(header.payload_len, fec_used, inter_used)
    except ValueError as exc:
        # 起止位置偏了就可能把载荷当帧头读出来（payload_len/fec 是垃圾值），
        # 这里必须"当作解析失败"而不是抛异常 —— 否则一个错误候选就能把整次提取打断
        result["reason"] = "invalid header: " + str(exc)
        return result
    if arr.size < total:
        result["reason"] = "truncated frame: have " + str(arr.size) + " bits, need " + str(total)
        return result

    payload_start = PREAMBLE_BITS + HEADER_BITS
    mode = normalize_fec(fec_used)
    fec_info: dict[str, Any] = {}
    if mode == "none":
        payload_end = payload_start + header.payload_len * 8
        payload = bits_to_bytes(arr[payload_start:payload_end]) if header.payload_len > 0 else b""
        crc_rx = bits_to_int(arr[payload_end:payload_end + CRC_BITS])
    else:
        # 已编码布局：解码 (信息 || CRC-16)，CRC 在编码之内
        body_bits = coded_length_bits(header.payload_len + CRC_BYTES, mode, inter_used)
        payload_end = payload_start + body_bits
        section = arr[payload_start:payload_end]
        section_llr = None
        if llr is not None:
            values = np.asarray(llr, dtype=np.float64).reshape(-1)
            if values.size >= payload_end:
                section_llr = values[payload_start:payload_end]
        # 擦除"该信多少"由 **CRC 仲裁**，而不是我们自己拍一个数：
        # 给的擦除位置可能比真实损伤宽（检测区间有边界扩展），而每个错误擦除要吃掉
        # 半个纠错能力（2f + e <= 2t）。所以从"全信"开始，逐步减少擦除数，谁先过 CRC 用谁；
        # 都不行就用最后一次尝试的结果如实报错。实测这条是"自动擦除不帮倒忙"的关键。
        budgets: list[int] = [0]
        if erasures:
            count = len(list(erasures))
            # 0 一定要在候选里：它等价于"完全不标擦除"。有了它，"自动擦除"在最坏情况下
            # 也只能退回到普通译码，**永远不会比不标更差**（实测过：擦除数超过 2t 时
            # 直接判不可纠，9 dB 下 6/8 会掉到 4/8）。
            budgets = sorted({0, count, 16, 8, 4, 2}, reverse=True)
        last_error: Optional[str] = None
        decoded = None
        for budget in budgets:
            attempt_erasures = (list(erasures)[:budget] if budget else None) if erasures else None
            try:
                candidate, candidate_info = decode_payload(
                    section, mode, header.payload_len + CRC_BYTES, inter_used,
                    llr=section_llr, erasures=attempt_erasures,
                    erasure_from_llr=bool(erasure_decoding) and not attempt_erasures,
                    max_erasures=max_erasures,
                )
            except Exception as exc:  # noqa: BLE001 - 译码失败要如实上报而不是崩
                last_error = "{}: {}".format(type(exc).__name__, exc)
                continue
            candidate_payload = candidate[: header.payload_len]
            candidate_crc = int.from_bytes(candidate[header.payload_len : header.payload_len + CRC_BYTES], "big")
            if candidate_crc == crc16_ccitt(hb + candidate_payload):
                decoded, payload, crc_rx, fec_info = candidate, candidate_payload, candidate_crc, dict(candidate_info)
                fec_info["erasure_budget_used"] = int(budget)
                break
            if decoded is None:      # 记住第一次尝试的结果，作为"都不行"时的如实回报
                decoded, payload, crc_rx, fec_info = candidate, candidate_payload, candidate_crc, dict(candidate_info)
        if decoded is None:
            result["reason"] = "fec decode error: " + str(last_error or "unknown")
            result["fec"] = {"fec": mode, "error": last_error or "unknown"}
            return result
    crc_calc = crc16_ccitt(hb + payload)
    result["fec"] = dict(fec_info, fec=mode, interleave=bool(inter_used), crc_inside_code=(mode != "none"))

    result["payload"] = payload
    result["crc_expected"] = int(crc_calc)
    result["crc_actual"] = int(crc_rx)
    result["crc_ok"] = bool(crc_rx == crc_calc)
    result["ok"] = bool(result["crc_ok"])
    result["bits_consumed"] = int(total)
    if not result["ok"]:
        result["reason"] = "crc mismatch: got 0x{:04x}, expected 0x{:04x}".format(int(crc_rx), int(crc_calc))
    return result


def verify_frame(bits) -> tuple[bool, str]:
    """便捷接口：返回 (是否通过, 失败原因)."""
    parsed = parse_frame(bits)
    return bool(parsed["ok"]), str(parsed["reason"])
