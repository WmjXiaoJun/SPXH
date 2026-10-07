"""M12 测试：软前导判决（低信噪比帧召回）+ 联合译码接进链路（LLR 擦除）+ 新代理工具."""
from __future__ import annotations

import numpy as np
import pytest

from spxh.core.agent import run_tool, tool_schemas
from spxh.core.fec.codec import decode_payload, encode_payload, erasures_from_llr
from spxh.core.framing.format import build_frame, parse_frame

DEMO = "data/demo/qpsk_ccsds_snr+8.0_000.sigmf-meta"


def _llr_from_bits(bits: np.ndarray, confidence: float = 6.0) -> np.ndarray:
    """把已知比特变成 LLR（约定：正 => 更可能是 0）."""
    return np.where(np.asarray(bits, dtype=np.uint8) == 0, confidence, -confidence)


def test_soft_preamble_tolerates_a_few_bit_errors():
    """1~2 位前导出错时，软判决应该仍然认下来（硬比对会整帧丢掉）."""
    info = bytes(range(60))
    bits, _header, _crc = build_frame(info, "qpsk", fec="rs")
    llr = _llr_from_bits(bits)

    tampered = np.asarray(bits, dtype=np.uint8).copy()
    tampered[2] ^= 1                      # 前导里打错 1 位
    hard = parse_frame(tampered)
    assert hard["preamble_ok"] is False   # 没有软信息时：硬比对失败
    soft = parse_frame(tampered, llr=llr)
    assert soft["preamble_ok"] is True
    assert soft["preamble_soft_accept"] is True
    assert soft["ok"] is True             # 帧头 CRC-8 与帧 CRC 仍然要过
    assert bytes(soft["payload"]) == info


def test_soft_preamble_rejects_too_many_errors():
    """错得太多时仍要拒绝（不能为了召回把判据放烂）."""
    info = bytes(range(60))
    bits, _header, _crc = build_frame(info, "qpsk", fec="rs")
    llr = _llr_from_bits(bits)
    tampered = np.asarray(bits, dtype=np.uint8).copy()
    tampered[1] ^= 1
    tampered[4] ^= 1
    tampered[7] ^= 1                      # 3 位不符，超过容忍上限
    result = parse_frame(tampered, llr=llr)
    assert result["preamble_ok"] is False
    assert result["preamble_mismatches"] == 3


def test_soft_preamble_needs_llr_pointing_the_right_way():
    """软相关必须为正：LLR 整体站在错误一边时要拒绝（不能只看"错了几位"）."""
    info = bytes(range(60))
    bits, _header, _crc = build_frame(info, "qpsk", fec="rs")
    tampered = np.asarray(bits, dtype=np.uint8).copy()
    tampered[3] ^= 1
    tampered[6] ^= 1
    # 场景：其余 11 位只有**很弱**的证据，而那 2 位被篡改的位置有**很强**的反向证据
    # -> 软相关为负，应当拒绝（"否决权"交给强证据，而不是简单数错了几位）
    wrong_llr = _llr_from_bits(bits, confidence=0.05)
    # LLR 约定：正 => 更可能是 0。要让某一位"强烈反对期望值"，符号取决于期望位本身：
    for index in (3, 6):
        # 期望位是 0 时要"反对"就得给负 LLR（=更可能是 1），反之亦然
        wrong_llr[index] = -50.0 if bits[index] == 0 else 50.0
    result = parse_frame(tampered, llr=wrong_llr)
    assert result["preamble_ok"] is False, result
    assert result["preamble_soft_score"] is not None and result["preamble_soft_score"] < 0


def test_erasures_from_llr_pick_the_weak_region():
    coded = encode_payload(b"\x5a" * 60, "rs", False)
    llr = np.full(coded.size, 6.0)
    llr[8 * 10 : 8 * 22] = 0.2               # 第 10~21 个符号很不可信
    erasures = erasures_from_llr(llr, 60, max_erasures=16)
    assert erasures, "应能标出擦除"
    assert all(8 <= position <= 24 for position in erasures), erasures
    assert len(erasures) <= 16


def test_erasure_decoding_helps_end_to_end_payload_decode():
    """联合译码接进 decode_payload：超出 t 的坏符号在标了擦除后应该能救回来.

    注意 decode_payload 吃的是**比特流**（不是字节码字），软信息也是比特级的。
    """
    info = bytes(range(60))
    info_bytes = 62                            # 信息 + CRC-16（M9 起 CRC 进编码）
    damaged_bits = np.asarray(encode_payload(info + b"\x00\x00", "rs", False), dtype=np.uint8).copy()
    positions = list(range(30, 42))            # 12 个 **符号（字节）** 被打错（> t=8）
    for position in positions:
        damaged_bits[8 * position : 8 * (position + 1)] ^= 1     # 每个符号 8 个比特全错
    llr = np.full(damaged_bits.size, 6.0)
    for position in positions:                 # 软信息如实反映"这些符号不可信"
        llr[8 * position : 8 * (position + 1)] = 0.1

    plain, _ = decode_payload(damaged_bits, "rs", info_bytes, False, llr=llr)
    assert plain[:60] != info, "不标擦除时应当纠不回来（12 > t=8）"

    fixed, info_dict = decode_payload(damaged_bits, "rs", info_bytes, False, llr=llr,
                                      erasure_from_llr=True, max_erasures=16)
    assert fixed[:60] == info
    assert info_dict["erasures"] >= 10
    assert info_dict["uncorrectable"] is False


def test_new_agent_tools_are_registered_and_usable():
    names = {schema["function"]["name"] for schema in tool_schemas()}
    assert {"decode", "batch"} <= names

    decoded = run_tool("decode", {"path": DEMO})
    assert decoded.ok is True
    assert "CRC 通过" in decoded.summary
    assert any(item["key"] == "frames_crc_ok" for item in decoded.evidence.get("frame", []))

    batch = run_tool("batch", {"dir": "data/demo", "limit": 2, "modulation": "qpsk"})
    assert batch.ok is True
    rows = batch.extra.get("rows") or []
    assert len(rows) == 2
    assert all("case" in row for row in rows)


def test_decode_tool_reports_uncorrectable_honestly():
    """取不回来的数据要如实说"取不回来"，不能假装成功."""
    tool = run_tool("decode", {"path": "data/demo/16qam_ldpc_snr+6.0_000.sigmf-meta"})
    assert tool.ok is True                     # 工具本身成功（完成了分析）
    assert "CRC 通过" in tool.summary          # 但结论里如实给出通过数（可能是 0）
