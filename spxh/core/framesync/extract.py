"""M5 端到端：解调 -> 前导同步（含相位模糊求解）-> 帧头解析 -> CRC -> 载荷.

与 M3/M4 的分工：
* M3 负责"符号级"的定时/载波跟踪与 LLR；
* M4 负责"比特级"的纠错译码；
* M5 负责"帧级"的同步与封装还原 —— 用 Barker-13 前导定位帧起点，顺便把 M3 遗留的
  相位模糊解掉（前导是已知序列，相关峰的复数相位就是模糊角），再解析帧头、校验 CRC、
  取出载荷。整条链**不依赖真值**；真值只在最后用来对账（逐字节比对）。

为什么不能"取相关峰最大者"就完事
--------------------------------
Barker-13 的低旁瓣是针对**它自己的非周期自相关**说的；当它埋进**随机载荷**里时，
随机数据在某个错位处凑出 0.9 以上的归一化相关是常事（实测 BPSK 20 dB 下最大峰
落在载荷深处、解析出 0xfc 这种非法 MAGIC）。因此这里做的是"**相关出候选 + 逐候选试解析**"：
每个候选峰给出一个旋转角与起点，按帧结构连续解析并统计"成功帧数 / CRC 通过数"，
取得分最高者。这既压掉虚警，又顺便把相位模糊定下来。

单帧 CRC 失败不应该让后面所有帧都丢掉：解析失败时按"预期帧长"跳过继续，
并把失败帧如实记录（前导/帧头/CRC 三段状态分别标出）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from spxh.core.demod import DemodResult, demodulate
from spxh.core.demod.filters import fractional_delay
from spxh.core.demod.llr import max_log_llr, symbol_noise_variance
from spxh.core.estimate import AnalyzeResult, analyze_signal
from spxh.core.framing.barker import BARKER_LEN, barker_bits
from spxh.core.framing.crc import crc8
from spxh.core.framing.format import (
    FEC_CODES,
    HEADER_BITS,
    HEADER_BYTES,
    HEADER_PAYLOAD_BYTES,
    PREAMBLE_BITS,
    frame_length_bits,
    parse_frame,
)
from spxh.core.framesync.preamble import (
    correlate_bits,
    correlate_symbols,
    detection_symbols,
    preamble_symbols,
)
from spxh.core.synth import mapping
from spxh.core.types import MOD_BITS_PER_SYMBOL, GroundTruth, Signal, jsonable

__all__ = ["FrameExtraction", "extract_frames"]

_SYMMETRY = {"bpsk": 2, "qpsk": 4, "8psk": 8, "16qam": 4}
_MIN_FRAME_BITS = PREAMBLE_BITS + HEADER_BITS + 16
_DEFAULT_FRAME_BITS = PREAMBLE_BITS + HEADER_BITS + 8 * 64 + 16   # 未知帧长时的跳过步长
_BLIND_MAX_BYTES = 512             # 盲搜索的长度上界（每帧 512 字节；再长请显式给出）


@dataclass
class FrameExtraction:
    modulation: str
    modulation_source: str
    symbol_rate: float
    cfo_hz: float
    preamble_bits: str
    correlation_peak: float
    peak_to_sidelobe_db: float
    frame_start_symbol: float
    frame_start_sample: int
    ambiguity_rotation_deg: float
    locked: bool
    correlation_index: np.ndarray
    correlation_value: np.ndarray
    correlation_stride: int
    frames: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    blind: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, max_payload_preview: int = 256, corr_points: int = 400) -> dict[str, Any]:
        index = np.asarray(self.correlation_index, dtype=np.int64)
        value = np.asarray(self.correlation_value, dtype=np.float64)
        if index.size and index.size > int(corr_points):
            stride = int(np.ceil(index.size / float(corr_points)))
            index = index[::stride]
            value = value[::stride]
        return jsonable(
            {
                "modulation": self.modulation,
                "modulation_source": self.modulation_source,
                "symbol_rate": self.symbol_rate,
                "cfo_hz": self.cfo_hz,
                "sync": {
                    "preamble": "barker13",
                    "preamble_bits": self.preamble_bits,
                    "correlation_peak": self.correlation_peak,
                    "peak_to_sidelobe_db": self.peak_to_sidelobe_db,
                    "frame_start_symbol": self.frame_start_symbol,
                    "frame_start_sample": int(self.frame_start_sample),
                    "ambiguity_rotation_deg": self.ambiguity_rotation_deg,
                    "locked": bool(self.locked),
                    "correlation": {
                        "index": index.tolist(),
                        "value": value.tolist(),
                        "stride": int(self.correlation_stride),
                    },
                    "blind": jsonable(self.blind) if self.blind else None,
                },
                "frames": self.frames,
                "summary": self.summary,
                "warnings": list(self.warnings),
                "evidence": jsonable(self.evidence),
            }
        )


def _payload_preview(payload: bytes, limit: int) -> dict[str, Any]:
    data = bytes(payload)
    text = "".join(chr(b) if 32 <= b < 127 else "." for b in data[:limit])
    return {
        "bytes": len(data),
        "hex": data[:limit].hex().upper(),
        "ascii": text,
        "head": [int(b) for b in data[:32]],
        "truncated": bool(len(data) > limit),
    }


def _snap_rotation(angle_rad: float, modulation: str) -> float:
    order = _SYMMETRY.get(mapping.normalize_modulation(modulation), 1)
    if order <= 1:
        return 0.0
    step = 2.0 * np.pi / order
    return float(np.round(angle_rad / step) * step)


def _candidate_peaks(magnitude: np.ndarray, spacing: int, limit: int = 40, floor_ratio: float = 0.35, min_abs: float = 0.25) -> list[int]:
    """挑归一化相关峰候选：按幅值排序 + 间隔抑制（防止同一主瓣被重复计数）."""
    values = np.asarray(magnitude, dtype=np.float64)
    if values.size == 0:
        return []
    threshold = max(float(min_abs), float(floor_ratio) * float(np.max(values)))
    chosen: list[int] = []
    for index in np.argsort(values)[::-1]:
        index = int(index)
        if values[index] < threshold:
            break
        if any(abs(index - other) < max(2, int(spacing)) for other in chosen):
            continue
        chosen.append(index)
        if len(chosen) >= int(limit):
            break
    if not chosen:
        chosen = [int(np.argmax(values))]
    return chosen


_BLIND_HYPOTHESES: list[tuple[str, bool]] = [
    (fec, inter) for fec in ("none", "conv", "rs", "ccsds", "ldpc") for inter in (False, True)
]


def _blind_header_candidates(bits: np.ndarray, cursor: int, limit: int = 12) -> list[dict[str, Any]]:
    """帧头 CRC-8 不通过时，枚举 (编码, 交织, 长度) 假设并用 CRC-8 过滤.

    CRC-8 是一步"很便宜的筛子"（1 字节开销、255/256 的随机错误能被挡掉），
    滤完之后只剩极少数假设需要真正去译码。这也是让**载荷长度被打错**仍可恢复的关键。
    """
    start = cursor + PREAMBLE_BITS
    if start < 0 or start + HEADER_BITS > bits.size:
        return []
    header_bits = np.asarray(bits[start : start + HEADER_BITS], dtype=np.uint8).reshape(-1)
    if header_bits.size < HEADER_BITS:
        return []
    raw = np.packbits(header_bits, bitorder="big").tobytes()
    if len(raw) < HEADER_BYTES:
        return []
    received = int(raw[HEADER_PAYLOAD_BYTES])
    body = raw[:HEADER_PAYLOAD_BYTES]
    out: list[dict[str, Any]] = []
    for fec in ("none", "conv", "rs", "ccsds", "ldpc"):
        for interleave in (False, True):
            byte1 = (body[1] & 0xF0) | ((FEC_CODES[fec] & 0x07) << 1) | (1 if interleave else 0)
            for length in range(0, _BLIND_MAX_BYTES + 1):
                candidate = bytes([body[0], byte1, (length >> 8) & 0xFF, length & 0xFF])
                if crc8(candidate) == received:
                    out.append({"fec": fec, "interleave": bool(interleave), "payload_len": int(length)})
    # 帧头被改坏时 CRC-8 会给少量假命中；按"声明优先、长度小优先"排序并截断
    out.sort(key=lambda item: (item["fec"] != "none", item["payload_len"]))
    return out[: int(limit)]


def _erasures_from_bursts(
    burst_ranges: list[tuple[int, int]],
    cursor: int,
    bps: int,
    unit_bits: int = 8,
) -> list[int]:
    """把"信号侧的突发符号区间"映射成"编码域里的 RS 符号下标".

    坐标换算：帧内第 b 个比特对应符号 (cursor + b) / bps（cursor 是该帧在候选比特流里的起点）。
    编码段从帧内 bit `PREAMBLE_BITS + HEADER_BITS` 开始，RS 的符号宽度是 8 比特。
    只保留落在编码段里的位置；越界一律丢弃（宁可少标，也不要把无关位置标成擦除）。
    """
    if not burst_ranges or bps <= 0:
        return []
    body_start = PREAMBLE_BITS + HEADER_BITS
    positions: set[int] = set()
    for start_symbol, end_symbol in burst_ranges:
        first_bit = int(start_symbol) * int(bps) - int(cursor) + body_start
        last_bit = int(end_symbol) * int(bps) - int(cursor) + body_start
        first_bit = max(first_bit, body_start)
        if last_bit <= body_start:
            continue
        first_symbol = (first_bit - body_start) // int(unit_bits)
        last_symbol = (max(last_bit, first_bit) - body_start + int(unit_bits) - 1) // int(unit_bits)
        positions.update(range(int(first_symbol), int(last_symbol)))
    return sorted(positions)


def _attempt_parse(
    bits: np.ndarray,
    cursor: int,
    llr: Optional[np.ndarray],
    blind: bool,
    stats: Optional[dict[str, Any]] = None,
    erasure_decoding: bool = False,
    burst_ranges: Optional[list[tuple[int, int]]] = None,
    bps: int = 2,
) -> dict[str, Any]:
    """在 cursor 处试解析一帧；blind=True 时对全部 (编码, 交织) 假设做检验.

    burst_ranges 非空且 erasure_decoding=True 时，用**信号侧检测到的突发区间**
    给出显式擦除位置（这比按 |LLR| 门限猜要可信得多，见 M12 §4）。

    帧头字段本身可能被打错（例如把 fec 打成 none），所以盲模式下按假设**重写帧头里的
    fec/interleave 位**再算 CRC —— 这是"假设 + 校验"的经典做法，而不是猜。
    """
    slice_llr = llr[cursor:] if (llr is not None and cursor < llr.size) else None
    explicit = _erasures_from_bursts(burst_ranges or [], cursor, int(bps)) if erasure_decoding else []
    try:
        parsed = parse_frame(bits[cursor:], llr=slice_llr, erasure_decoding=erasure_decoding,
                             erasures=explicit or None)
    except Exception as exc:  # noqa: BLE001
        parsed = {"ok": False, "reason": "{}: {}".format(type(exc).__name__, exc), "header": None,
                  "payload": b"", "crc_expected": None, "crc_actual": None, "crc_ok": False,
                  "preamble_ok": False, "bits_consumed": 0}
    if parsed["ok"] or not blind or parsed["header"] is None:
        return parsed
    declared = (str(parsed["header"].fec), bool(parsed["header"].interleave))
    header_ok = bool(parsed.get("header_crc_ok"))
    if stats is not None:
        if header_ok:
            stats["header_crc_ok"] = True
        elif stats.get("header_crc_ok") is None:
            stats["header_crc_ok"] = False
    if header_ok:
        # 帧头自校验通过：长度可信，只需要试 (编码, 交织)
        for fec, inter in _BLIND_HYPOTHESES:
            if (fec, inter) == declared:
                continue
            try:
                candidate = parse_frame(bits[cursor:], llr=slice_llr, force_fec=fec, force_interleave=inter,
                                        erasure_decoding=erasure_decoding)
            except Exception:  # noqa: BLE001
                continue
            if candidate["ok"]:
                return candidate
        return parsed
    # 帧头 CRC 不通过：连**载荷长度**一起搜（按 CRC-8 筛出的假设逐个试）
    specs = _blind_header_candidates(bits, cursor)
    if stats is not None:
        stats["hypotheses_tried"] = int(stats.get("hypotheses_tried", 0)) + len(specs)
        stats["header_crc_ok"] = False
    for spec in specs:
        try:
            candidate = parse_frame(
                bits[cursor:], llr=slice_llr,
                force_fec=spec["fec"], force_interleave=spec["interleave"],
                force_payload_len=spec["payload_len"],
            )
        except Exception:  # noqa: BLE001
            continue
        if candidate["ok"]:
            candidate["blind_recovered_len"] = int(spec["payload_len"])
            if stats is not None:
                stats["length_recovered"] = True
                stats["repaired_header"] = {
                    "payload_len": int(spec["payload_len"]),
                    "fec": spec["fec"],
                    "interleave": bool(spec["interleave"]),
                }
            return candidate
    return parsed


def _parse_sequence(
    bits: np.ndarray,
    start_bit: int,
    max_frames: int,
    resync_window: int,
    bps: int,
    llr: Optional[np.ndarray] = None,
    blind: bool = False,
    stats: Optional[dict[str, Any]] = None,
    erasure_decoding: bool = False,
    burst_ranges: Optional[list[tuple[int, int]]] = None,
) -> tuple[list[dict[str, Any]], int]:
    """从 start_bit 开始连续解析帧，返回 (帧列表, 失败次数).

    CRC 失败的帧也会被记录（如实标出 ok=false）；只有连前导/帧头都读不出来才算失败，
    并按预期帧长跳过继续 —— 不能让一帧坏掉就丢掉后面所有帧。
    """
    frames: list[dict[str, Any]] = []
    failures = 0
    cursor = int(start_bit)
    expected_bits: Optional[int] = None
    trusted_bits: Optional[int] = None      # 最近一次"帧头自校验通过"的帧长，作为可信步长
    guard = 0
    while len(frames) < int(max_frames) and guard < int(max_frames) * 4:
        guard += 1
        if cursor < 0 or cursor + _MIN_FRAME_BITS > bits.size:
            break
        parsed = _attempt_parse(bits, cursor, llr, blind, stats, erasure_decoding, burst_ranges, bps)
        if not parsed["preamble_ok"] or parsed["header"] is None:
            recovered = None
            for delta in range(-int(resync_window), int(resync_window) + 1):
                if delta == 0 or cursor + delta < 0 or cursor + delta + _MIN_FRAME_BITS > bits.size:
                    continue
                candidate = _attempt_parse(bits, cursor + delta, llr, blind, stats, erasure_decoding, burst_ranges, bps)
                # bits_consumed > 0 才说明"帧结构完整且长度自洽"；只有前导命中、
                # 但帧头是垃圾值的候选不能当成一帧（否则后面按垃圾长度跳会越界/抛错）
                if candidate["preamble_ok"] and candidate["header"] is not None and int(candidate["bits_consumed"]) > 0:
                    recovered = (delta, candidate)
                    break
            if recovered is None:
                failures += 1
                cursor += int(expected_bits or _DEFAULT_FRAME_BITS)
                continue
            cursor += recovered[0]
            parsed = recovered[1]
        header = parsed["header"]
        frame_bits = int(parsed["bits_consumed"])
        if frame_bits <= 0:
            # 帧头是垃圾值时 frame_length_bits 会抛异常；这里必须降级成"这帧没解出来"，
            # 而不是让一个错误候选把整次提取打断
            try:
                frame_bits = frame_length_bits(header.payload_len, header.fec, header.interleave)
            except ValueError:
                failures += 1
                cursor += int(expected_bits or _DEFAULT_FRAME_BITS)
                continue
        # 帧头自校验通过时，这个帧长才可信（M9 起帧头带 CRC-8）。
        # 帧头被打坏、长度字段是垃圾时，**不能**按垃圾长度往下跳 —— 那样整条序列会被
        # 一帧带跑偏（16QAM 上实测：坏长度导致后面所有帧都找不到）。改用最近一次可信帧长。
        if parsed.get("header_crc_ok") is not None and bool(parsed.get("header_crc_ok")):
            trusted_bits = frame_bits
        elif trusted_bits is not None and int(parsed.get("header_crc8") or 0) != int(parsed.get("crc8_expected") or -1):
            frame_bits = int(trusted_bits)
        expected_bits = frame_bits
        header_hex = np.packbits(
            bits[cursor + PREAMBLE_BITS : cursor + PREAMBLE_BITS + HEADER_BITS]
        ).tobytes().hex().upper()
        frames.append(
            {
                "index": len(frames),
                "start_bit": int(cursor),
                "start_symbol": float(cursor) / float(max(1, bps)),
                "preamble_ok": bool(parsed["preamble_ok"]),
                "header": {
                    "payload_len": int(header.payload_len),
                    "modulation": header.modulation,
                    "fec": header.fec,
                    "interleave": bool(header.interleave),
                    "header_hex": header_hex,
                    # M9：帧头自带 CRC-8，这三项按契约放在 header 里（帧顶层保留同名字段做别名）
                    "header_crc_ok": parsed.get("header_crc_ok"),
                    "header_crc8": "0x{:02X}".format(int(parsed.get("header_crc8") or 0)),
                    "crc8_expected": "0x{:02X}".format(int(parsed.get("crc8_expected") or 0)),
                },
                "crc": {
                    "expected": "0x{:04X}".format(int(parsed["crc_expected"] or 0)),
                    "computed": "0x{:04X}".format(int(parsed["crc_actual"] or 0)),
                    "ok": bool(parsed["crc_ok"]),
                },
                "header_crc_ok": parsed.get("header_crc_ok"),
                "header_crc8": "0x{:02X}".format(int(parsed.get("header_crc8") or 0)),
                "crc8_expected": "0x{:02X}".format(int(parsed.get("crc8_expected") or 0)),
                "blind_recovered_len": parsed.get("blind_recovered_len"),
                "payload": _payload_preview(bytes(parsed["payload"]), 256),
                "fec": parsed.get("fec") or {},
                "reason": str(parsed["reason"]),
                "bit_errors_vs_truth": None,
            }
        )
        cursor += frame_bits
    return frames, failures


def _score(frames: list[dict[str, Any]]) -> tuple[int, int]:
    return sum(1 for f in frames if f["crc"]["ok"]), len(frames)


def extract_frames(
    signal: Signal,
    analysis: Optional[AnalyzeResult] = None,
    modulation: Optional[str] = None,
    modulation_source: str = "provided",
    truth: Optional[GroundTruth] = None,
    max_frames: int = 32,
    model_dir: str = "models",
    payload_preview: int = 256,
    demod: Optional[DemodResult] = None,
    candidate_limit: int = 40,
    blind_fec: bool = False,
    erasure_decoding: bool = False,
    auto_erasures: bool = False,
) -> FrameExtraction:
    """完整 M5 链路；truth 只用于最后对账，不参与同步."""
    warnings: list[str] = []
    burst_ranges: list[tuple[int, int]] = []

    blind_stats: Optional[dict[str, Any]] = None
    if blind_fec:
        blind_stats = {
            "enabled": True,
            "header_crc_ok": None,
            "length_recovered": False,
            "hypotheses_tried": 0,
            "repaired_header": None,
        }
    estimates = analysis or analyze_signal(signal)
    burst_ranges: list[tuple[int, int]] = []
    burst_sample_regions: list[tuple[int, int]] = []
    if auto_erasures:
        # 信号侧突发检测：**必须在解调之前**做 —— 检测结果要交给解调器做置零 + 环路冻结，
        # 否则强干扰会把定时/载波环一起拽走，损伤从"一段"扩散成"整帧"（M12-C 实测）。
        try:
            from spxh.core.dsp.burst import detect_bursts

            regions = detect_bursts(
                np.asarray(signal.samples, dtype=np.complex128).reshape(-1),
                sample_rate=float(signal.sample_rate),
                sps=float(signal.sample_rate) / max(float(estimates.symbol_rate.value or 0.0), 1e-9),
            )
            burst_sample_regions = [(int(r.start_sample), int(r.end_sample)) for r in regions]
            if regions:
                warnings.append("检测到 {} 段突发损伤（最深处 {:.1f} dB）：已置零并在突发期间冻结同步环".format(
                    len(regions), min(region.depth_db for region in regions)))
        except Exception as exc:  # noqa: BLE001 - 检测失败不该影响正常解帧
            warnings.append("突发检测失败（{}），本次不启用抗突发".format(type(exc).__name__))

    result = demod or demodulate(
        signal,
        analysis=estimates,
        burst_regions=burst_sample_regions or None,
        blank_bursts=True,
        modulation=modulation,
        modulation_source=modulation_source,
        truth=None,
        model_dir=model_dir,
    )

    # 突发：把"采样区间"映射成"符号区间"（用定时环给出的每符号采样位置）
    if burst_sample_regions:
        positions = np.asarray(result.symbol_samples, dtype=np.float64).reshape(-1)
        for start_sample, end_sample in burst_sample_regions:
            inside = np.flatnonzero((positions >= start_sample) & (positions < end_sample))
            if inside.size:
                burst_ranges.append((int(inside[0]), int(inside[-1]) + 1))

    mod = mapping.normalize_modulation(result.modulation)
    bps = int(MOD_BITS_PER_SYMBOL[mod])
    symbol_rate = float(result.symbol_rate)
    resync_window = max(1, int(round(0.02 * symbol_rate / max(symbol_rate, 1e-9)))) if False else 4

    fsk = mod in ("2fsk", "4fsk")
    if fsk:
        # FSK 走非相干音调检测，M3 里**没有定时环**（恒模信号本来也不需要相位跟踪）。
        # 但符号边界一旦偏了（哪怕 0.3 个符号），相邻音调的相关窗就会骑到边界上、
        # 造成几个比特出错 —— 实测正是"4 帧里恰好 1 帧 CRC 不过"。这里补一个
        # 采样相位搜索：按 ±几个采样重解调，取解析得分最高者。
        def _fsk_score(samples: np.ndarray) -> Optional[dict[str, Any]]:
            shifted = Signal(samples=samples, sample_rate=signal.sample_rate)
            sub = demodulate(
                shifted,
                analysis=estimates,
                modulation=mod,
                modulation_source=modulation_source,
                truth=None,
                model_dir=model_dir,
            )
            sub_bits = np.asarray(sub.bits, dtype=np.uint8).reshape(-1)
            corr_local = correlate_bits(sub_bits)
            if corr_local.size == 0:
                return None
            mag_local = np.abs(corr_local)
            local_best: Optional[dict[str, Any]] = None
            for candidate in _candidate_peaks(mag_local, spacing=BARKER_LEN, limit=candidate_limit,
                                             floor_ratio=0.2, min_abs=0.2):
                sub_llr = np.asarray(sub.llr, dtype=np.float64).reshape(-1) if getattr(sub, "llr", None) is not None else None
                frames, failures = _parse_sequence(
                    sub_bits, candidate, max_frames, resync_window, bps, llr=sub_llr, blind=blind_fec,
                    stats=blind_stats, erasure_decoding=erasure_decoding, burst_ranges=burst_ranges,
                )
                entry = {
                    "start_bit": int(candidate),
                    "start_symbol": float(candidate) / float(bps),
                    "peak": float(mag_local[candidate]),
                    "angle_deg": 0.0,
                    "frames": frames,
                    "failures": failures,
                    "score": _score(frames),
                    "magnitude": mag_local,
                }
                if local_best is None or entry["score"] > local_best["score"]:
                    local_best = entry
                if local_best["score"][0] >= int(max_frames):
                    break
            return local_best

        samples0 = np.asarray(signal.samples, dtype=np.complex128).reshape(-1)
        best = _fsk_score(samples0)
        if best is None or best["score"][0] < int(max_frames):
            # 采样相位搜索：整体平移若干采样再试（FSK 对定时最敏感的是这个自由度）
            for shift in range(1, max(2, int(round(0.5 * result.evidence.get("sps", 8) if isinstance(result.evidence, dict) else 4))) + 1):
                for direction in (1, -1):
                    shifted = np.roll(samples0, direction * shift)
                    candidate = _fsk_score(shifted)
                    if candidate is None:
                        continue
                    if best is None or candidate["score"] > best["score"]:
                        candidate["shift_samples"] = int(direction * shift)
                        best = candidate
                    if best is not None and best["score"][0] >= int(max_frames):
                        break
                if best is not None and best["score"][0] >= int(max_frames):
                    break
        if best is None:
            best = {
                "start_bit": 0, "start_symbol": 0.0, "peak": 0.0, "angle_deg": 0.0,
                "frames": [], "failures": 0, "score": (0, 0),
                "magnitude": np.zeros(max(1, int(result.symbols.size)), dtype=np.float64),
            }
        magnitude = np.asarray(best.get("magnitude", np.zeros(1)), dtype=np.float64)
    else:
        # 检测模板用 13 位 Barker（M9 试过"Barker + MAGIC + 调制码"的更长模板，
        # 在 QPSK 6~10 dB 上 A/B 实测与只用 Barker **完全相同**，且 16QAM 的
        # 相位歧义解析出现退化 —— 说明瓶颈在帧头保护而不是检测统计量，故回退。）
        pattern, count = preamble_symbols(mod)
        symbols = np.asarray(result.symbols, dtype=np.complex128).reshape(-1)
        corr = correlate_symbols(symbols, pattern)
        magnitude = np.abs(corr)
        candidates = _candidate_peaks(magnitude, spacing=count, limit=candidate_limit)

        order = int(_SYMMETRY.get(mod, 1))
        angles = [2.0 * np.pi * k / float(order) for k in range(max(1, order))]
        candidate_best: Optional[dict[str, Any]] = None
        for candidate in candidates:
            for angle in angles:
                rotated = symbols * np.exp(-1j * angle)
                indices = mapping.points_to_indices(rotated, mod)
                candidate_bits = mapping.indices_to_bits(indices, mod).reshape(-1)
                # 旋转后的符号重新算 LLR：软判决译码(Viterbi/LDPC)直接吃它，
                # 这样 M3 标定过的软信息一路贯到 M6 的译码器，不需要额外校准。
                candidate_llr, _, _ = max_log_llr(rotated, mod, symbol_noise_variance(float(result.snr_db)))
                frames, failures = _parse_sequence(
                    candidate_bits, candidate * bps, max_frames, resync_window, bps,
                    llr=candidate_llr, blind=blind_fec, stats=blind_stats,
                    erasure_decoding=erasure_decoding, burst_ranges=burst_ranges,
                )
                entry = {
                    "start_bit": int(candidate * bps),
                    "start_symbol": float(candidate),
                    "peak": float(magnitude[candidate]),
                    "angle_deg": float(np.degrees(angle)),
                    "frames": frames,
                    "failures": failures,
                    "score": _score(frames),
                }
                if candidate_best is None or entry["score"] > candidate_best["score"]:
                    candidate_best = entry
                if candidate_best["score"][0] >= int(max_frames):
                    break
            if candidate_best is not None and candidate_best["score"][0] >= int(max_frames):
                break
        if candidate_best is None:
            candidate_best = {
                "start_bit": 0, "start_symbol": 0.0, "peak": 0.0, "angle_deg": 0.0,
                "frames": [], "failures": 0, "score": (0, 0),
            }
        best = candidate_best

    frames = list(best["frames"])
    if not frames:
        warnings.append("没有解析出任何帧：同步失败或记录太短")

    peak = float(best.get("peak", 0.0))
    # 峰显著性用"峰值 / 全序列中位数"的 dB 比，而不是"峰值 / 最大旁瓣"：
    # Barker 的低旁瓣只对它自己的非周期自相关成立；埋在**随机载荷**里时，
    # 别处的随机数据经常凑出比真峰还大的相关值（实测 QPSK 出现 -0.17 dB 这种负数，
    # 说明"最大旁瓣"口径在这个场景下没有意义）。中位数是稳健参考，且必然 <= 峰值，
    # 所以这个指标天然 >= 0 dB，越大越说明该峰在整条相关序列里越突出。
    median = float(np.median(magnitude)) if magnitude.size else 0.0
    psr_db = float(20.0 * np.log10(max(peak, 1e-9) / max(median, 1e-9)))
    peak_percentile = float(np.mean(magnitude <= peak)) if magnitude.size else 0.0

    summary: dict[str, Any] = {
        "frames": len(frames),
        "frames_crc_ok": int(sum(1 for f in frames if f["crc"]["ok"])),
        "payload_bytes_total": int(sum(f["payload"]["bytes"] for f in frames)),
        "payload_bytes_exact": None,
        "candidates_failed": int(best.get("failures", 0)),
        "all_ok": False,
    }
    if truth is not None and getattr(truth, "frames", None) and frames:
        first = _hex_to_bytes(frames[0]["payload"]["hex"])
        offset = 0
        if first:
            for i, truth_frame in enumerate(truth.frames):
                if bytes(truth_frame.payload).startswith(first[: min(8, len(first))]):
                    offset = i
                    break
        exact = 0
        for i, frame in enumerate(frames):
            truth_index = offset + i
            if truth_index >= len(truth.frames):
                break
            expected = bytes(truth.frames[truth_index].payload)
            got = _hex_to_bytes(frame["payload"]["hex"])
            compare = min(len(expected), len(got))
            errors = int(np.sum(np.frombuffer(expected[:compare], dtype=np.uint8) != np.frombuffer(got[:compare], dtype=np.uint8)))
            errors += abs(len(expected) - len(got)) * 8
            frame["bit_errors_vs_truth"] = int(errors)
            if errors == 0 and len(expected) == len(got):
                exact += len(expected)
        summary["payload_bytes_exact"] = int(exact)
        summary["truth_frame_offset"] = int(offset)
    summary["all_ok"] = bool(
        summary["frames"] > 0
        and summary["frames_crc_ok"] == summary["frames"]
        and (summary["payload_bytes_exact"] is None or summary["payload_bytes_exact"] == summary["payload_bytes_total"])
    )

    if peak < 0.25:
        warnings.append("相关峰偏低（{:.3f}）：可能没锁上".format(peak))
    if frames and summary["frames_crc_ok"] < summary["frames"]:
        warnings.append("有 {} 帧 CRC 未通过".format(summary["frames"] - summary["frames_crc_ok"]))

    return FrameExtraction(
        modulation=mod,
        modulation_source=result.modulation_source,
        symbol_rate=symbol_rate,
        cfo_hz=float(result.cfo_hz),
        preamble_bits="".join(str(int(b)) for b in barker_bits(BARKER_LEN)),
        correlation_peak=peak,
        peak_to_sidelobe_db=psr_db,
        frame_start_symbol=float(best["start_symbol"]),
        frame_start_sample=int(round(float(best["start_symbol"]) * signal.sample_rate / max(symbol_rate, 1e-9))),
        ambiguity_rotation_deg=float(best["angle_deg"]),
        locked=bool(peak >= 0.25 and summary["frames"] > 0),
        correlation_index=np.arange(magnitude.size, dtype=np.int64),
        correlation_value=magnitude,
        correlation_stride=1,
        frames=frames,
        summary=summary,
        warnings=warnings,
        blind=blind_stats or {},
        evidence={
            "receiver": "spxh-m5-framesync",
            "bits_per_symbol": bps,
            "sync_mode": "bit" if fsk else "symbol",
            "candidates_tried": int(candidate_limit),
            "burst_ranges": [[int(a), int(b)] for a, b in (burst_ranges or [])],
            "auto_erasures": bool(auto_erasures),
            "burst_blanking": bool(burst_sample_regions),
            "peak_to_median_db": psr_db,
            "peak_percentile": peak_percentile,
            "locked_rule": "correlation_peak >= 0.25 且至少解析出 1 帧",
            "demod_lock": {
                "timing_metric": float(result.timing_metric),
                "carrier_metric": float(result.carrier_metric),
                "frozen_symbols": int((result.evidence or {}).get("frozen_symbols") or 0),
                "blanked_samples": int((result.evidence or {}).get("blanked_samples") or 0),
            },
        },
    )


def _hex_to_bytes(text: str) -> bytes:
    """把帧载荷预览里的十六进制字符串还原成字节（真值对账用）."""
    cleaned = "".join(ch for ch in str(text) if ch in "0123456789abcdefABCDEF")
    if len(cleaned) % 2:
        cleaned = cleaned[:-1]
    try:
        return bytes.fromhex(cleaned)
    except ValueError:
        return b""
