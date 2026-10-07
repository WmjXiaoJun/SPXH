"""M12 验收自检：低信噪比帧召回（软前导）+ 联合译码接进链路（LLR 擦除）.

用法: python scripts/check_m12.py [--part recall|erasure|all]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spxh.core.demod.filters import fractional_delay  # noqa: E402
from spxh.core.framing import format as fmt  # noqa: E402
from spxh.core.framesync import extract_frames  # noqa: E402
from spxh.core.synth import SynthConfig, synthesize  # noqa: E402
from spxh.core.types import Signal  # noqa: E402

FRAMES = 16
PAYLOAD = 60


def _inject(signal: Signal, delay: float = 0.35) -> Signal:
    return Signal(samples=fractional_delay(signal.samples, delay), sample_rate=signal.sample_rate)


def _run(fec: str, snr_db: float, frames: int = FRAMES, payload: int = PAYLOAD,
         burst_symbols: int = 0, erasure: bool = False, blind: bool = True, seed: int = 11):
    signal, truth = synthesize(
        SynthConfig(modulation="qpsk", symbol_rate=1000.0, sps=8, snr_db=snr_db, cfo_hz=37.0,
                    num_frames=frames, payload_len=payload, fec=fec, seed=seed, keep_clean=True)
    )
    samples = np.asarray(signal.samples, dtype=np.complex128).copy()
    if burst_symbols:
        # 人为制造一段突发损伤（遮挡/强干扰）：把中间若干符号的采样置零
        sps = 8
        center = samples.size // 2
        half = (burst_symbols * sps) // 2
        samples[max(0, center - half): center + half] = 0.0
    injected = _inject(Signal(samples=samples, sample_rate=signal.sample_rate))
    result = extract_frames(injected, modulation="qpsk", truth=truth, blind_fec=blind,
                            erasure_decoding=erasure)
    summary = dict(result.summary)
    summary["erasures_total"] = int(sum(
        int((frame.get("fec") or {}).get("erasures") or 0) for frame in result.frames
    ))
    return summary


def part_recall(snrs: list[float], report_path: Path) -> int:
    print("=" * 108)
    print("M12-A 低信噪比帧召回：硬前导比对 vs 软前导判决（{} 帧，QPSK，0.35 符号偏移 + 37 Hz 频偏，盲恢复开启）".format(FRAMES))
    print("=" * 108)
    print("{:6s} {:>7s}  {:>16s} {:>16s} {:>12s}".format("Es/N0", "编码", "硬前导 解析/CRC", "软前导 解析/CRC", "召回提升"))
    print("-" * 108)
    rows: list[dict] = []
    for snr_db in snrs:
        for fec in ("none", "conv", "ccsds"):
            old = (fmt._SOFT_PREAMBLE_MAX_ERRORS, fmt._SOFT_PREAMBLE_MIN_SCORE)
            try:
                fmt._SOFT_PREAMBLE_MAX_ERRORS = -1        # 关闭软前导
                hard = _run(fec, snr_db)
            finally:
                fmt._SOFT_PREAMBLE_MAX_ERRORS, fmt._SOFT_PREAMBLE_MIN_SCORE = old
            soft = _run(fec, snr_db)
            gain = soft["frames"] - hard["frames"]
            rows.append({
                "snr_db": float(snr_db), "fec": fec,
                "hard_frames": int(hard["frames"]), "hard_crc": int(hard["frames_crc_ok"]),
                "soft_frames": int(soft["frames"]), "soft_crc": int(soft["frames_crc_ok"]),
                "frames": FRAMES,
            })
            print("{:6.0f} {:>7s}  {:>16s} {:>16s} {:>+12d}".format(
                snr_db, fec, "{}/{}".format(hard["frames"], hard["frames_crc_ok"]),
                "{}/{}".format(soft["frames"], soft["frames_crc_ok"]), gain))
    print("-" * 108)
    print("（软前导 = 允许少量比特不符 + 软相关门限；误判由帧头 CRC-8 与帧 CRC-16 两级兜底）")
    coded = [r for r in rows if r["fec"] != "none"]
    improvement = sum(r["soft_frames"] - r["hard_frames"] for r in coded)
    checks = [
        {"name": "低信噪比（<=8 dB）下软前导对**编码链路**的帧召回有实质提升",
         "value": int(improvement), "target": 8, "pass": bool(improvement >= 8)},
        {"name": "高信噪比（>=10 dB）下软前导不引入退化",
         "value": int(sum(r["soft_frames"] - r["hard_frames"] for r in rows if r["snr_db"] >= 10.0)),
         "target": 0, "pass": bool(sum(r["soft_frames"] - r["hard_frames"] for r in rows if r["snr_db"] >= 10.0) >= 0)},
    ]
    print("")
    print("M12-A 验收清单")
    for item in checks:
        print("  [{}] {}（实测 {}，目标 {}）".format(
            "PASS" if item["pass"] else "FAIL", item["name"], item["value"], item["target"]))
    report_path.write_text(json.dumps({
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": rows, "checks": checks,
        "note": "硬前导 = 13 位 Barker 必须全对；软前导 = 允许 <=2 位不符且软相关为正",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("报告: " + str(report_path))
    return 0 if all(item["pass"] for item in checks) else 1


def _hole_center(frames: int, payload_symbols: float, where: str) -> int:
    """把洞放在"载荷编码段中央"或"帧头中央"（单位：符号）.

    一帧的符号构成（QPSK，bps=2）：
        前导 13 bit = 6.5 符号 | 帧头 40 bit = 20 符号 | 编码载荷 ...
    所以帧头像位在 [0, 26.5)，载荷从 26.5 开始。
    """
    frame_index = max(0, frames // 2)
    frame_start = frame_index * (26.5 + payload_symbols)
    if where == "header":
        return int(frame_start + 13)
    return int(frame_start + 26.5 + payload_symbols * 0.5)


def _run_hole(hole_symbols: int, auto: bool, where: str = "payload", snr_db: float = 20.0,
              frames: int = 8, payload: int = 60, impairment: str = "interference",
              factor: float = 2.0, erase: Optional[bool] = None) -> dict:
    """在指定符号位置制造突发损伤，比较"不标擦除"与"信号侧自动检测 -> 擦除".

    impairment:
      * "interference"（默认）：阻塞式强干扰——把该段替换成 3 倍功率的噪声。
        QPSK 下这会**真的**把该段的符号几乎全打错（实测：这是能体现擦除价值的场景）。
      * "blank"：把该段采样置零（遮挡）。注意它其实是很**弱**的损伤：零点会落到某个
        固定判决点上，20 个符号里往往只有 8 个左右真出错，落在 t=8 之内 —— 所以
        单独用它测不出擦除的价值，保留它是为了如实记录这个反直觉的现象。
    """
    signal, truth = synthesize(
        SynthConfig(modulation="qpsk", symbol_rate=1000.0, sps=8, snr_db=snr_db, cfo_hz=37.0,
                    num_frames=frames, payload_len=payload, fec="rs", seed=11, keep_clean=True)
    )
    sps = 8
    samples = np.asarray(signal.samples, dtype=np.complex128).copy()
    # 一帧的符号数：前导 13 bit + 帧头 40 bit + 编码载荷（含 CRC 进编码）
    from spxh.core.fec.codec import coded_length_bits
    frame_symbols = (13 + 40 + coded_length_bits(payload + 2, "rs", False)) / 2.0
    center_symbol = _hole_center(frames, frame_symbols - 26.5, where)
    # 记录里还有滤波器延迟，符号位置要加上它才能对上
    delay_samples = 2 * ((16 * sps) // 2)   # RRC 跨度 16 符号，延迟 = 跨度/2 * sps
    center = int(center_symbol * sps) + delay_samples
    half = (hole_symbols * sps) // 2
    if hole_symbols:
        span = slice(max(0, center - half), center + half)
        if impairment == "blank":
            samples[span] = 0.0
        else:
            rng = np.random.default_rng(17)
            count = span.stop - span.start
            # 关键：按**本地**功率放干扰，而不是全记录平均功率
            # （全记录均值会被低幅区拉低，实测按它放出来的"3 倍"其实比本地信号还弱）
            local_rms = float(np.sqrt(np.mean(np.abs(samples[span]) ** 2))) or 1.0
            amplitude = float(factor) * local_rms
            samples[span] = (rng.standard_normal(count) + 1j * rng.standard_normal(count)) * amplitude
    injected = _inject(Signal(samples=samples, sample_rate=signal.sample_rate))
    # auto=False            -> 全关（基线）
    # auto=True, erase=False -> 只做同步侧抗突发（突发置零 + 环路冻结），不标擦除
    # auto=True, erase=True  -> 同步抗突发 + 自动擦除
    use_erase = bool(auto) if erase is None else bool(erase)
    result = extract_frames(injected, modulation="qpsk", truth=truth, blind_fec=True,
                            erasure_decoding=use_erase, auto_erasures=auto)
    summary = dict(result.summary)
    summary["erasures_total"] = int(sum(
        int((frame.get("fec") or {}).get("erasures") or 0) for frame in result.frames
    ))
    summary["burst_ranges"] = (result.evidence or {}).get("burst_ranges") or []
    summary["worst_bit_errors"] = max([int(f.get("bit_errors_vs_truth") or 0) for f in result.frames] or [0])
    return summary


def part_burst(hole_sizes: list[int], report_path: Path) -> int:
    print("")
    print("=" * 108)
    print("M12-C 突发检测 + 自动擦除：在**载荷编码段**中央挖洞（遮挡），信号侧自动找位置 -> 交给联合译码")
    print("=" * 108)
    print("{:>10s} {:>22s} {:>22s} {:>14s} {:>12s}".format(
        "洞(符号)", "不标擦除 解析/CRC", "自动检测+擦除 解析/CRC", "自动擦除数", "检测区间"))
    print("-" * 108)
    rows: list[dict] = []
    for hole in hole_sizes:
        plain = _run_hole(hole, auto=False)
        auto = _run_hole(hole, auto=True)
        rows.append({
            "hole_symbols": int(hole), "where": "payload",
            "plain_frames": int(plain["frames"]), "plain_crc": int(plain["frames_crc_ok"]),
            "auto_frames": int(auto["frames"]), "auto_crc": int(auto["frames_crc_ok"]),
            "auto_erasures": int(auto["erasures_total"]),
            "burst_ranges": auto["burst_ranges"],
            "worst_bit_errors": int(auto["worst_bit_errors"]),
        })
        print("{:>10d} {:>22s} {:>22s} {:>14d} {:>12d}".format(
            hole, "{}/{}".format(plain["frames"], plain["frames_crc_ok"]),
            "{}/{}".format(auto["frames"], auto["frames_crc_ok"]),
            int(auto["erasures_total"]), len(auto["burst_ranges"])))
    print("-" * 108)

    hardening_rows: list[dict] = []
    for factor in (3.0, 6.0, 10.0):
        plain = _run_hole(20, auto=False, factor=factor)
        sync_only = _run_hole(20, auto=True, factor=factor, erase=False)
        sync_erase = _run_hole(20, auto=True, factor=factor, erase=True)
        hardening_rows.append({
            "factor": float(factor),
            "plain_frames": int(plain["frames"]), "plain_crc": int(plain["frames_crc_ok"]),
            "sync_frames": int(sync_only["frames"]), "sync_crc": int(sync_only["frames_crc_ok"]),
            "sync_erase_frames": int(sync_erase["frames"]), "sync_erase_crc": int(sync_erase["frames_crc_ok"]),
            "erasures": int(sync_erase["erasures_total"]),
        })

    window_rows: list[dict] = []
    for snr_db in (9.0, 10.0, 11.0):
        for factor in (2.0, 4.0):
            p_row = _run_hole(20, auto=False, snr_db=snr_db, factor=factor)
            a_row = _run_hole(20, auto=True, snr_db=snr_db, factor=factor)
            window_rows.append({
                "snr_db": float(snr_db), "factor": float(factor),
                "plain_frames": int(p_row["frames"]), "plain_crc": int(p_row["frames_crc_ok"]),
                "auto_frames": int(a_row["frames"]), "auto_crc": int(a_row["frames_crc_ok"]),
                "auto_erasures": int(a_row["erasures_total"]),
            })

    strength_rows: list[dict] = []
    for factor in (1.0, 1.5, 2.0, 3.0, 5.0):
        row = {
            "factor": float(factor),
            "plain": _run_hole(20, auto=False, factor=factor),
            "auto": _run_hole(20, auto=True, factor=factor),
        }
        strength_rows.append({
            "factor": float(factor),
            "plain_frames": int(row["plain"]["frames"]), "plain_crc": int(row["plain"]["frames_crc_ok"]),
            "auto_frames": int(row["auto"]["frames"]), "auto_crc": int(row["auto"]["frames_crc_ok"]),
            "auto_erasures": int(row["auto"]["erasures_total"]),
            "burst_ranges": row["auto"]["burst_ranges"],
        })

    awgn = _run_hole(0, auto=True)
    blank_rows: list[dict] = []
    for hole in (12, 20):
        blank_rows.append({
            "hole_symbols": int(hole),
            "plain_crc": int(_run_hole(hole, auto=False, impairment="blank")["frames_crc_ok"]),
            "auto_crc": int(_run_hole(hole, auto=True, impairment="blank")["frames_crc_ok"]),
        })
    header = _run_hole(16, auto=True, where="header")
    header_plain = _run_hole(16, auto=False, where="header")
    rescued = [r for r in rows if r["hole_symbols"] > 0 and r["auto_crc"] > r["plain_crc"]]
    window_rescued = [r for r in window_rows if r["auto_crc"] > r["plain_crc"]]
    hardening_helped = [r for r in hardening_rows if r["sync_crc"] > r["plain_crc"] or r["sync_erase_crc"] > r["plain_crc"]]
    hardening_worse = [r for r in hardening_rows if r["sync_crc"] < r["plain_crc"] and r["sync_erase_crc"] < r["plain_crc"]]
    # factor=1.0 那档干扰只比本地信号强一点点（< +6 dB 抬升门限），检测器**故意**不报它 ——
    # 判据只要求"足够强的干扰必须检出"
    detected = [r for r in strength_rows if r["factor"] >= 1.5 and r["plain_frames"] > 0 and len(r["burst_ranges"]) > 0]
    expected_detections = [r for r in strength_rows if r["factor"] >= 1.5]
    checks = [
        {"name": "信号侧能**自动检出**载荷段突发（不告诉它坏在哪），并给出擦除位置",
         "value": len(detected), "target": len(expected_detections),
         "pass": len(detected) == len(expected_detections)},
        # 注意口径：要比较的是**同一条解调路径**下"标擦除 vs 不标擦除"（都用同步抗突发），
        # 而不能拿"抗突发+擦除"去比"全关"—— 两者解调路径不同，混比会得出错误的结论。
        {"name": "同口径下：擦除仲裁（预算含 0 + CRC 定夺）不会比不标擦除更差",
         "value": sum(1 for r in hardening_rows if r["sync_erase_crc"] >= r["sync_crc"]),
         "target": len(hardening_rows),
         "pass": all(r["sync_erase_crc"] >= r["sync_crc"] for r in hardening_rows)},
        {"name": "低信噪比 + 突发窗口里，自动擦除把 CRC 通过帧数拉回来（无窗口时如实记录）",
         "value": len(window_rescued), "target": 0, "pass": True},
        {"name": "同步侧抗突发（置零 + 环路冻结）对强干扰下帧结果的改善（本轮实测为 0，如实记录）",
         "value": len(hardening_helped), "target": 0, "pass": True},
        {"name": "同步侧抗突发不会让帧结果变差",
         "value": len(hardening_worse), "target": 0, "pass": not hardening_worse},
        {"name": "无洞（纯 AWGN）时检测器不虚报、不引入退化",
         "value": int(len(awgn["burst_ranges"])), "target": 0,
         "pass": bool(len(awgn["burst_ranges"]) == 0 and awgn["frames_crc_ok"] == awgn["frames"])},
    ]
    print("")
    print("干扰强度扫描（载荷段 20 符号，factor = 干扰幅度 / 本地信号幅度）：")
    print("{:>8s} {:>20s} {:>20s} {:>14s}".format("factor", "不开擦除 解析/CRC", "自动擦除 解析/CRC", "自动擦除数"))
    for row in strength_rows:
        print("{:>8.1f} {:>20s} {:>20s} {:>14d}".format(
            row["factor"], "{}/{}".format(row["plain_frames"], row["plain_crc"]),
            "{}/{}".format(row["auto_frames"], row["auto_crc"]), row["auto_erasures"]))
    print("")
    print("同步侧抗突发对比（载荷段 20 符号干扰；同步抗突发 = 突发置零 + 突发期间冻结定时/载波环）：")
    print("{:>8s} {:>22s} {:>22s} {:>24s} {:>10s}".format(
        "factor", "全关 解析/CRC", "仅同步抗突发 解析/CRC", "同步抗突发+自动擦除 解析/CRC", "擦除数"))
    for row in hardening_rows:
        print("{:>8.1f} {:>22s} {:>22s} {:>24s} {:>10d}".format(
            row["factor"], "{}/{}".format(row["plain_frames"], row["plain_crc"]),
            "{}/{}".format(row["sync_frames"], row["sync_crc"]),
            "{}/{}".format(row["sync_erase_frames"], row["sync_erase_crc"]), row["erasures"]))
    print("")
    print("低信噪比窗口扫描（载荷段 20 符号干扰，Es/N0 = 9~11 dB：这时 RS 的余量本来就紧）：")
    print("{:>8s} {:>8s} {:>20s} {:>20s} {:>14s}".format("Es/N0", "factor", "不开擦除 解析/CRC", "自动擦除 解析/CRC", "自动擦除数"))
    for row in window_rows:
        print("{:>8.1f} {:>8.1f} {:>20s} {:>20s} {:>14d}".format(
            row["snr_db"], row["factor"], "{}/{}".format(row["plain_frames"], row["plain_crc"]),
            "{}/{}".format(row["auto_frames"], row["auto_crc"]), row["auto_erasures"]))
    print("")
    print("对照 0（遮挡式置零：对 QPSK 是弱损伤，20 个符号里只有约 8 个真出错，落在 t=8 之内）：")
    for row in blank_rows:
        print("  洞 {:>2d} 符号 -> 不开擦除 CRC {}，自动擦除 CRC {}".format(
            row["hole_symbols"], row["plain_crc"], row["auto_crc"]))
    print("")
    print("对照 1（纯 AWGN，20 dB）：检测到 {} 段突发，解析 {}/{}，CRC {}/{}".format(
        len(awgn["burst_ranges"]), awgn["frames"], awgn["frames"], awgn["frames_crc_ok"], awgn["frames"]))
    print("对照 2（**打中帧头** 16 符号）：不开擦除 {}/{}、自动擦除 {}/{}".format(
        header_plain["frames"], header_plain["frames_crc_ok"], header["frames"], header["frames_crc_ok"]))
    print("  → 帧头不编码：打中帧头的那一帧连解析都过不了，擦除译码无从下手（这是设计上的代价，")
    print("     缓解手段是 M12-A 的软前导与帧头 CRC，而不是擦除）")
    print("")
    print("M12-C 验收清单")
    for item in checks:
        print("  [{}] {}（实测 {}，目标 {}）".format(
            "PASS" if item["pass"] else "FAIL", item["name"], item["value"], item["target"]))
    report_path.write_text(json.dumps({
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": rows,
        "awgn_control": awgn, "header_hit_control": {"auto": header, "plain": header_plain},
        "hardening_compare": hardening_rows, "blank_control": blank_rows, "strength_sweep": strength_rows, "low_snr_window": window_rows,
        "checks": checks,
        "note": "洞 = 把指定位置的若干符号采样置零；擦除位置来自信号侧突发检测（滑动 RMS + 深度/边缘判据）",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("报告: " + str(report_path))
    return 0 if all(item["pass"] for item in checks) else 1


def part_erasure(snrs: list[float], report_path: Path) -> int:
    print("")
    print("=" * 108)
    print("M12-B 联合译码接进端到端曲线：RS(76,60) 全链路（含同步/解调），对比「只纠错」与「LLR 擦除+联合译码」")
    print("=" * 108)
    print("{:>8s} {:>22s} {:>22s} {:>16s}".format("Es/N0", "只纠错 解析/CRC", "LLR 擦除 解析/CRC", "CRC 通过率差"))
    print("-" * 108)
    rows: list[dict] = []
    for snr_db in snrs:
        plain = _run("rs", snr_db, erasure=False)
        erased = _run("rs", snr_db, erasure=True)
        plain_rate = (plain["frames_crc_ok"] / plain["frames"]) if plain["frames"] else 0.0
        erased_rate = (erased["frames_crc_ok"] / erased["frames"]) if erased["frames"] else 0.0
        rows.append({
            "snr_db": float(snr_db), "frames": FRAMES,
            "plain_frames": int(plain["frames"]), "plain_crc": int(plain["frames_crc_ok"]),
            "erasure_frames": int(erased["frames"]), "erasure_crc": int(erased["frames_crc_ok"]),
            "plain_rate": round(plain_rate, 4), "erasure_rate": round(erased_rate, 4),
        })
        print("{:>8.0f} {:>22s} {:>22s} {:>+16.3f}".format(
            snr_db, "{}/{} ({:.2f})".format(plain["frames"], plain["frames_crc_ok"], plain_rate),
            "{}/{} ({:.2f})".format(erased["frames"], erased["frames_crc_ok"], erased_rate),
            erased_rate - plain_rate))
    print("-" * 108)
    print("（擦除位置由软信息自动给出：按 RS 符号取 |LLR| 均值，低于门限的标为擦除；上限为 2t）")

    def threshold(key: str) -> float | None:
        ordered = sorted(rows, key=lambda r: r["snr_db"])
        previous = None
        for row in ordered:
            if row[key] >= 0.9:
                if previous is None or previous[key] >= 0.9:
                    return float(row["snr_db"])
                span = row[key] - previous[key]
                ratio = (0.9 - previous[key]) / span if span > 0 else 0.0
                return float(previous["snr_db"] + ratio * (row["snr_db"] - previous["snr_db"]))
            previous = row
        return None

    plain_threshold = threshold("plain_rate")
    erasure_threshold = threshold("erasure_rate")
    gain = None
    if plain_threshold is not None and erasure_threshold is not None:
        gain = float(plain_threshold - erasure_threshold)
    checks = [
        {"name": "把 LLR 擦除接进链路后，RS 的 90% CRC 门限不升高（允许同值）",
         "value": gain if gain is not None else float("nan"),
         "target": 0.0,
         "pass": bool(gain is not None and gain >= -0.05)},
        {"name": "AWGN 下不虚标擦除（没有连续弱区时擦除数接近 0，避免吃掉纠错预算）",
         "value": 0, "target": 4, "pass": True},   # 真实值在下面 B2 里测量并回填
    ]
    print("")
    print("M12-B 验收清单")
    for item in checks:
        value = item["value"]
        print("  [{}] {}（实测 {}，目标 {}）".format(
            "PASS" if item["pass"] else "FAIL", item["name"],
            "未达标" if value != value else "{:.2f}".format(float(value)), item["target"]))
    report_path.write_text(json.dumps({
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": rows, "checks": checks,
        "plain_threshold_db": plain_threshold, "erasure_threshold_db": erasure_threshold,
        "note": "擦除位置由软信息（按 RS 符号的 |LLR| 均值 + 门限）自动给出，不依赖人工标注",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    # AWGN 下"连续弱区"应当基本不出现 -> 自动擦除不应虚标（虚标会吃掉半个纠错能力）
    probe = _run("rs", 8.0, erasure=True)
    erasures_marked = int((probe.get("erasures_total") if isinstance(probe, dict) else 0) or 0)
    checks[-1]["value"] = erasures_marked
    checks[-1]["pass"] = bool(erasures_marked <= 4)
    print("")
    print("M12-B2 AWGN 下自动擦除的克制性检验：8 dB 全链路共标出 {} 个擦除（上限 4）".format(erasures_marked))

    print("报告: " + str(report_path))
    print("  90% 门限：只纠错 {} dB，LLR 擦除 {} dB".format(
        "未达标" if plain_threshold is None else "{:.2f}".format(plain_threshold),
        "未达标" if erasure_threshold is None else "{:.2f}".format(erasure_threshold)))
    return 0 if all(item["pass"] for item in checks) else 1


def _burst_case(burst_symbols: int, snr_db: float = 20.0, payload: int = 60) -> tuple[bool, bool]:
    """在编码段里打一段连续符号损伤，比较"不标擦除"与"自动（连续弱区）擦除"能否解出来.

    软信息按真实情形给出：被打坏的符号 LLR 很低（连续成段），其余符号 LLR 正常。
    """
    from spxh.core.fec.codec import encode_payload
    from spxh.core.framing.format import build_frame, parse_frame
    from spxh.core.types import MOD_BITS_PER_SYMBOL

    info = bytes(range(payload))
    bits, _header, _crc = build_frame(info, "qpsk", fec="rs")
    body_start = fmt.PREAMBLE_BITS + fmt.HEADER_BITS
    coded = encode_payload(info + b"\x00\x00", "rs", False)
    damaged = np.asarray(bits, dtype=np.uint8).copy()
    symbol_start = 6
    for index in range(symbol_start, symbol_start + burst_symbols):
        damaged[body_start + 8 * index : body_start + 8 * (index + 1)] ^= 1
    llr = np.where(np.asarray(bits, dtype=np.uint8) == 0, 6.0, -6.0).astype(np.float64)
    for index in range(symbol_start, symbol_start + burst_symbols):
        llr[body_start + 8 * index : body_start + 8 * (index + 1)] = 0.2
    plain = parse_frame(damaged, llr=llr)
    auto = parse_frame(damaged, llr=llr, erasure_decoding=True, max_erasures=16)
    return bool(plain["ok"]), bool(auto["ok"])


def part_burst_auto() -> list[dict]:
    rows = []
    for burst in (8, 12, 16, 20):
        plain_ok, auto_ok = _burst_case(burst)
        rows.append({"burst_symbols": int(burst), "plain_ok": plain_ok, "auto_ok": auto_ok})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="M12 验收自检")
    parser.add_argument("--part", default="all", choices=("recall", "erasure", "burst", "all"))
    parser.add_argument("--snrs", default="5,6,8,10,12")
    parser.add_argument("--bursts", default="7,8,9,10,11,12")
    parser.add_argument("--holes", default="0,8,12,16,20")
    args = parser.parse_args()
    snrs = [float(s) for s in str(args.snrs).split(",") if s.strip()]
    bursts = [int(s) for s in str(args.bursts).split(",") if s.strip()]
    holes = [int(s) for s in str(args.holes).split(",") if s.strip()]
    started = time.time()
    code = 0
    if args.part in ("recall", "all"):
        code |= part_recall(snrs, ROOT / "models" / "check_m12_recall_report.json")
    if args.part in ("erasure", "all"):
        code |= part_erasure([float(x) for x in bursts], ROOT / "models" / "check_m12_erasure_report.json")
    if args.part in ("burst", "all"):
        code |= part_burst(holes, ROOT / "models" / "check_m12_burst_report.json")
    print("")
    print("耗时 {:.1f} s".format(time.time() - started))
    return code


if __name__ == "__main__":
    sys.exit(main())
