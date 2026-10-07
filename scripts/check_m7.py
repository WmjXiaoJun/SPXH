"""M7 验收自检：平滑端到端曲线 / 盲编码识别 / RS 擦除译码.

用法:
    python scripts/check_m7.py --part curves     # 每点跑更多帧，给出门限与曲线
    python scripts/check_m7.py --part blind      # 盲编码识别（帧头不可信时的假设检验）
    python scripts/check_m7.py --part erasure    # RS 擦除译码（纠错能力翻倍）
    python scripts/check_m7.py --part all
输出: 终端表格 + docs/figures/m7_*.png + models/check_m7_<part>_report.json
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

from spxh.console import ensure_utf8  # noqa: E402
ensure_utf8()

from spxh.core.demod.filters import fractional_delay  # noqa: E402
from spxh.core.fec.codec import FEC_MODES  # noqa: E402
from spxh.core.framing.format import HEADER_BITS, PREAMBLE_BITS, build_frame, parse_frame  # noqa: E402
from spxh.core.framesync import extract_frames  # noqa: E402
from spxh.core.framesync.extract import _BLIND_HYPOTHESES  # noqa: E402
from spxh.core.synth import SynthConfig, synthesize  # noqa: E402
from spxh.core.types import Signal  # noqa: E402

FIG_DIR = ROOT / "docs" / "figures"
CODE_LABEL = {
    "none": "无编码",
    "conv": "卷积(软判决)",
    "rs": "RS(76,60)",
    "ccsds": "CCSDS 级联",
    "ldpc": "LDPC(1152,578)",
}
PAYLOAD = 60
TARGET_CRC = 0.9


def _signal(fec: str, snr_db: float, frames: int, payload: int = PAYLOAD, seed: int = 11, interleave: bool = False):
    signal, truth = synthesize(
        SynthConfig(
            modulation="qpsk",
            symbol_rate=1000.0,
            sps=8,
            snr_db=snr_db,
            cfo_hz=37.0,
            num_frames=frames,
            payload_len=payload,
            fec=fec,
            interleave=interleave,
            seed=seed,
            keep_clean=True,
        )
    )
    return signal, truth


def _inject(signal: Signal, delay: float = 0.35) -> Signal:
    return Signal(samples=fractional_delay(signal.samples, delay), sample_rate=signal.sample_rate)


def _threshold(rows: list[dict], target: float = TARGET_CRC) -> float | None:
    ordered = sorted(rows, key=lambda r: r["snr_db"])
    previous = None
    for row in ordered:
        if row["crc_rate"] >= target:
            if previous is None or previous["crc_rate"] >= target:
                return float(row["snr_db"])
            span = row["crc_rate"] - previous["crc_rate"]
            ratio = (target - previous["crc_rate"]) / span if span > 0 else 0.0
            return float(previous["snr_db"] + ratio * (row["snr_db"] - previous["snr_db"]))
        previous = row
    return None


def part_curves(snrs: list[float], frames: int, report_path: Path) -> int:
    print("=" * 112)
    print("M7-A 平滑端到端曲线：QPSK，每点 {} 帧，注入 0.35 符号偏移 + 37 Hz 频偏".format(frames))
    print("=" * 112)
    print("{:8s}".format("Es/N0") + "".join("{:>16s}".format(CODE_LABEL[c]) for c in FEC_MODES) + "{:>10s}".format("已解析帧"))
    print("-" * 112)
    table: dict[str, list[dict]] = {c: [] for c in FEC_MODES}
    for snr_db in snrs:
        line = "{:8.0f}".format(snr_db)
        parsed_note = ""
        for code in FEC_MODES:
            signal, truth = _signal(code, snr_db, frames)
            result = extract_frames(_inject(signal), modulation="qpsk", truth=truth)
            summary = result.summary
            parsed = summary["frames"]
            crc_ok = summary["frames_crc_ok"]
            exact = summary["payload_bytes_exact"]
            total_bytes = summary["payload_bytes_total"]
            row = {
                "fec": code,
                "snr_db": float(snr_db),
                "frames_generated": frames,
                "frames": parsed,
                "frames_crc_ok": crc_ok,
                "crc_rate": (crc_ok / parsed) if parsed else 0.0,
                "payload_bytes_total": total_bytes,
                "payload_bytes_exact": exact,
                "exact_rate": (exact / total_bytes) if total_bytes else 0.0,
            }
            table[code].append(row)
            parsed_note = str(parsed)
            line += "{:>16s}".format("{:.2f}".format(row["crc_rate"]))
        print(line + "{:>10s}".format(parsed_note))
    print("-" * 112)
    print("（每格 = CRC 通过率 = 通过帧数 / 已解析帧数；最后一列是该信噪比下解析出的帧数，上限 {}）".format(frames))

    print("")
    print("M7-A2 可用门限（CRC 通过率 >= {:.0%} 所需 Es/N0，线性插值）".format(TARGET_CRC))
    thresholds: dict[str, float | None] = {}
    for code in FEC_MODES:
        value = _threshold(table[code])
        thresholds[code] = value
        print("  {:14s} {}".format(
            CODE_LABEL[code],
            "未达标（>= {:.0f} dB）".format(max(snrs)) if value is None else "{:.2f} dB".format(value),
        ))
    uncoded = thresholds.get("none")
    best_code = min((c for c in FEC_MODES if c != "none" and thresholds.get(c) is not None),
                    key=lambda c: thresholds[c], default=None)
    gain = (uncoded - thresholds[best_code]) if (uncoded is not None and best_code) else None
    if gain is not None:
        print("  最优：{}，相对无编码改善 **{:.2f} dB**".format(CODE_LABEL[best_code], gain))

    report = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frames_per_point": int(frames),
        "payload_bytes": PAYLOAD,
        "snrs": [float(s) for s in snrs],
        "table": table,
        "thresholds": thresholds,
        "best_code": best_code,
        "gain_db": gain,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        from spxh.tools.plots import plot_accuracy_curve

        FIG_DIR.mkdir(parents=True, exist_ok=True)
        series = {"CRC 通过率·" + CODE_LABEL[c]: [r["crc_rate"] for r in table[c]] for c in FEC_MODES}
        plot_accuracy_curve(snrs, series, FIG_DIR / "m7_smooth_curves.png",
                            title="M7 端到端平滑曲线：CRC 通过率 vs Es/N0（QPSK，每点 {} 帧）".format(frames))
        print("")
        print("图: docs/figures/m7_smooth_curves.png")
    except Exception as exc:  # noqa: BLE001
        print("跳过出图: " + str(exc))
    print("报告: " + str(report_path))
    return 0


def part_blind(report_path: Path) -> int:
    """盲编码识别：帧头里的 fec/interleave 字段被打错时，用"假设 + 重写帧头 + CRC 校验"找回来."""
    print("=" * 112)
    print("M7-B 盲编码识别：把帧头里的 fec/interleave 字段打成 0（谎报为无编码），看能不能自己纠正")
    print("=" * 112)
    print("{:8s} {:>12s} {:>14s} {:>16s} {:>10s}".format("真编码", "按帧头声明", "盲搜索恢复", "命中的假设", "载荷一致"))
    print("-" * 112)
    rows: list[dict] = []
    ok = True
    for fec in ("conv", "rs", "ccsds", "ldpc"):
        rng = np.random.default_rng(7)
        info = bytes(rng.integers(0, 256, size=PAYLOAD, dtype=np.uint8))
        bits, _header, _crc = build_frame(info, "qpsk", fec=fec, interleave=False)
        tampered = np.asarray(bits, dtype=np.uint8).copy()
        # 帧头 byte1 的 bit3..1 = 帧比特 25,26,27（fec 字段）、bit0 = 帧比特 28（交织位）
        tampered[25] = 0
        tampered[26] = 0
        tampered[27] = 0
        tampered[28] = 0
        declared = parse_frame(tampered)
        hit = None
        for hypothesis_fec, hypothesis_inter in _BLIND_HYPOTHESES:
            candidate = parse_frame(tampered, force_fec=hypothesis_fec, force_interleave=hypothesis_inter)
            if candidate["ok"] and bytes(candidate["payload"]) == info:
                hit = (hypothesis_fec, hypothesis_inter)
                break
        same = bool(hit and bytes(parse_frame(tampered, force_fec=hit[0], force_interleave=hit[1])["payload"]) == info)
        passed = (not declared["ok"]) and hit is not None and hit[0] == fec and same
        ok = ok and passed
        rows.append({
            "fec": fec, "declared_ok": bool(declared["ok"]),
            "blind_hit": None if hit is None else {"fec": hit[0], "interleave": bool(hit[1])},
            "payload_match": same, "pass": passed,
        })
        print("{:8s} {:>12s} {:>14s} {:>16s} {:>10s}".format(
            fec, "通过" if declared["ok"] else "失败（预期）",
            "恢复" if hit else "未恢复",
            "-" if hit is None else "{}{}".format(hit[0], "+交织" if hit[1] else ""),
            "一致" if same else "不一致"))
    print("-" * 112)
    checks = [{
        "name": "帧头 fec 字段被打错后，盲搜索能把四种编码全部恢复",
        "value": sum(1 for r in rows if r["pass"]), "target": 4, "pass": bool(ok),
    }]
    print("M7-B 验收清单")
    for item in checks:
        print("  [{}] {}（实测 {}，目标 {}）".format(
            "PASS" if item["pass"] else "FAIL", item["name"], item["value"], item["target"]))
    report_path.write_text(json.dumps({
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": rows, "checks": checks,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("报告: " + str(report_path))
    return 0 if ok else 1


def part_erasure(report_path: Path) -> int:
    """RS 擦除译码：已知不可信位置时，纠正能力从 t 翻倍到 2t."""
    print("")
    print("=" * 112)
    print("M7-C RS 擦除译码：在编码段里打一段连续字节错误（RS(76,60)，t=8，2t=16）")
    print("=" * 112)
    print("{:>6s} {:>14s} {:>16s} {:>24s}".format("字节突发", "无擦除信息", "已知位置(擦除)", "按置信度自适应挑擦除"))
    print("-" * 112)
    rng = np.random.default_rng(11)
    info = bytes(rng.integers(0, 256, size=PAYLOAD, dtype=np.uint8))
    bits, _header, _crc = build_frame(info, "qpsk", fec="rs", interleave=False)
    body_start = PREAMBLE_BITS + HEADER_BITS   # M9 起帧头 5 字节（含 CRC-8），不能写死 13+32
    rows: list[dict] = []
    ok = True
    for burst in (8, 9, 12, 16, 17):
        tampered = np.asarray(bits, dtype=np.uint8).copy()
        start = body_start + 8 * 10
        tampered[start : start + 8 * burst] ^= 1
        plain = parse_frame(tampered)
        positions = list(range(10, 10 + burst))
        known = parse_frame(tampered, erasures=positions)
        # 现实做法：解调给出每个比特的置信度，挑最不可信的 12 个字节当擦除
        llr = np.full(tampered.size, 6.0)
        weak = np.zeros(tampered.size, dtype=bool)
        weak[start : start + 8 * burst] = True
        llr[weak] = 0.3
        means = llr[body_start : body_start + 8 * (PAYLOAD + 2)].reshape(-1, 8).mean(axis=1)
        # 现实做法：置信度低于门限的字节全部标为擦除（上限 = nsym = 2t），
        # 而不是"固定挑 12 个"——后者在突发更长时会漏掉真正的坏位置
        low = np.flatnonzero(means < 1.0)
        if low.size == 0:
            low = np.argsort(means)[:12]
        guessed = [int(x) for x in low[:16]]
        heuristic = parse_frame(tampered, llr=llr, erasures=guessed)
        rows.append({
            "burst_bytes": burst, "plain_ok": bool(plain["ok"]), "known_ok": bool(known["ok"]),
            "heuristic_ok": bool(heuristic["ok"]),
        })
        # 8 字节在 t 之内：不标擦除也能纠；9~16 字节在 t 与 2t 之间：只有擦除译码能救
        # 17 字节超出 2t：谁都救不了，必须如实报不可纠（不许猜）
        if 8 < burst <= 16:
            ok = ok and (not plain["ok"]) and known["ok"]
        elif burst > 16:
            ok = ok and (not known["ok"])
        print("{:>6d} {:>14s} {:>16s} {:>22s}".format(
            burst, "通过" if plain["ok"] else "失败", "通过" if known["ok"] else "失败",
            "通过" if heuristic["ok"] else "失败"))
    print("-" * 112)
    print("（t=8：未知位置最多纠 8 个；已知位置最多纠 16 个。9~16 字节这一档正好体现擦除译码的价值。）")
    checks = [{
        "name": "超过 t 但在 2t 以内的突发，只有擦除译码能救回来",
        "value": sum(1 for r in rows if 8 < r["burst_bytes"] <= 16 and (not r["plain_ok"]) and r["known_ok"]),
        "target": 3, "pass": bool(ok),
    }]
    print("M7-C 验收清单")
    for item in checks:
        print("  [{}] {}（实测 {}，目标 {}）".format(
            "PASS" if item["pass"] else "FAIL", item["name"], item["value"], item["target"]))
    report_path.write_text(json.dumps({
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "t": 8, "nsym": 16, "rows": rows, "checks": checks,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("报告: " + str(report_path))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="M7 验收自检")
    parser.add_argument("--part", default="all", choices=("curves", "blind", "erasure", "all"))
    parser.add_argument("--snrs", default="2,4,5,6,7,8,9,10,11,12,13,14")
    parser.add_argument("--frames", type=int, default=24)
    args = parser.parse_args()
    snrs = [float(s) for s in str(args.snrs).split(",") if s.strip()]
    started = time.time()
    code = 0
    if args.part in ("curves", "all"):
        code |= part_curves(snrs, int(args.frames), ROOT / "models" / "check_m7_curves_report.json")
    if args.part in ("blind", "all"):
        code |= part_blind(ROOT / "models" / "check_m7_blind_report.json")
    if args.part in ("erasure", "all"):
        code |= part_erasure(ROOT / "models" / "check_m7_erasure_report.json")
    print("")
    print("耗时 {:.1f} s".format(time.time() - started))
    return code


if __name__ == "__main__":
    sys.exit(main())
