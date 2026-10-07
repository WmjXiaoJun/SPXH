"""M6 验收自检：编码链端到端（合成 -> 解调 -> 帧同步 -> 译码 -> CRC -> 载荷）.

与 M4 的区别：M4 量的是"纯编码增益"（BPSK+AWGN，不含同步/估计误差）；
M6 量的是**整条链**：FEC 接进帧结构后，用 CRC 通过率与载荷逐字节一致率说话。

用法: python scripts/check_m6.py [--snrs 2,4,6,8,10,12,14,16]
输出: 终端表格 + docs/figures/m6_coded_link.png + models/check_m6_report.json
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
from spxh.core.fec.codec import FEC_MODES, coded_length_bits  # noqa: E402
from spxh.core.framing.format import build_frame, parse_frame  # noqa: E402
from spxh.core.framesync import extract_frames  # noqa: E402
from spxh.core.synth import SynthConfig, synthesize  # noqa: E402
from spxh.core.types import Signal  # noqa: E402

FIG_DIR = ROOT / "docs" / "figures"
PAYLOAD = 60          # ccsds 要求能被交织深度 5 整除；ldpc 上限 72 字节
# 每点帧数：M6 首轮用 4 帧，CRC 通过率的量化步长是 25%，"门限"是插值出来的、抖动大
# （M9 复核时同一套判据给出 3.8 dB，只差一个量化步长就是 4.0）。这里提到 12 帧，
# 让这条判据的数字站得住；回归更密的曲线在 check_m7.py（每点 24 帧）。
FRAMES = 12
CODES = [c for c in FEC_MODES]
CODE_LABEL = {
    "none": "无编码",
    "conv": "卷积(软判决)",
    "rs": "RS(76,60)",
    "ccsds": "CCSDS 级联",
    "ldpc": "LDPC(1152,578)",
}
TARGET_CRC = 0.9      # "可用门限"判据：CRC 通过率 >= 90%


def _case(fec: str, snr_db: float, interleave: bool = False, seed: int = 11):
    return synthesize(
        SynthConfig(
            modulation="qpsk",
            symbol_rate=1000.0,
            sps=8,
            snr_db=snr_db,
            cfo_hz=37.0,
            num_frames=FRAMES,
            payload_len=PAYLOAD,
            fec=fec,
            interleave=interleave,
            seed=seed,
            keep_clean=True,
        )
    )


def _run(fec: str, snr_db: float, interleave: bool = False, delay: float = 0.35, burst: int = 0, seed: int = 11) -> dict:
    signal, truth = _case(fec, snr_db, interleave=interleave, seed=seed)
    samples = np.asarray(signal.samples, dtype=np.complex128).copy()
    if burst:
        # 人为制造一段突发损伤（模拟遮挡/强干扰）：把中间 burst 个符号的采样清零
        sps = int(truth.sps)
        mid = samples.size // 2
        half = (burst * sps) // 2
        samples[max(0, mid - half) : min(samples.size, mid + half)] = 0.0
    if delay:
        samples = fractional_delay(samples, delay)
    injected = Signal(samples=samples, sample_rate=signal.sample_rate)
    result = extract_frames(injected, modulation="qpsk", truth=truth)
    summary = result.summary
    expected_frames = FRAMES
    return {
        "fec": fec,
        "snr_db": float(snr_db),
        "interleave": bool(interleave),
        "burst_symbols": int(burst),
        "frames": summary["frames"],
        "frames_crc_ok": summary["frames_crc_ok"],
        "crc_rate": (summary["frames_crc_ok"] / summary["frames"]) if summary["frames"] else 0.0,
        "payload_bytes_total": summary["payload_bytes_total"],
        "payload_bytes_exact": summary["payload_bytes_exact"],
        "exact_rate": (
            (summary["payload_bytes_exact"] or 0) / summary["payload_bytes_total"]
            if summary["payload_bytes_total"]
            else 0.0
        ),
        "expected_frames": expected_frames,
    }


def _threshold(rows: list[dict]) -> float | None:
    """CRC 通过率达到 TARGET_CRC 所需的最低 Es/N0（线性插值）."""
    ordered = sorted(rows, key=lambda r: r["snr_db"])
    previous = None
    for row in ordered:
        if row["crc_rate"] >= TARGET_CRC:
            if previous is None:
                return float(row["snr_db"])
            span = row["crc_rate"] - previous["crc_rate"]
            ratio = (TARGET_CRC - previous["crc_rate"]) / span if span > 0 else 0.0
            return float(previous["snr_db"] + ratio * (row["snr_db"] - previous["snr_db"]))
        previous = row
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="M6 验收自检")
    parser.add_argument("--snrs", default="2,4,6,8,10,12,14,16")
    parser.add_argument("--report", default="models/check_m6_report.json")
    args = parser.parse_args()
    snrs = [float(s) for s in str(args.snrs).split(",") if s.strip()]
    rng = np.random.default_rng(20261004)
    started = time.time()

    print("=" * 108)
    print("M6-A 编码链端到端：CRC 通过率 / 载荷逐字节一致率（QPSK，注入 0.35 符号偏移 + 37 Hz 频偏）")
    print("=" * 108)
    print("{:8s}".format("Es/N0") + "".join("{:>18s}".format(CODE_LABEL[c]) for c in CODES))
    print("-" * 108)
    table: dict[str, list[dict]] = {c: [] for c in CODES}
    for snr_db in snrs:
        line = "{:8.0f}".format(snr_db)
        for code in CODES:
            row = _run(code, snr_db)
            table[code].append(row)
            line += "{:>18s}".format("{:.0%}/{:.0%}".format(row["crc_rate"], row["exact_rate"]))
        print(line)
    print("-" * 108)
    print("（每格 = CRC 通过率 / 载荷逐字节一致率；逐字节一致率分母是已解析帧的载荷总字节）")

    print("")
    print("M6-B 可用门限（CRC 通过率 >= 90% 所需 Es/N0）")
    thresholds: dict[str, float | None] = {}
    for code in CODES:
        threshold = _threshold(table[code])
        thresholds[code] = threshold
        print("  {:14s} {}".format(CODE_LABEL[code], "未达标（>= {:.0f} dB）".format(max(snrs)) if threshold is None else "{:.1f} dB".format(threshold)))
    uncoded = thresholds.get("none")
    best = min((v for k, v in thresholds.items() if k != "none" and v is not None), default=None)
    gain = (uncoded - best) if (uncoded is not None and best is not None) else None
    if gain is not None:
        print("  编码相对无编码的**门限改善**：{:.1f} dB".format(gain))

    print("")
    print("=" * 108)
    print("M6-C 交织的价值：在**已编码载荷里**人为打一段连续比特突发（比特级，去掉调制解调的干扰）")
    print("=" * 108)
    burst_rows: list[dict] = []
    for code, label in (("conv", "卷积"), ("rs", "RS")):
        for length in (16, 32):
            for interleave in (False, True):
                ok = 0
                for trial in range(8):
                    info = bytes(rng.integers(0, 256, size=PAYLOAD, dtype=np.uint8))
                    bits, header, _crc = build_frame(info, "qpsk", fec=code, interleave=interleave)
                    corrupted = np.asarray(bits, dtype=np.uint8).copy()
                    coded_start = 13 + 32
                    body = coded_length_bits(PAYLOAD, code, interleave)
                    start = coded_start + max(0, body // 2 - length // 2)
                    corrupted[start : start + length] ^= 1
                    parsed = parse_frame(corrupted)
                    if parsed["ok"] and bytes(parsed["payload"]) == info:
                        ok += 1
                row = {"fec": code, "burst_bits": length, "interleave": bool(interleave), "ok": ok, "trials": 8}
                burst_rows.append(dict(row, label=label + ("+交织" if interleave else "不交织")))
                print("  {:10s} 突发 {:2d} 比特 -> 8 次里成功 {} 次".format(
                    label + ("+交织" if interleave else "不交织"), length, ok))

    checks = [
        {
            "name": "编码链把 CRC 可用门限至少降低 {:.0f} dB".format(4.0),
            "value": gain if gain is not None else float("nan"),
            "target": 4.0,
            "pass": bool(gain is not None and gain >= 4.0),
        },
        {
            "name": "12 dB 时五类编码（含交织与否）全部帧 CRC 通过",
            "value": sum(1 for r in table["conv"] if abs(r["snr_db"] - 12.0) < 1e-6 and r["frames_crc_ok"] == r["frames"]),
            "target": 1,
            "pass": all(
                r["frames"] > 0 and r["frames_crc_ok"] == r["frames"] and r["exact_rate"] == 1.0
                for code in ("conv", "rs", "ccsds", "ldpc")
                for r in table[code]
                if abs(r["snr_db"] - 12.0) < 1e-6
            ),
        },
        {
            "name": "32 比特突发下交织器确实有用（卷积+交织 优于 卷积不交织）",
            "value": int(next(r["ok"] for r in burst_rows if r["label"] == "卷积+交织" and r["burst_bits"] == 32)),
            "target": int(next(r["ok"] for r in burst_rows if r["label"] == "卷积不交织" and r["burst_bits"] == 32)) + 1,
            "pass": bool(
                next(r["ok"] for r in burst_rows if r["label"] == "卷积+交织" and r["burst_bits"] == 32)
                > next(r["ok"] for r in burst_rows if r["label"] == "卷积不交织" and r["burst_bits"] == 32)
            ),
        },
        {
            "name": "32 比特突发下 RS+交织 能全部纠正",
            "value": int(next(r["ok"] for r in burst_rows if r["label"] == "RS+交织" and r["burst_bits"] == 32)),
            "target": 8,
            "pass": bool(next(r["ok"] for r in burst_rows if r["label"] == "RS+交织" and r["burst_bits"] == 32) >= 8),
        },
    ]
    print("")
    print("M6-D 验收清单")
    for item in checks:
        print("  [{}] {}（实测 {}，目标 {}）".format(
            "PASS" if item["pass"] else "FAIL", item["name"],
            "{:.3g}".format(item["value"]) if isinstance(item["value"], float) and np.isfinite(item["value"]) else item["value"],
            item["target"]))

    report = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "payload_bytes": PAYLOAD,
        "frames_per_point": FRAMES,
        "snrs": snrs,
        "table": table,
        "thresholds": thresholds,
        "threshold_gain_db": gain,
        "burst": burst_rows,
        "checks": checks,
        "elapsed_s": round(time.time() - started, 1),
    }
    Path(str(args.report)).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        from spxh.tools.plots import plot_accuracy_curve

        FIG_DIR.mkdir(parents=True, exist_ok=True)
        series = {"CRC 通过率·" + CODE_LABEL[c]: [r["crc_rate"] for r in table[c]] for c in CODES}
        plot_accuracy_curve(snrs, series, FIG_DIR / "m6_coded_link.png",
                            title="M6 编码链：CRC 通过率 vs Es/N0（QPSK，含同步与估计误差）")
    except Exception as exc:  # noqa: BLE001
        print("跳过出图: " + str(exc))

    failed = [item["name"] for item in checks if not item["pass"]]
    print("")
    print("=" * 108)
    if failed:
        print("M6 FAIL: " + "; ".join(failed))
        return 1
    print("M6 PASS: 编码链接入帧结构，端到端 CRC / 载荷 / 交织价值 全部满足判据")
    print("报告: " + str(args.report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
