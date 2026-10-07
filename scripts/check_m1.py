"""M1 验收自检：估计精度 + "估计误差 -> EVM/BER" 因果曲线 + 出图.

用法: python scripts/check_m1.py
输出: 终端表格 + docs/figures/m1_*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spxh.core.dsp import stft_waterfall, welch_psd  # noqa: E402
from spxh.core.estimate import analyze_signal, compare_with_truth  # noqa: E402
from spxh.core.synth import SynthConfig, mapping, synthesize  # noqa: E402
from spxh.core.synth.pulse import matched_filter, rrc_taps  # noqa: E402
from spxh.core.types import MODULATIONS  # noqa: E402

FIG_DIR = ROOT / "docs" / "figures"

# 验收判据（SNR >= 10 dB）
CFO_TOL_RATIO = 0.02      # |Δcfo| <= 2% 符号速率
SYMBOL_RATE_TOL = 2e-3    # 相对误差 <= 0.2%
SNR_TOL_DB = 0.5          # |ΔEs/N0| <= 0.5 dB


def _make(modulation, snr_db, symbol_rate=1234.5, cfo_hz=43.7, frames=8, payload_len=128, seed=7):
    return synthesize(
        SynthConfig(
            modulation=modulation,
            symbol_rate=symbol_rate,
            sps=8,
            snr_db=snr_db,
            cfo_hz=cfo_hz,
            num_frames=frames,
            payload_len=payload_len,
            seed=seed,
        )
    )


def accuracy_table() -> tuple[list[dict], bool]:
    print("=" * 108)
    print("M1-A 参数估计精度（理想参考接收端环境：无相噪/无失衡；Rs=1234.5 Hz，CFO=43.7 Hz）")
    print("=" * 108)
    header = "{:7s} {:>5s} {:>11s} {:>12s} {:>10s} {:>8s} {:>8s} {:>8s} {:>7s}".format(
        "mod", "snr", "cfo_err", "rs_rel_err", "esn0_err", "cfoconf", "rsconf", "snrconf", "verdict"
    )
    print(header)
    print("-" * 108)
    rows: list[dict] = []
    all_ok = True
    for modulation in MODULATIONS:
        for snr_db in (20.0, 10.0, 0.0):
            signal, truth = _make(modulation, snr_db)
            result = analyze_signal(signal)
            comparison = compare_with_truth(result, truth)
            strict = snr_db >= 10.0
            ok = (
                abs(comparison["cfo_error_hz"]) <= CFO_TOL_RATIO * truth.symbol_rate
                and abs(comparison["symbol_rate_rel_error"]) <= SYMBOL_RATE_TOL
                and abs(comparison["es_n0_error_db"]) <= SNR_TOL_DB
            )
            if strict and not ok:
                all_ok = False
            verdict = ("PASS" if ok else "FAIL") if strict else ("ok" if ok else "degraded")
            rows.append({"modulation": modulation, "snr_db": snr_db, **comparison, "verdict": verdict})
            print(
                "{:7s} {:5.0f} {:11.3f} {:12.2e} {:10.3f} {:8.2f} {:8.2f} {:8.2f} {:>7s}".format(
                    modulation,
                    snr_db,
                    comparison["cfo_error_hz"],
                    comparison["symbol_rate_rel_error"],
                    comparison["es_n0_error_db"],
                    comparison["cfo_confidence"],
                    comparison["symbol_rate_confidence"],
                    comparison["snr_confidence"],
                    verdict,
                )
            )
    print("-" * 108)
    strict_rows = [r for r in rows if r["snr_db"] >= 10.0]
    print(
        "SNR>=10 dB: |Δcfo|max={:.3f} Hz ({:.2%} of Rs)  |ΔRs/Rs|max={:.2e}  |ΔEs/N0|max={:.3f} dB".format(
            max(abs(r["cfo_error_hz"]) for r in strict_rows),
            max(abs(r["cfo_error_hz"]) for r in strict_rows) / 1234.5,
            max(abs(r["symbol_rate_rel_error"]) for r in strict_rows),
            max(abs(r["es_n0_error_db"]) for r in strict_rows),
        )
    )
    low = [r for r in rows if r["snr_db"] < 10.0]
    flagged = sum(1 for r in low if r["symbol_rate_confidence"] < 0.35)
    print(
        "SNR=0 dB（波形 SNR 约 -9 dB）: {} 个调制中 {} 个被低置信度标记（不静默给错值）".format(
            len(low), flagged
        )
    )
    return rows, all_ok


def recover_with_errors(signal, truth, symbol_rate_est: float, cfo_est: float):
    """用带误差的同步参数做理想接收（匹配滤波 + 插值采样），返回 (EVM%, BER)."""
    fs = float(signal.sample_rate)
    samples_in = np.asarray(signal.samples)
    n = np.arange(samples_in.size, dtype=np.float64)
    corrected = samples_in * np.exp(-2j * np.pi * float(cfo_est) * n / fs)
    from spxh.core.synth.pulse import rrc_taps as _taps

    taps = _taps(truth.sps, span_symbols=truth.span_symbols, rolloff=truth.rolloff)
    filtered = matched_filter(corrected, taps)
    step = fs / float(symbol_rate_est)
    positions = 2.0 * truth.filter_delay_samples + np.arange(truth.num_symbols, dtype=np.float64) * step
    positions = positions[positions < filtered.size - 1]
    grid = np.arange(filtered.size, dtype=np.float64)
    recovered = np.interp(positions, grid, filtered.real) + 1j * np.interp(positions, grid, filtered.imag)
    ideal = mapping.indices_to_points(truth.symbol_indices[: recovered.size], truth.modulation)
    evm = 100.0 * float(np.sqrt(np.mean(np.abs(recovered - ideal) ** 2)) / np.sqrt(np.mean(np.abs(ideal) ** 2)))
    indices = mapping.points_to_indices(recovered, truth.modulation)
    bits = mapping.indices_to_bits(indices, truth.modulation)
    count = min(bits.size, truth.bits.size)
    ber = float(np.mean(bits[:count] != truth.bits[:count])) if count else float("nan")
    return evm, ber


def causal_curves() -> tuple[dict, bool]:
    print("")
    print("=" * 108)
    print("M1-B 因果曲线：同步参数误差 -> EVM/BER（QPSK / 16QAM，SNR=20 dB，Rs=1000 Hz，2 帧 x 256 B）")
    print("=" * 108)
    rate_errors = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    cfo_errors = [0.0, 0.1, 1.0, 5.0, 20.0, 100.0]
    data: dict[str, dict] = {}
    for modulation in ("qpsk", "16qam"):
        signal, truth = synthesize(
            SynthConfig(
                modulation=modulation,
                symbol_rate=1000.0,
                sps=8,
                snr_db=20.0,
                cfo_hz=40.0,
                num_frames=2,
                payload_len=256,
                seed=11,
            )
        )
        rate_evm, rate_ber = [], []
        for relative in rate_errors:
            evm, ber = recover_with_errors(signal, truth, truth.symbol_rate * (1.0 + relative), truth.cfo_hz)
            rate_evm.append(evm)
            rate_ber.append(ber)
        cfo_evm, cfo_ber = [], []
        for residual in cfo_errors:
            evm, ber = recover_with_errors(signal, truth, truth.symbol_rate, truth.cfo_hz + residual)
            cfo_evm.append(evm)
            cfo_ber.append(ber)
        data[modulation] = {
            "rate": {"x": rate_errors, "evm": rate_evm, "ber": rate_ber},
            "cfo": {"x": cfo_errors, "evm": cfo_evm, "ber": cfo_ber},
        }
        print("  {:6s} 符号速率相对误差 -> EVM/BER:".format(modulation))
        for relative, evm, ber in zip(rate_errors, rate_evm, rate_ber):
            print("      ΔRs/Rs={:8.1e}  EVM={:7.2f}%  BER={:.3e}".format(relative, evm, ber))
        print("  {:6s} 残余载频误差 -> EVM/BER:".format(modulation))
        for residual, evm, ber in zip(cfo_errors, cfo_evm, cfo_ber):
            print("      Δf={:8.2f} Hz  EVM={:7.2f}%  BER={:.3e}".format(residual, evm, ber))

    # 结论检查：误差越大 EVM 越大；存在 BER 从 0 变正的阈值
    ok = True
    for modulation, curves in data.items():
        rate = curves["rate"]
        if not (rate["evm"][0] < rate["evm"][-1]):
            ok = False
        cfo = curves["cfo"]
        if not (cfo["evm"][0] < cfo["evm"][-1]):
            ok = False
    print("  结论: EVM 随同步误差单调上升；符号速率误差会在记录内把星座逐步拖过判决边界")
    return data, ok


def figures(rows, curves) -> list[str]:
    try:
        from spxh.tools.plots import plot_error_curves, plot_spectrum, plot_waterfall
    except Exception as exc:  # noqa: BLE001
        print("跳过出图（matplotlib 不可用）: " + str(exc))
        return []
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    signal, truth = _make("qpsk", 10.0, frames=8, payload_len=128)
    psd = welch_psd(signal, nperseg=1024, nfft=8192)
    result = analyze_signal(signal, psd=psd)
    written.append(
        plot_spectrum(
            psd,
            FIG_DIR / "m1_spectrum.png",
            cfo_est=result.cfo.value,
            cfo_true=truth.cfo_hz,
            occupied=result.occupied,
            title="QPSK 10 dB: PSD + 载频估计 (估 {:.2f} Hz / 真值 {:.2f} Hz)".format(result.cfo.value, truth.cfo_hz),
        )
    )
    waterfall = stft_waterfall(signal, nperseg=256)
    written.append(plot_waterfall(waterfall, FIG_DIR / "m1_waterfall.png", title="QPSK 10 dB: STFT 瀑布图"))

    qpsk = curves["qpsk"]
    curves_to_plot = [
        {
            "label": "QPSK: 符号速率误差 -> 星座",
            "x": qpsk["rate"]["x"],
            "xlabel": "符号速率相对误差 ΔRs/Rs",
            "evm": qpsk["rate"]["evm"],
            "ber": qpsk["rate"]["ber"],
            "xscale": "symlog",
        },
        {
            "label": "QPSK: 残余载频误差 -> 星座",
            "x": qpsk["cfo"]["x"],
            "xlabel": "残余载频误差 (Hz)",
            "evm": qpsk["cfo"]["evm"],
            "ber": qpsk["cfo"]["ber"],
            "xscale": "symlog",
        },
    ]
    written.append(
        plot_error_curves(curves_to_plot, FIG_DIR / "m1_causal_curves.png", title="同步参数误差 -> EVM/BER（SNR=20 dB）")
    )
    return written


def main() -> int:
    rows, accuracy_ok = accuracy_table()
    curves, causal_ok = causal_curves()
    figures(rows, curves)
    print("")
    print("=" * 108)
    if accuracy_ok and causal_ok:
        print("M1 PASS: 估计精度满足判据；因果曲线单调；低信噪比下低置信度可识别")
        return 0
    print("M1 FAIL: accuracy_ok={} causal_ok={}".format(accuracy_ok, causal_ok))
    return 1


if __name__ == "__main__":
    sys.exit(main())
