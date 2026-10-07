"""M1/M2 可视化工具（可选依赖 matplotlib，仅用于出图，不参与估计链路）."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from spxh.core.dsp.spectrum import PsdResult, smooth_psd
from spxh.core.dsp.timefreq import Waterfall

__all__ = [
    "setup_matplotlib",
    "plot_spectrum",
    "plot_waterfall",
    "plot_error_curves",
    "plot_confusion",
    "plot_reliability",
    "plot_accuracy_curve",
    "plot_sync_traces",
    "plot_ber_curves",
]


def setup_matplotlib():
    """配置 matplotlib（Agg 后端 + 尽量使用中文字体）."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams

    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC"):
        if name in available:
            rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    rcParams["axes.unicode_minus"] = False
    return matplotlib


def _save(fig, out_path) -> str:
    target = Path(str(out_path))
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(target, dpi=130)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return str(target)


def plot_spectrum(
    psd: PsdResult,
    out_path,
    cfo_est: Optional[float] = None,
    cfo_true: Optional[float] = None,
    occupied: Optional[dict[str, Any]] = None,
    title: str = "功率谱（Welch）",
    residual: bool = True,
) -> str:
    """画功率谱：原始 PSD（dB）、噪声底、残余谱、估计/真实载频、占用带宽."""
    setup_matplotlib()
    import matplotlib.pyplot as plt

    freqs = psd.freqs
    floor = psd.noise_floor()
    psd_db = 10.0 * np.log10(np.maximum(psd.psd, 1e-300))
    residual_linear = np.maximum(smooth_psd(psd.psd, 9) - floor, 0.0)
    peak_residual = float(np.max(residual_linear)) or 1.0
    residual_db = 10.0 * np.log10(np.maximum(residual_linear / peak_residual, 1e-6))

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(freqs, psd_db, lw=0.8, color="#1f77b4", label="PSD (dB/Hz)")
    axes[0].axhline(10.0 * np.log10(max(floor, 1e-300)), color="#d62728", ls="--", lw=1.0, label="噪声底估计")
    axes[0].set_ylabel("dB/Hz")
    axes[0].set_title(title)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.25)

    axes[1].plot(freqs, residual_db, lw=0.9, color="#2ca02c", label="残余谱（扣噪声底，相对峰值）")
    axes[1].set_xlabel("频率 (Hz)")
    axes[1].set_ylabel("dB (相对残余峰值)")
    axes[1].set_ylim(-62.0, 6.0)
    axes[1].grid(alpha=0.25)

    if occupied is not None:
        for ax in axes:
            ax.axvspan(occupied["f_low"], occupied["f_high"], color="#ff7f0e", alpha=0.10, label="占用带宽")
        axes[1].axvline(occupied["center"], color="#ff7f0e", ls=":", lw=1.2)
    if cfo_est is not None:
        axes[1].axvline(cfo_est, color="#d62728", lw=1.4, label="载频估计 {:.3f} Hz".format(cfo_est))
    if cfo_true is not None:
        axes[1].axvline(cfo_true, color="#000000", ls="--", lw=1.2, label="载频真值 {:.3f} Hz".format(cfo_true))
    axes[1].legend(loc="upper right", fontsize=9)

    return _save(fig, out_path)


def plot_waterfall(waterfall: Waterfall, out_path, title: str = "STFT 瀑布图") -> str:
    setup_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    mesh = ax.pcolormesh(waterfall.times, waterfall.freqs, waterfall.magnitude_db, shading="auto", cmap="viridis")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("频率 (Hz)")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="幅度 (dB)")
    return _save(fig, out_path)


def plot_error_curves(curves: Sequence[dict[str, Any]], out_path, title: str = "") -> str:
    """画"同步误差 -> EVM/BER"因果曲线.

    curves: [{label, x, xlabel, evm, ber, xscale?}, ...]
    """
    setup_matplotlib()
    import matplotlib.pyplot as plt

    count = len(curves)
    fig, axes = plt.subplots(1, count, figsize=(6.0 * count, 4.6), squeeze=False)
    for index, curve in enumerate(curves):
        ax = axes[0][index]
        x = np.asarray(curve["x"], dtype=np.float64)
        evm = np.asarray(curve["evm"], dtype=np.float64)
        ber = np.asarray(curve["ber"], dtype=np.float64)
        ax.plot(x, evm, "o-", color="#1f77b4", label="EVM (%)")
        ax.set_xlabel(curve.get("xlabel", "误差"))
        ax.set_ylabel("EVM (%)", color="#1f77b4")
        ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax.grid(alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(x, np.maximum(ber, 1e-6), "s--", color="#d62728", label="BER")
        ax2.set_ylabel("BER", color="#d62728")
        ax2.set_yscale("log")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax.set_title(curve.get("label", ""))
        if "xscale" in curve:
            ax.set_xscale(curve["xscale"])
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=9)
    if title:
        fig.suptitle(title)
    return _save(fig, out_path)


def plot_confusion(matrix, labels, out_path, title: str = "混淆矩阵") -> str:
    """画混淆矩阵热图（行=真值，列=预测）."""
    setup_matplotlib()
    import matplotlib.pyplot as plt

    data = np.asarray(matrix, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 3.0, 1.0 * len(labels) + 2.4))
    mesh = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("预测")
    ax.set_ylabel("真值")
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = int(data[i, j])
            if value:
                ax.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if data[i, j] > 0.5 * data.max() else "#111111",
                )
    fig.colorbar(mesh, ax=ax, label="样本数")
    return _save(fig, out_path)


def plot_reliability(reliability: dict, out_path, title: str = "可靠性图", ece: float = float("nan")) -> str:
    """可靠性图：横轴置信度、纵轴实际准确率，对角虚线为完美校准."""
    setup_matplotlib()
    import matplotlib.pyplot as plt

    centers = np.asarray(reliability["bin_center"], dtype=np.float64)
    accuracy = np.asarray(reliability["accuracy"], dtype=np.float64)
    counts = np.asarray(reliability["count"], dtype=np.float64)
    mask = np.isfinite(accuracy) & (counts > 0)

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.plot([0, 1], [0, 1], "--", color="#888888", lw=1.0, label="完美校准")
    if np.any(mask):
        sizes = 20.0 + 240.0 * (counts[mask] / max(counts.max(), 1.0))
        ax.scatter(centers[mask], accuracy[mask], s=sizes, color="#1f77b4", edgecolor="white", zorder=3, label="分箱实际准确率")
        ax.plot(centers[mask], accuracy[mask], color="#1f77b4", lw=1.0, alpha=0.6)
    ax.set_xlabel("预测置信度")
    ax.set_ylabel("实际准确率")
    ax.set_title("{:s}（ECE={:.4f}，点大小=样本数）".format(title, ece))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return _save(fig, out_path)


def plot_sync_traces(traces: dict, out_path, title: str = "同步环路观测量") -> str:
    """画定时误差 / 载波相位误差 / 频率估计三条轨迹（M3 解调页同款数据）."""
    setup_matplotlib()
    import matplotlib.pyplot as plt

    index = np.asarray(traces.get("index", []), dtype=np.float64)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    series = [
        ("timing_error", "定时误差（归一化）", "#1f77b4"),
        ("phase_error_rad", "载波相位误差 (rad)", "#d62728"),
        ("freq_estimate_hz", "载波频率估计 (Hz)", "#2ca02c"),
    ]
    for ax, (key, label, color) in zip(axes, series):
        values = np.asarray(traces.get(key, []), dtype=np.float64)
        if values.size == index.size and values.size:
            ax.plot(index, values, lw=0.9, color=color)
        ax.axhline(0.0, color="#888888", ls="--", lw=0.8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("符号序号")
    axes[0].set_title(title)
    return _save(fig, out_path)


def plot_ber_curves(x, series: dict, out_path, title: str = "误码率 vs Es/N0", xlabel: str = "Es/N0 (dB)") -> str:
    """多条 BER 曲线的对比图（对数纵轴）."""
    setup_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    for index, (label, values) in enumerate(series.items()):
        curve = np.maximum(np.asarray(values, dtype=np.float64), 1e-6)
        ax.semilogy(x, curve, "o-", color=colors[index % len(colors)], label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("BER")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    if title:
        ax.set_title(title)
    return _save(fig, out_path)


def plot_accuracy_curve(x, series: dict, out_path, title: str = "") -> str:
    """准确率 / 门控通过率随 SNR 变化."""
    setup_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    for index, (label, values) in enumerate(series.items()):
        ax.plot(x, values, "o-", color=colors[index % len(colors)], label=label)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("比例")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    if title:
        ax.set_title(title)
    return _save(fig, out_path)
