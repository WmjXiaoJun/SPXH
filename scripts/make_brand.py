import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Circle

# 找可用中文字体（找不到就只用拉丁字母，图里不硬塞中文）
names = {f.name for f in font_manager.fontManager.ttflist}
cjk = next((n for n in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "SimSun") if n in names), None)
print("可用中文字体:", cjk)

BG = "#0b1220"
CYAN = "#38bdf8"
GREEN = "#34d399"
AMBER = "#fbbf24"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"


def emblem(ax, cx, cy, r):
    """一个"接收机 + 星座"的抽象标记."""
    ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor=CYAN, lw=max(1.5, r * 0.09)))
    ax.add_patch(Circle((cx, cy), r * 0.62, facecolor="none", edgecolor=CYAN, lw=max(0.8, r * 0.045), alpha=0.55))
    # QPSK 星座点
    for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        ax.add_patch(Circle((cx + dx * r * 0.42, cy + dy * r * 0.34), r * 0.085,
                            facecolor=GREEN if dx * dy > 0 else AMBER, edgecolor="none"))
    # 底部一条"频谱/相关峰"
    xs = np.linspace(cx - r * 0.86, cx + r * 0.86, 220)
    peak = np.exp(-((xs - cx) / (r * 0.16)) ** 2)
    ripple = 0.18 * np.sin((xs - cx) / (r * 0.1))
    ys = cy - r * 0.72 + r * 0.30 * peak + r * 0.05 * ripple
    ax.plot(xs, ys, color=CYAN, lw=max(1.0, r * 0.06), solid_capstyle="round")
    return ax


# ---------------------------------------------------------------- logo 512x512
fig = plt.figure(figsize=(5.12, 5.12), dpi=100, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(BG); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.03, 0.03), 0.94, 0.94, boxstyle="round,pad=0.02,rounding_size=0.16",
                            facecolor="#111c33", edgecolor="#1e3a5f", lw=1.5))
emblem(ax, 0.5, 0.56, 0.26)
ax.text(0.5, 0.20, "SPXH", ha="center", va="center", color=TEXT, fontsize=34, fontweight="bold",
        fontfamily="DejaVu Sans")
ax.text(0.5, 0.115, "RF SIGNAL ANALYSIS", ha="center", va="center", color=MUTED, fontsize=9,
        fontfamily="DejaVu Sans")
fig.savefig("docs/figures/logo.png", facecolor=BG)
plt.close(fig)
print("logo 写好")

# -------------------------------------------------------------- banner 1280x360
fig = plt.figure(figsize=(12.8, 3.6), dpi=100, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(BG); ax.set_xlim(0, 1280); ax.set_ylim(0, 360); ax.axis("off")
# 背景网格
for x in range(0, 1280, 40):
    ax.plot([x, x], [0, 360], color="#13203a", lw=0.6, zorder=0)
for y in range(0, 360, 40):
    ax.plot([0, 1280], [y, y], color="#13203a", lw=0.6, zorder=0)
# 背景频谱剪影
rng = np.random.default_rng(7)
xs = np.arange(0, 1280, 4)
env = np.exp(-((xs - 300) / 260) ** 2) + 0.35 * np.exp(-((xs - 900) / 180) ** 2)
spec = env * (0.35 + 0.35 * np.abs(np.sin(xs / 23.0)) + 0.2 * rng.random(xs.size))
ax.fill_between(xs, 0, spec * 90, color=CYAN, alpha=0.10, zorder=0)
emblem(ax, 150, 180, 78)
title = "SPXH · 智能射频信号分析平台" if cjk else "SPXH · Smart RF Signal Analysis Platform"
ax.text(280, 226, title, color=TEXT, fontsize=34, fontweight="bold",
        fontfamily=(cjk or "DejaVu Sans"), va="center")
ax.text(280, 168, "从一段未知 I/Q 到可读业务比特 —— 每一步都说得出依据", color=CYAN,
        fontsize=17, fontfamily=(cjk or "DejaVu Sans"), va="center")
ax.text(280, 124, "M0 合成/IO · M1 参数估计 · M2 调制识别 · M3 解调/LLR · M4 信道编码 · M5 帧同步",
        color=MUTED, fontsize=11, fontfamily=(cjk or "DejaVu Sans"), va="center")
ax.text(280, 100, "M6 编码链路 · M7 盲编码识别 · M8 谱相关 · M9 帧头 CRC · M10 LLM 报告 · M11 工具代理 · M12 软前导",
        color=MUTED, fontsize=11, fontfamily=(cjk or "DejaVu Sans"), va="center")
fig.savefig("docs/figures/banner.png", facecolor=BG)
plt.close(fig)
print("banner 写好")
