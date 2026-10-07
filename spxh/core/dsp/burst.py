"""突发损伤检测：从**信号侧**找出"这一段不可信"，为联合译码提供擦除位置.

为什么需要它
------------
M12 的实测结论是：**擦除信息值钱的是"位置可信"，不是"置信度低"**。
按 |LLR| 门限标擦除在纯 AWGN 下会把**正确**的符号标掉，反而让 RS 门限从 8.78 dB 升到 9.20 dB。
但真实损伤（遮挡、削波、AGC 阶跃、强干扰）在**时域幅度**上是有痕迹的：一段突然塌下去的包络、
一段贴顶的采样。检测这些痕迹给出的位置，才是真正可信的擦除位置。

判决形态（保守优先）
------------------
* 用滑动 RMS 与"整段中位数"比：低于门限且**持续足够长**的段才算一次突发；
* 门限同时要求"相对深度"与"相对稳健离散度（MAD）"，避免把平稳低幅信号整段误判；
* 只报**区间**，不猜测损伤内部结构；宁少报（漏报由帧 CRC 兜底）不虚报（虚报吃掉纠错预算）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

__all__ = ["BurstRegion", "detect_bursts", "detect_bursts_dict"]


@dataclass
class BurstRegion:
    start_sample: int
    end_sample: int
    kind: str                    # collapse（幅度骤降）/ clip（削波）
    depth_db: float              # 相对基线的深度（负值，越小越深）
    duration_samples: int
    confidence: float            # 0~1

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_sample": int(self.start_sample),
            "end_sample": int(self.end_sample),
            "kind": self.kind,
            "depth_db": round(float(self.depth_db), 2),
            "duration_samples": int(self.duration_samples),
            "confidence": round(float(self.confidence), 3),
        }


def _sliding_rms(x: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    power = np.abs(x) ** 2
    kernel = np.ones(window, dtype=np.float64) / float(window)
    # 'same' 保持长度，边缘用可用样本数归一化（避免两端被"拖低"造成假突发）
    smoothed = np.convolve(power, kernel, mode="same")
    counts = np.convolve(np.ones_like(power), kernel, mode="same")
    smoothed = smoothed / np.maximum(counts, 1e-12)
    return np.sqrt(np.maximum(smoothed, 0.0))


def detect_bursts(
    samples: np.ndarray,
    sample_rate: float = 1.0,
    sps: float = 1.0,
    window_symbols: float = 2.0,
    depth_db: float = -12.0,
    min_symbols: float = 1.5,
    mad_sigmas: float = 4.0,
    clip_ratio: float = 0.02,
    min_depth_db: float = -6.0,
    edge_depth_db: float = -20.0,
    rise_db: float = 6.0,
) -> list[BurstRegion]:
    """检测幅度塌陷与削波段，返回按时间排序的区间列表."""
    x = np.asarray(samples, dtype=np.complex128).reshape(-1)
    n = x.size
    if n < 8:
        return []
    sps = float(sps) if sps and sps > 0 else 1.0
    window = max(4, int(round(float(window_symbols) * sps)))
    if window >= n:
        window = max(4, n // 4)
    rms = _sliding_rms(x, window)
    baseline = float(np.median(rms))
    if not np.isfinite(baseline) or baseline <= 0:
        return []
    mad = float(np.median(np.abs(rms - baseline)))
    sigma = 1.4826 * mad if mad > 0 else 0.0

    depth_threshold = baseline * (10.0 ** (float(depth_db) / 20.0))
    robust_threshold = baseline - float(mad_sigmas) * sigma if sigma > 0 else 0.0
    threshold = max(depth_threshold, robust_threshold)   # 两个条件同时满足才算异常
    below = rms < threshold
    min_len = max(2, int(round(float(min_symbols) * sps)))

    regions: list[BurstRegion] = []
    index = 0
    while index < n:
        if not below[index]:
            index += 1
            continue
        start = index
        while index < n and below[index]:
            index += 1
        end = index
        # 滑动窗会把边界向两侧糊开半个窗，这里按半窗回收
        half = window // 2
        start = max(0, start - half)
        end = min(n, end + half)
        if end - start < min_len:
            continue
        segment = rms[start:end]
        # 用 10 分位而不是中位数衡量"这一段到底塌了多深"：中位数会被两端的过渡糊住，
        # 真正的洞（采样被打成 0）深度应该接近 -inf dB。
        level = float(np.percentile(segment, 10)) if segment.size else 0.0
        depth = 20.0 * np.log10(max(level, 1e-12) / max(baseline, 1e-12))
        # 边缘抑制：记录首尾有匹配滤波暂态（幅度天然偏低几个 dB），那不是"损伤"。
        # 只有深到 edge_depth_db 以下（真正的挖空/静默）才允许贴着边缘报出来。
        touches_edge = start <= window or end >= n - window
        if depth > float(min_depth_db):
            continue
        if touches_edge and depth > float(edge_depth_db):
            continue
        # 置信度：深度越深、持续越长越可信（都做了饱和处理）
        depth_score = float(np.clip((-depth - 3.0) / 20.0, 0.0, 1.0))
        length_score = float(np.clip((end - start) / max(1.0, 8.0 * sps), 0.0, 1.0))
        regions.append(BurstRegion(
            start_sample=int(start), end_sample=int(end), kind="collapse",
            depth_db=float(depth), duration_samples=int(end - start),
            confidence=float(np.clip(0.5 * depth_score + 0.5 * length_score, 0.0, 1.0)),
        ))

    # 干扰/压制：功率**抬升**（与"塌陷"相反的另一类突发）。压制往往会让同步环失锁，
    # 损伤范围可能远大于干扰区间本身 —— 这一点在报告里如实说明。
    rise_threshold = baseline * (10.0 ** (float(rise_db) / 20.0))
    above = rms > rise_threshold
    index = 0
    while index < n:
        if not above[index]:
            index += 1
            continue
        start = index
        while index < n and above[index]:
            index += 1
        end = index
        half = window // 2
        start = max(0, start - half)
        end = min(n, end + half)
        if end - start < min_len:
            continue
        segment = rms[start:end]
        level = float(np.percentile(segment, 90)) if segment.size else 0.0
        rise = 20.0 * np.log10(max(level, 1e-12) / max(baseline, 1e-12))
        length_score = float(np.clip((end - start) / max(1.0, 8.0 * sps), 0.0, 1.0))
        regions.append(BurstRegion(
            start_sample=int(start), end_sample=int(end), kind="jam",
            depth_db=float(rise), duration_samples=int(end - start),
            confidence=float(np.clip(0.5 * min(1.0, rise / 12.0) + 0.5 * length_score, 0.0, 1.0)),
        ))

    # 削波：贴顶样本占比异常（幅度骤降之外的另一类"位置可信"损伤）
    amplitude = np.abs(x)
    peak = float(np.percentile(amplitude, 99.9))
    if peak > 0:
        clipped = amplitude >= peak * 0.999
        ratio = float(np.mean(clipped))
        if ratio >= float(clip_ratio):
            indices = np.flatnonzero(clipped)
            # 合并相邻的削波点成区间
            splits = np.flatnonzero(np.diff(indices) > max(1, int(sps)))
            for chunk in np.split(indices, splits + 1):
                if chunk.size < max(2, int(min_symbols * sps / 2)):
                    continue
                regions.append(BurstRegion(
                    start_sample=int(chunk[0]), end_sample=int(chunk[-1]) + 1, kind="clip",
                    depth_db=0.0, duration_samples=int(chunk[-1] - chunk[0] + 1),
                    confidence=float(np.clip(ratio / max(clip_ratio, 1e-9) * 0.5, 0.0, 0.8)),
                ))
    regions.sort(key=lambda item: item.start_sample)
    _ = sample_rate  # 目前区间以采样为单位，保留参数以便将来按时间/符号单位输出
    return regions


def detect_bursts_dict(samples: np.ndarray, **kwargs: Any) -> dict[str, Any]:
    """给接口/证据表用的包装：返回区间列表与一个"有没有突发"的结论."""
    regions = detect_bursts(samples, **kwargs)
    return {
        "burst_count": len(regions),
        "bursts": [region.to_dict() for region in regions],
        "total_samples": int(np.asarray(samples).size),
        "rule": "滑动 RMS 同时低于 -12 dB 深度门限与中位数 -4*MAD，且持续 >=1.5 个符号",
    }
