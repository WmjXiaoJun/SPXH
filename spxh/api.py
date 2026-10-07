"""服务层：CLI 与前端共用的稳定 JSON 接口.

接口定义见 docs/前端接口约定-v1.md —— 前端只认这份契约，任何字段变更必须先改文档。
路径安全：所有 path 参数都会被解析并限制在工作区内（越界抛 PermissionError）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from spxh import __version__
from spxh.core.classify import ModulationClassifier, classify_signal
from spxh.core.classify.model import DEFAULT_MODEL_DIR
from spxh.core.demod import demodulate
from spxh.core.agent import agent_tools_payload, run_agent
from spxh.core.explain import compute_ssca
from spxh.core.report import (
    narrate,
    list_models as llm_list_models,
    masked_config,
    save_config as save_llm_config,
    test_connection as llm_test_connection,
)
from spxh.core.framesync import extract_frames
from spxh.core.dsp import stft_waterfall, welch_psd
from spxh.core.dsp.spectrum import smooth_psd
from spxh.core.estimate import analyze_signal, compare_with_truth
from spxh.core.features import FeatureConfig, extract_features
from spxh.core.features.constellation import constellation_features, recover_symbols
from spxh.core.io import load_signal
from spxh.core.types import GroundTruth, jsonable

__all__ = [
    "WORKSPACE",
    "resolve_path",
    "health",
    "list_cases",
    "analyze_path",
    "spectrum_path",
    "waterfall_path",
    "constellation_path",
    "features_path",
    "classify_path",
    "demod_path",
    "fec_report",
    "frame_path",
    "ssca_path",
    "narrate_path",
    "llm_config",
    "llm_models",
    "llm_test",
    "agent_tools",
    "agent_run",
]

# spxh/api.py -> parents[0]=spxh, parents[1]=仓库根
WORKSPACE = Path(__file__).resolve().parents[1]
ESTIMATOR_TAG = "spxh-m1"


def _root() -> Path:
    import os

    env = os.environ.get("SPXH_WORKSPACE")
    return Path(env).resolve() if env else WORKSPACE


def resolve_path(path: str) -> Path:
    """把前端传来的路径解析为工作区内的绝对路径（禁止越界）."""
    root = _root()
    candidate = Path(str(path))
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError("路径越界：{} 不在工作区 {} 内".format(resolved, root)) from exc
    if not resolved.exists():
        raise FileNotFoundError("文件不存在：" + str(resolved))
    return resolved


def _truth_for(path: Path) -> Optional[GroundTruth]:
    name = path.name.lower()
    if name.endswith(".sigmf-meta"):
        prefix = path.with_name(path.name[: -len(".sigmf-meta")])
    elif name.endswith(".spxh.npz"):
        prefix = path.with_name(path.name[: -len(".spxh.npz")])
    else:
        prefix = path.with_suffix("")
    if Path(str(prefix) + ".truth.json").exists():
        return GroundTruth.load(prefix)
    return None


def health(model_dir=DEFAULT_MODEL_DIR) -> dict[str, Any]:
    classifier = ModulationClassifier.load(model_dir)
    return {
        "ok": True,
        "version": __version__,
        "estimator": ESTIMATOR_TAG,
        "classifier": None if classifier is None else "spxh-rf-v1",
        "classes": [] if classifier is None else list(classifier.metadata.classes),
        "workspace": str(_root()),
    }


def list_cases(directory: str = "data") -> dict[str, Any]:
    root = _root()
    base = resolve_path(directory) if (root / str(directory)).exists() else None
    cases: list[dict[str, Any]] = []
    if base is not None:
        for meta_path in sorted(base.rglob("*.sigmf-meta")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                global_section = meta.get("global", {})
                prefix = meta_path.with_name(meta_path.name[: -len(".sigmf-meta")])
                data_path = Path(str(prefix) + ".sigmf-data")
                truth_path = Path(str(prefix) + ".truth.json")
                truth = None
                if truth_path.exists():
                    truth = json.loads(truth_path.read_text(encoding="utf-8"))
                num_samples = int(data_path.stat().st_size // 8) if data_path.exists() else 0
                sample_rate = float(global_section.get("core:sample_rate", 0.0))
                cases.append(
                    {
                        "name": prefix.name,
                        "prefix": str(prefix.relative_to(root)).replace("\\", "/"),
                        "modulation": (truth or {}).get("modulation"),
                        "snr_db": (truth or {}).get("snr_db"),
                        "sample_rate": sample_rate,
                        "num_samples": num_samples,
                        "duration_s": float(num_samples / sample_rate) if sample_rate else 0.0,
                        "has_truth": truth is not None,
                        "files": {
                            "sigmf_meta": str(meta_path.relative_to(root)).replace("\\", "/"),
                            "sigmf_data": str(data_path.relative_to(root)).replace("\\", "/") if data_path.exists() else "",
                            "truth_json": str(truth_path.relative_to(root)).replace("\\", "/") if truth_path.exists() else "",
                        },
                    }
                )
            except Exception:  # noqa: BLE001 - 单个坏文件不影响列表
                continue
    cases.sort(key=lambda c: (str(c.get("modulation") or ""), float(c.get("snr_db") or 0.0), c["name"]))
    return {"dir": str(directory), "cases": cases}


def analyze_path(path: str, nperseg: int = 1024, nfft: Optional[int] = None) -> dict[str, Any]:
    target = resolve_path(path)
    signal = load_signal(target)
    result = analyze_signal(signal, nperseg=int(nperseg), nfft=nfft)
    payload = result.to_dict()
    payload["path"] = str(target.relative_to(_root())).replace("\\", "/")
    truth = _truth_for(target)
    payload["comparison"] = compare_with_truth(result, truth) if truth is not None else None
    return jsonable(payload)


def spectrum_path(path: str, nperseg: int = 1024, nfft: Optional[int] = None, points: int = 1200) -> dict[str, Any]:
    target = resolve_path(path)
    signal = load_signal(target)
    psd = welch_psd(signal, nperseg=int(nperseg), nfft=nfft)
    floor = psd.noise_floor()
    peak = psd.peak()
    band = psd.occupied_band(noise_floor=floor)

    residual_linear = np.maximum(smooth_psd(psd.psd, 9) - floor, 0.0)
    peak_residual = float(np.max(residual_linear)) or 1.0
    residual_db = 10.0 * np.log10(np.maximum(residual_linear / peak_residual, 1e-6))
    psd_db = 10.0 * np.log10(np.maximum(psd.psd, 1e-300))

    grid = np.linspace(float(psd.freqs[0]), float(psd.freqs[-1]), int(max(64, points)))
    freqs_out = grid.tolist()
    psd_out = np.interp(grid, psd.freqs, psd_db).tolist()
    residual_out = np.interp(grid, psd.freqs, residual_db).tolist()

    estimate = analyze_signal(signal, psd=psd)
    truth = _truth_for(target)
    return jsonable(
        {
            "path": str(target.relative_to(_root())).replace("\\", "/"),
            "sample_rate": signal.sample_rate,
            "freqs": freqs_out,
            "psd_db": psd_out,
            "residual_db": residual_out,
            "noise_floor_db": float(10.0 * np.log10(max(floor, 1e-300))),
            "peak": {"frequency_hz": float(peak["frequency"]), "snr_db": float(peak["snr_db"])},
            "occupied": band,
            "cfo": {"value": float(estimate.cfo.value), "confidence": float(estimate.cfo.confidence)},
            "cfo_true": None if truth is None else float(truth.cfo_hz),
        }
    )


def waterfall_path(path: str, nperseg: int = 256, max_frames: int = 400, max_bins: int = 192) -> dict[str, Any]:
    target = resolve_path(path)
    signal = load_signal(target)
    waterfall = stft_waterfall(signal, nperseg=int(nperseg))
    magnitude = np.asarray(waterfall.magnitude_db)
    bins, frames = magnitude.shape
    bin_stride = max(1, int(np.ceil(bins / max(1, int(max_bins)))))
    frame_stride = max(1, int(np.ceil(frames / max(1, int(max_frames)))))
    reduced = magnitude[::bin_stride, ::frame_stride]
    freqs = waterfall.freqs[::bin_stride]
    times = waterfall.times[::frame_stride]
    return jsonable(
        {
            "path": str(target.relative_to(_root())).replace("\\", "/"),
            "times": times.tolist(),
            "freqs": freqs.tolist(),
            "magnitude_db": reduced.tolist(),
            "min_db": float(np.min(reduced)),
            "max_db": float(np.max(reduced)),
            "shape": [int(reduced.shape[0]), int(reduced.shape[1])],
        }
    )


def constellation_path(
    path: str,
    symbol_rate: Optional[float] = None,
    cfo: Optional[float] = None,
    max_points: int = 3000,
    nperseg: int = 1024,
) -> dict[str, Any]:
    target = resolve_path(path)
    signal = load_signal(target)
    estimate = analyze_signal(signal, nperseg=int(nperseg)) if (symbol_rate is None or cfo is None) else None
    symbol_rate_used = float(symbol_rate if symbol_rate is not None else estimate.symbol_rate.value)
    cfo_used = float(cfo if cfo is not None else estimate.cfo.value)
    points = recover_symbols(signal, cfo_hz=cfo_used, symbol_rate=symbol_rate_used, max_symbols=int(max_points))
    stats = constellation_features(points, symbol_rate_used)
    return jsonable(
        {
            "path": str(target.relative_to(_root())).replace("\\", "/"),
            "i": points.real.tolist(),
            "q": points.imag.tolist(),
            "count": int(points.size),
            "evm_percent": stats["const_evm_percent"],
            "cluster_count": stats["const_cluster_count"],
            "radial_spread": stats["const_radial_spread"],
            "symbol_rate": symbol_rate_used,
            "cfo_hz": cfo_used,
            "source": "provided" if (symbol_rate is not None and cfo is not None) else "estimated",
        }
    )


def features_path(path: str, nperseg: int = 1024) -> dict[str, Any]:
    target = resolve_path(path)
    signal = load_signal(target)
    feature_set = extract_features(signal, config=FeatureConfig(nperseg=int(nperseg)))
    payload = feature_set.as_dict()
    payload["path"] = str(target.relative_to(_root())).replace("\\", "/")
    payload["symbol_rate"] = float(feature_set.evidence.get("symbol_rate", 0.0))
    payload["cfo_hz"] = float(feature_set.evidence.get("cfo_hz", 0.0))
    payload["snr_db"] = float(feature_set.evidence.get("es_n0_db", 0.0))
    return jsonable(payload)


def demod_path(
    path: str,
    symbol_rate: Optional[float] = None,
    cfo: Optional[float] = None,
    modulation: Optional[str] = None,
    max_symbols: int = 2000,
    trace_points: int = 400,
    model_dir=DEFAULT_MODEL_DIR,
    nperseg: int = 1024,
) -> dict[str, Any]:
    """M3 解调：返回判决符号、比特、LLR、环路锁定状态与三条同步轨迹（供前端画图）."""
    target = resolve_path(path)
    signal = load_signal(target)
    analysis = analyze_signal(signal, nperseg=int(nperseg))
    if symbol_rate is not None and symbol_rate > 0:
        analysis.symbol_rate.value = float(symbol_rate)
    if cfo is not None:
        analysis.cfo.value = float(cfo)
    truth = _truth_for(target)
    result = demodulate(
        signal,
        analysis=analysis,
        modulation=modulation,
        modulation_source="provided" if modulation else "classifier",
        truth=truth,
        model_dir=model_dir,
    )
    payload = result.to_dict(max_symbols=int(max_symbols), trace_points=int(trace_points))
    payload["path"] = str(target.relative_to(_root())).replace("\\", "/")
    return jsonable(payload)


def agent_tools() -> dict[str, Any]:
    """代理可用工具与预算（前端渲染工具清单与预算显示）."""
    return jsonable(agent_tools_payload())


def agent_run(
    path: str,
    goal: str = "",
    provider: str = "auto",
    max_rounds: Optional[int] = None,
    max_tool_calls: Optional[int] = None,
    timeout_s: Optional[float] = None,
) -> dict[str, Any]:
    """M11 工具调用代理：有界循环 + 结构化工具结果 + 数值落地检查."""
    target = resolve_path(path)
    payload = run_agent(
        str(target),
        goal=str(goal or ""),
        provider=str(provider or "auto"),
        max_rounds=max_rounds,
        max_tool_calls=max_tool_calls,
        timeout_s=timeout_s,
    )
    payload["path"] = str(target.relative_to(_root())).replace("\\", "/")
    return jsonable(payload)


def llm_config(update: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """读取/保存 LLM 配置（**永远脱敏**：只回显密钥尾 4 位）.

    空 body / 只有空字段时**当 no-op**：不能因为"探测式请求"就把默认配置物化落盘
    （前端 teammate 联调时踩到过：source 从 default 变成了 file）。
    """
    change = dict(update or {})
    has_field = any(
        key not in ("clear_api_key",) and value not in (None, "")
        for key, value in change.items()
    )
    if has_field or change.get("clear_api_key"):
        save_llm_config(change)
    return masked_config()


def llm_models() -> dict[str, Any]:
    """拉取该服务商可用的模型列表（失败也返回 200 + error 文案）."""
    return llm_list_models()


def llm_test(overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """测试 LLM 连通性（临时覆盖不落盘；失败返回 ok=false 而不是抛错）."""
    allowed = {"base_url", "api_key", "model", "temperature", "max_tokens", "timeout_s"}
    clean = {k: v for k, v in (overrides or {}).items() if k in allowed and v not in (None, "")}
    return llm_test_connection(**clean)


def narrate_path(
    path: str,
    provider: str = "auto",
    max_tokens: int = 1200,
) -> dict[str, Any]:
    """M9 分析报告：把结构化证据交给人话（LLM 可选，离线确定性回退）."""
    target = resolve_path(path)
    payload = narrate(str(target), provider=str(provider or "auto"), max_tokens=int(max_tokens))
    payload["path"] = str(target.relative_to(_root())).replace("\\", "/")
    payload["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return jsonable(payload)


def ssca_path(
    path: str,
    nfft: int = 256,
    max_points: int = 160,
    nperseg: int = 1024,
) -> dict[str, Any]:
    """M8 谱相关（SSCA）双频平面：alpha 轴给符号速率、f 轴脊线给载频偏移（解释性证据）."""
    target = resolve_path(path)
    signal = load_signal(target)
    analysis = analyze_signal(signal, nperseg=int(nperseg))
    symbol_rate_hint = float(analysis.symbol_rate.value) if np.isfinite(analysis.symbol_rate.value) else None
    result = compute_ssca(
        signal.samples,
        signal.sample_rate,
        nfft=int(nfft),
        symbol_rate_hint=symbol_rate_hint,
    )
    payload = result.to_dict(max_points=int(max_points))
    payload["path"] = str(target.relative_to(_root())).replace("\\", "/")
    payload["symbol_rate"] = symbol_rate_hint
    payload["estimate"] = {
        "symbol_rate_hz": float(analysis.symbol_rate.value) if np.isfinite(analysis.symbol_rate.value) else None,
        "cfo_hz": float(analysis.cfo.value) if np.isfinite(analysis.cfo.value) else None,
        "snr_db": float(analysis.snr.value) if np.isfinite(analysis.snr.value) else None,
    }
    return jsonable(payload)


def frame_path(
    path: str,
    modulation: Optional[str] = None,
    symbol_rate: Optional[float] = None,
    cfo: Optional[float] = None,
    max_frames: int = 32,
    model_dir=DEFAULT_MODEL_DIR,
    nperseg: int = 1024,
    blind: bool = False,
    erasure_decoding: bool = False,
) -> dict[str, Any]:
    """M5 帧同步与载荷提取：Barker-13 定位帧起点、解相位模糊、解析帧头、校验 CRC、取载荷.

    erasure_decoding=True 时，RS 类编码会用软信息（按符号的 |LLR| 门限）自动标出擦除位置，
    走"误差 + 擦除"联合译码（能力从 t 扩到 2t 量级）。

    blind=True 时，若按帧头声明的编码解不出来，就对全部 (编码, 交织) 假设做检验
    （重写帧头里的 fec 位再算 CRC），用于帧头字段本身被打错的场景。
    """
    target = resolve_path(path)
    signal = load_signal(target)
    analysis = analyze_signal(signal, nperseg=int(nperseg))
    if symbol_rate is not None and symbol_rate > 0:
        analysis.symbol_rate.value = float(symbol_rate)
    if cfo is not None:
        analysis.cfo.value = float(cfo)
    truth = _truth_for(target)
    result = extract_frames(
        signal,
        analysis=analysis,
        modulation=modulation,
        modulation_source="provided" if modulation else "classifier",
        truth=truth,
        max_frames=int(max_frames),
        model_dir=model_dir,
        blind_fec=bool(blind),
        erasure_decoding=bool(erasure_decoding),
    )
    payload = result.to_dict()
    payload["path"] = str(target.relative_to(_root())).replace("\\", "/")
    payload["blind"] = bool(blind)
    payload["erasure_decoding"] = bool(erasure_decoding)
    return jsonable(payload)


DEFAULT_FEC_REPORT = "models/check_m4_report.json"


def fec_report(report_path: str = DEFAULT_FEC_REPORT) -> dict[str, Any]:
    """M4 译码报告：直接读取 scripts/check_m4.py 落盘的结果（前端"译码"页数据源）."""
    target = resolve_path(report_path)
    if not target.exists():
        raise FileNotFoundError("译码报告不存在，请先运行 python scripts/check_m4.py")
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload.setdefault("report_path", str(report_path).replace("\\", "/"))
    return jsonable(payload)


def classify_path(path: str, model_dir=DEFAULT_MODEL_DIR, nperseg: int = 1024, threshold: Optional[float] = None) -> dict[str, Any]:
    target = resolve_path(path)
    signal = load_signal(target)
    payload = classify_signal(
        signal,
        model_dir=model_dir,
        feature_config=FeatureConfig(nperseg=int(nperseg)),
        threshold=threshold,
    )
    payload["path"] = str(target.relative_to(_root())).replace("\\", "/")
    truth = _truth_for(target)
    payload["truth"] = None if truth is None else {"modulation": truth.modulation, "snr_db": float(truth.snr_db)}
    return jsonable(payload)
