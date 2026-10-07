"""LLM 分析报告：把整条链路的**结构化证据**交给 LLM 讲成人话（没有 LLM 时走确定性渲染）.

设计原则（这是"和 LLM 结合"的关键，不是把 I/Q 丢给模型）
--------------------------------------------------------
1. **LLM 不做数值估计**：所有数字都来自确定性管线（M1 估计 / M2 分类 / M5 帧同步 /
   M8 谱相关），每个值都带 source / unit / confidence。模型只负责**解释、整合、找矛盾、
   给下一步建议**。让模型直接看波形去"估符号速率"必然产生幻觉。
2. **结构化证据 -> 提示词 -> 结构化报告**：提示词里是紧凑的 JSON 证据表，并要求模型
   只允许引用给定字段的数字（缺就写 null）。返回的报告结构与离线模式完全一致。
3. **离线确定性回退**：没有 key / 网络 / provider 时，用同一份证据做确定性中文渲染，
   结构一模一样 —— 平台不能因为 key 失效就不可用。
4. **可审计**：接口会回传实际发出去的提示词（prompt_preview），报告里每条结论都指向证据。
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from spxh.core.report.llm_config import (
    LLMConfig,
    chat_completion,
    list_models as llm_list_models,
    load_config as load_llm_config,
    masked_config,
    save_config as save_llm_config,
    test_connection as llm_test_connection,
)
from spxh.core.types import jsonable

__all__ = [
    "collect_evidence",
    "render_offline",
    "build_prompt",
    "narrate",
    "call_llm",
    "llm_available",
    "llm_info",
    "check_numeric_grounding",
    "collect_evidence_numbers",
    "DEFAULT_MODEL",
]

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
ENV_BASE = "SPXH_LLM_BASE_URL"
ENV_KEY = "SPXH_LLM_API_KEY"
ENV_MODEL = "SPXH_LLM_MODEL"


def llm_available(**overrides: Any) -> bool:
    """是否具备调用 LLM 的条件（配置开关 + 密钥齐全；只读配置，不联网）."""
    try:
        return bool(load_llm_config(**overrides).ready)
    except Exception:  # noqa: BLE001 - 配置坏掉时按"不可用"处理，报告仍能产出
        return False


def llm_info(**overrides: Any) -> dict[str, Any]:
    """给报告用的 LLM 元信息（脱敏，绝不含密钥）."""
    cfg = load_llm_config(**overrides)
    return {
        "base_url": cfg.base_url,
        "model": cfg.model,
        "api_key_hint": masked_config(cfg)["api_key_hint"],
        "enabled": bool(cfg.enabled),
        "ready": bool(cfg.ready),
    }


def _evidence(key: str, value: Any, source: str, unit: str = "", confidence: Optional[float] = None, note: str = "") -> dict[str, Any]:
    item: dict[str, Any] = {"key": key, "value": jsonable(value), "source": source}
    if unit:
        item["unit"] = unit
    if confidence is not None:
        item["confidence"] = float(confidence)
    if note:
        item["note"] = note
    return item


def _confidence_of(estimate) -> Optional[float]:
    try:
        value = float(getattr(estimate, "confidence", float("nan")))
        return value if np.isfinite(value) else None
    except Exception:  # noqa: BLE001
        return None


def collect_evidence(
    path: str,
    signal=None,
    analysis=None,
    classification: Optional[dict] = None,
    frames=None,
    ssca=None,
    extra: Optional[dict] = None,
) -> dict[str, Any]:
    """跑（或复用）整条管线，产出**纯数据**的证据表；不生成任何自然语言."""
    return _collect(path, signal, analysis, classification, frames, ssca, extra)


def _collect(path, signal, analysis, classification, frames, ssca, extra) -> dict[str, Any]:
    from spxh.core.classify import classify_signal
    from spxh.core.estimate import analyze_signal
    from spxh.core.explain import compute_ssca
    from spxh.core.framesync import extract_frames
    from spxh.core.io import load_signal

    if signal is None:
        signal = load_signal(path)
    if analysis is None:
        analysis = analyze_signal(signal)
    if classification is None:
        try:
            classification = classify_signal(signal, estimates=analysis)
        except Exception as exc:  # noqa: BLE001 - 分类失败不该让报告整体失败
            classification = {"error": "{}: {}".format(type(exc).__name__, exc)}
    modulation = None
    if isinstance(classification, dict) and classification.get("modulation"):
        modulation = str(classification["modulation"])
    if frames is None:
        try:
            frames = extract_frames(signal, analysis=analysis, modulation=modulation)
        except Exception as exc:  # noqa: BLE001
            frames = {"error": "{}: {}".format(type(exc).__name__, exc)}
    if ssca is None:
        try:
            ssca = compute_ssca(
                signal.samples, signal.sample_rate, nfft=256,
                symbol_rate_hint=float(analysis.symbol_rate.value) if np.isfinite(analysis.symbol_rate.value) else None,
            )
        except Exception as exc:  # noqa: BLE001
            ssca = {"error": "{}: {}".format(type(exc).__name__, exc)}

    evidence: dict[str, Any] = {
        "path": str(path),
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "signal": [
            _evidence("sample_rate", float(signal.sample_rate), "M0", "Hz"),
            _evidence("num_samples", int(signal.num_samples), "M0"),
            _evidence("duration_s", round(float(signal.duration), 4), "M0", "s"),
            _evidence("snr_db", float(analysis.snr.value), "M1", "dB", _confidence_of(analysis.snr),
                      str(getattr(analysis.snr, "method", "") or "")),
            _evidence("cfo_hz", float(analysis.cfo.value), "M1", "Hz", _confidence_of(analysis.cfo),
                      str(getattr(analysis.cfo, "method", "") or "")),
            _evidence("symbol_rate_hz", float(analysis.symbol_rate.value), "M1", "Hz",
                      _confidence_of(analysis.symbol_rate), str(getattr(analysis.symbol_rate, "method", "") or "")),
        ],
        "classification": [],
        "frame": [],
        "cyclostationary": [],
        "pipeline_errors": [],
    }
    for name, payload in (("classification", classification), ("frame", frames), ("cyclostationary", ssca)):
        if isinstance(payload, dict) and payload.get("error"):
            evidence["pipeline_errors"].append({"stage": name, "error": payload["error"]})

    if isinstance(classification, dict) and classification.get("modulation"):
        probabilities = classification.get("probabilities") or {}
        best = None
        if isinstance(probabilities, dict) and probabilities:
            best = max(probabilities.items(), key=lambda kv: float(kv[1]))
        evidence["classification"].append(
            _evidence("modulation", classification.get("modulation"), "M2",
                      confidence=float(classification.get("confidence", float("nan")))
                      if classification.get("confidence") is not None else None)
        )
        evidence["classification"].append(_evidence("gate", classification.get("gate"), "M2",
                                                    note="ml=机器学习判决 / fallback=物理规则回退 / reject=拒绝作答"))
        if best is not None:
            evidence["classification"].append(_evidence("top_probability", round(float(best[1]), 4), "M2",
                                                        note="最可能的类别：" + str(best[0])))
        if classification.get("ood_score") is not None:
            evidence["classification"].append(_evidence("ood_score", classification.get("ood_score"), "M2",
                                                        note="越大越像分布外样本"))
        if classification.get("physical"):
            evidence["classification"].append(_evidence("physical_rule", classification.get("physical"), "M2",
                                                        note="不依赖训练数据的物理判据"))

    if hasattr(frames, "summary"):
        summary = frames.summary
        lock = frames
        evidence["frame"].append(_evidence("correlation_peak", round(float(lock.correlation_peak), 4), "M5"))
        evidence["frame"].append(_evidence("peak_to_sidelobe_db", round(float(lock.peak_to_sidelobe_db), 2), "M5",
                                          "dB", note="峰值/中位数的 dB 比（峰显著性）"))
        evidence["frame"].append(_evidence("ambiguity_rotation_deg", float(lock.ambiguity_rotation_deg), "M5",
                                           "deg", note="由 Barker-13 前导解出的相位模糊"))
        evidence["frame"].append(_evidence("frames", int(summary.get("frames", 0)), "M5"))
        evidence["frame"].append(_evidence("frames_crc_ok", int(summary.get("frames_crc_ok", 0)), "M5"))
        evidence["frame"].append(_evidence("payload_bytes_total", int(summary.get("payload_bytes_total", 0)), "M5", "B"))
        if summary.get("payload_bytes_exact") is not None:
            evidence["frame"].append(_evidence("payload_bytes_exact", int(summary["payload_bytes_exact"]), "M5", "B",
                                               note="与真值逐字节一致的数量"))

    if hasattr(ssca, "symbol_rate_hz"):
        if ssca.symbol_rate_hz is not None:
            evidence["cyclostationary"].append(_evidence("alpha_peak_hz", float(ssca.symbol_rate_hz), "M8", "Hz",
                                                         note="alpha 轴最强峰 ≈ 符号速率"))
        if ssca.cfo_hz is not None:
            evidence["cyclostationary"].append(_evidence("ridge_offset_hz", float(ssca.cfo_hz), "M8", "Hz",
                                                         note="alpha!=0 脊线的 f 位置 ≈ 载频偏移（分辨率 fs/nfft）"))
        if ssca.peaks:
            evidence["cyclostationary"].append(
                _evidence("top_peaks", [[round(p["alpha_hz"], 1), round(p["freq_hz"], 1), round(p["level_db"], 1)]
                                        for p in ssca.peaks[:4]], "M8", note="[alpha Hz, f Hz, dB]")
            )
    if extra:
        evidence["extra"] = jsonable(extra)
    return evidence


def _score_confidence(evidence: dict[str, Any]) -> tuple[float, list[str]]:
    """把各环节的置信度/门控/CRC 通过率合成一个 0~1 的可信度，并列出不确定项."""
    uncertainties: list[str] = []
    scores: list[float] = []
    for item in evidence.get("signal", []):
        if item["key"] in ("snr_db", "symbol_rate_hz", "cfo_hz") and item.get("confidence") is not None:
            value = float(item["confidence"])
            scores.append(max(0.0, min(1.0, value)))
            if value < 0.35:
                uncertainties.append("{} 的估计置信度偏低（{:.2f}）".format(item["key"], value))
    gate = None
    for item in evidence.get("classification", []):
        if item["key"] == "gate":
            gate = str(item["value"])
        if item["key"] == "ood_score":
            uncertainties.append("OOD 分数 {:.2f}：样本可能超出训练分布".format(float(item["value"])))
    if gate == "ml":
        scores.append(0.9)
    elif gate == "fallback":
        scores.append(0.5)
        uncertainties.append("调制识别走了物理规则回退（未用机器学习判决）")
    elif gate == "reject":
        scores.append(0.2)
        uncertainties.append("调制识别拒绝给出结论")
    frames = {item["key"]: item["value"] for item in evidence.get("frame", [])}
    total = int(frames.get("frames", 0) or 0)
    ok = int(frames.get("frames_crc_ok", 0) or 0)
    if total > 0:
        ratio = ok / total
        scores.append(float(ratio))
        if ratio < 1.0:
            uncertainties.append("有 {} 帧 CRC 未通过（{}/{})".format(total - ok, ok, total))
    else:
        uncertainties.append("没有解析出任何帧（同步失败或记录太短）")
    for error in evidence.get("pipeline_errors", []):
        uncertainties.append("{} 环节报错：{}".format(error.get("stage"), str(error.get("error"))[:80]))
    confidence = float(np.mean(scores)) if scores else 0.0
    return confidence, uncertainties


def render_offline(evidence: dict[str, Any]) -> dict[str, Any]:
    """确定性中文渲染：与 LLM 模式**同结构**，同样输入必然同样输出."""
    def value_of(section: str, key: str, default=None):
        for item in evidence.get(section, []):
            if item["key"] == key:
                return item["value"]
        return default

    def unit_of(section: str, key: str) -> str:
        for item in evidence.get(section, []):
            if item["key"] == key:
                return str(item.get("unit", ""))
        return ""

    confidence, uncertainties = _score_confidence(evidence)
    modulation = value_of("classification", "modulation")
    snr = value_of("signal", "snr_db")
    symbol_rate = value_of("signal", "symbol_rate_hz")
    frames = value_of("frame", "frames")
    crc_ok = value_of("frame", "frames_crc_ok")
    alpha = value_of("cyclostationary", "alpha_peak_hz")

    headline_bits = []
    if modulation:
        headline_bits.append(str(modulation).upper())
    if snr is not None:
        headline_bits.append("Es/N0 约 {:.1f} dB".format(float(snr)))
    if symbol_rate is not None:
        headline_bits.append("符号速率 {:.1f} Hz".format(float(symbol_rate)))
    if frames:
        headline_bits.append("帧同步锁定（{}/{} 帧 CRC 通过）".format(int(crc_ok or 0), int(frames)))
    headline = "，".join(headline_bits) if headline_bits else "证据不足，无法给出结论"

    sections: list[dict[str, Any]] = []
    signal_text = "采样率 {} Hz，时长 {} s。".format(
        value_of("signal", "sample_rate"), value_of("signal", "duration_s"))
    if snr is not None:
        signal_text += " 盲估计 Es/N0 为 {:.2f} dB，载频偏移 {:.2f} Hz，符号速率 {:.2f} Hz。".format(
            float(snr), float(value_of("signal", "cfo_hz") or 0.0), float(symbol_rate or 0.0))
    if alpha is not None:
        signal_text += " 谱相关在 alpha = {:.1f} Hz 处给出独立佐证。".format(float(alpha))
    sections.append({"id": "signal", "title": "信号与参数", "text": signal_text,
                     "evidence": [item for item in evidence.get("signal", [])]})

    if modulation:
        gate = value_of("classification", "gate")
        gate_text = {"ml": "机器学习判决", "fallback": "物理规则回退（未用模型）", "reject": "拒绝作答"}.get(str(gate), str(gate))
        text = "调制识别结论：{}（{}）。".format(str(modulation).upper(), gate_text)
        top = value_of("classification", "top_probability")
        if top is not None:
            text += " 最可能类别的概率为 {:.3f}。".format(float(top))
        if value_of("classification", "ood_score") is not None:
            text += " OOD 分数 {:.2f}。".format(float(value_of("classification", "ood_score")))
        sections.append({"id": "modulation", "title": "调制识别", "text": text,
                         "evidence": list(evidence.get("classification", []))})

    if frames is not None:
        text = "帧同步：相关峰 {:.3f}，前导解出相位模糊 {:+.1f} 度；解析 {} 帧，{} 帧 CRC 通过，载荷 {} 字节。".format(
            float(value_of("frame", "correlation_peak") or 0.0),
            float(value_of("frame", "ambiguity_rotation_deg") or 0.0),
            int(frames), int(crc_ok or 0), int(value_of("frame", "payload_bytes_total") or 0))
        if value_of("frame", "payload_bytes_exact") is not None:
            text += " 与真值逐字节一致 {} 字节。".format(int(value_of("frame", "payload_bytes_exact")))
        sections.append({"id": "frame", "title": "帧同步与载荷", "text": text,
                         "evidence": list(evidence.get("frame", []))})

    if evidence.get("cyclostationary"):
        peaks = value_of("cyclostationary", "top_peaks") or []
        text = "谱相关双频平面：alpha 轴峰 {:.1f} Hz（符号速率），脊线偏移 {:.1f} Hz。".format(
            float(alpha or 0.0), float(value_of("cyclostationary", "ridge_offset_hz") or 0.0))
        if peaks:
            text += " 前几个峰（alpha, f, dB）：{}。".format(peaks)
        sections.append({"id": "cyclostationary", "title": "谱相关（可解释性）", "text": text,
                         "evidence": list(evidence.get("cyclostationary", []))})

    next_actions: list[str] = []
    if not frames:
        next_actions.append("换更长/更高信噪比的记录，或显式指定 modulation 与 symbol_rate 重试")
    elif int(crc_ok or 0) < int(frames):
        next_actions.append("对 CRC 未通过的帧打开盲编码识别（--blind / blind=1）重试")
        next_actions.append("若记录里有突发损伤，用置信度标擦除后走 RS 联合译码")
    if value_of("classification", "gate") != "ml":
        next_actions.append("调制识别未走模型：补充同类型样本重训分类器，或人工指定制式")
    if uncertainties:
        next_actions.append("先处理 uncertainties 里列出的低置信环节再下结论")

    return {
        "mode": "offline",
        "model": "offline-rule-based",
        "headline": headline,
        "confidence": round(confidence, 4),
        "sections": sections,
        "uncertainties": uncertainties,
        "next_actions": next_actions,
        "warnings": [],
    }


def build_prompt(evidence: dict[str, Any], max_bytes: int = 6000) -> str:
    """把证据表编成提示词；明确"只许引用给定数值"."""
    compact = json.dumps(evidence, ensure_ascii=False, indent=1)
    if len(compact) > max_bytes:
        compact = compact[:max_bytes] + "\n... (证据表已截断)"
    return (
        "你是射频频谱分析平台 SPXH 的分析员。下面是从确定性信号处理管线得到的**结构化证据**\n"
        "（每个值都带 source=来源环节、可选 unit/confidence）。请据此写一份中文分析报告。\n\n"
        "硬性要求：\n"
        "1. 只允许引用证据表里出现的数字；证据缺失就写 null，**绝不允许编造或推测数值**。\n"
        "2. 每个结论都必须能追溯到某条证据（在 sections[].evidence 里带上用到的 key）。\n"
        "3. 明确指出不确定性与矛盾（例如置信度低、CRC 未通过、门控是 fallback/reject）。\n"
        "4. 只输出 JSON，结构如下：\n"
        '{"headline": "一句话结论", "confidence": 0.0, "sections": [{"id":"signal|modulation|frame|cyclostationary",'
        '"title":"标题","text":"中文分析","evidence":[{"key":"...","value":...,"unit":"...","source":"..."}]}],'
        '"uncertainties": ["..."], "next_actions": ["..."]}\n\n'
        "证据表：\n" + compact
    )


@dataclass
class LLMResult:
    text: str
    model: str


def call_llm(
    prompt: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    config: Optional[LLMConfig] = None,
) -> LLMResult:
    """调用 OpenAI 兼容的 /chat/completions（只用标准库，不引入依赖）.

    未显式给的参数一律取自**本地配置**（models/llm_config.json > 环境变量 > 默认）。
    """
    cfg = config or load_llm_config()
    base = (base_url or cfg.base_url or DEFAULT_BASE_URL).rstrip("/")
    key = (api_key if api_key is not None else cfg.api_key) or ""
    name = model or cfg.model or DEFAULT_MODEL
    timeout = float(timeout if timeout is not None else cfg.timeout_s)
    max_tokens = int(max_tokens if max_tokens is not None else cfg.max_tokens)
    temp = float(temperature if temperature is not None else cfg.temperature)
    if not key:
        raise RuntimeError("未配置 API Key（可在「设置」页或 models/llm_config.json 中填写）")
    body = json.dumps({
        "model": name,
        "messages": [
            {"role": "system", "content": "你是严谨的射频信号分析员，只依据给定证据作答。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": temp,
        "max_tokens": int(max_tokens),
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    # 统一走 llm_config.chat_completion：它负责 JSON 模式的自动降级重试与友好错误
    result = chat_completion(
        [
            {"role": "system", "content": "你是严谨的射频信号分析员，只依据给定证据作答。"},
            {"role": "user", "content": prompt},
        ],
        json_mode=True,
        temperature=temp,
        max_tokens=int(max_tokens),
        timeout=float(timeout),
        config=cfg,
    )
    text = str((result.get("message") or {}).get("content") or "")
    return LLMResult(text=text, model=str(result.get("model") or name))


def narrate(
    path: str,
    provider: str = "auto",
    signal=None,
    analysis=None,
    classification: Optional[dict] = None,
    frames=None,
    ssca=None,
    extra: Optional[dict] = None,
    max_tokens: int = 1200,
    timeout: float = 40.0,
) -> dict[str, Any]:
    """产出分析报告：provider=llm/auto 时尝试调用 LLM，失败/未配置一律降级为离线渲染."""
    evidence = collect_evidence(path, signal=signal, analysis=analysis,
                               classification=classification, frames=frames, ssca=ssca, extra=extra)
    report = render_offline(evidence)
    prompt = build_prompt(evidence)
    report["prompt_preview"] = prompt[:1500]
    report["evidence"] = evidence
    report["warnings"] = list(report.get("warnings") or [])

    mode = str(provider or "auto").lower()
    if mode not in ("llm", "auto"):
        mode = "offline"
    cfg = load_llm_config()
    if mode == "llm" and not cfg.ready:
        reason = "配置里未启用 LLM" if not cfg.enabled else "未配置 API Key"
        report["warnings"].append("provider=llm 但{}，已降级为离线渲染".format(reason))
        mode = "offline"
    if mode == "auto" and not cfg.ready:
        mode = "offline"

    if mode in ("llm", "auto"):
        try:
            result = call_llm(prompt, max_tokens=int(max_tokens), timeout=float(timeout), config=cfg)
            parsed = _parse_llm_payload(result.text, report)
            parsed["mode"] = "llm"
            parsed["model"] = result.model
            parsed["prompt_preview"] = prompt[:1500]
            parsed["evidence"] = evidence
            parsed["llm"] = llm_info()
            grounding = check_numeric_grounding(parsed, evidence)
            parsed["grounding"] = grounding
            extra_warnings = list(parsed.get("warnings") or [])
            if not grounding["grounded"]:
                extra_warnings.append(
                    "LLM 报告里有 {} 个数字无法在证据表中溯源（已标出，请人工复核）：{}".format(
                        len(grounding["unsupported"]),
                        ", ".join(str(item["number"]) for item in grounding["unsupported"][:6]),
                    )
                )
            parsed["warnings"] = report["warnings"] + extra_warnings
            return jsonable(parsed)
        except Exception as exc:  # noqa: BLE001 - LLM 失败绝不 5xx，降级继续
            report["warnings"].append("LLM 调用失败（{}: {}），已降级为离线渲染".format(type(exc).__name__, str(exc)[:160]))
    if mode == "offline":
        # 离线渲染的数字全部来自证据，这里跑一遍同样是"自证"：能把渲染器的 bug 暴露出来
        report["grounding"] = check_numeric_grounding(report, evidence)
    return jsonable(report)


_NUMBER_RE = None


def _numbers_in(text: str) -> list[float]:
    """抽取文本里的数字（含小数、百分号、负号）；用于"LLM 不许编数"的落地检查."""
    global _NUMBER_RE
    import re

    if _NUMBER_RE is None:
        # 只认"独立"的数字：不能紧贴字母/下划线 —— 否则 Es/N0 的 0、16QAM/2FSK/8PSK 里的数字
        # 都会被当成"编造的数据"，检查器就变成噪声源了（这是第一版踩的坑）
        _NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?(?![A-Za-z0-9_])")
    out: list[float] = []
    for token in _NUMBER_RE.findall(str(text)):
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out


def collect_evidence_numbers(evidence: dict[str, Any]) -> set[float]:
    """证据表里出现过的所有数字（含 value/confidence/嵌套列表），作为"允许引用"的白名单."""
    allowed: set[float] = set()

    def walk(node: Any) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            allowed.add(round(float(node), 4))
            return
        if isinstance(node, str):
            for number in _numbers_in(node):
                allowed.add(round(number, 4))
            return
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
            return
        if isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(evidence)
    return allowed


def check_numeric_grounding(report: dict[str, Any], evidence: dict[str, Any], tolerance: float = 0.05) -> dict[str, Any]:
    """数值落地检查：报告正文里的每个数字都应能在证据表里找到（容差内）.

    LLM 最危险的失败模式不是"文笔差"，而是**编造一个看起来很专业的数字**。
    提示词里写"不许编数"是祈使句，这里把它变成可自动检查的硬约束：
    正文数字若不在证据白名单里，就在报告里标出来（并计入 warnings），
    这样"LLM 层"也能像其他模块一样进 CI。
    """
    allowed = collect_evidence_numbers(evidence)
    unsupported: list[dict[str, Any]] = []
    checked = 0
    for section in report.get("sections") or []:
        text = str(section.get("text") or "")
        for number in _numbers_in(text):
            checked += 1
            value = round(number, 4)
            if any(abs(value - candidate) <= max(tolerance, abs(candidate) * 0.01) for candidate in allowed):
                continue
            unsupported.append({"section": section.get("id"), "number": number})
    # 去重（同一个数字在多个段落里出现只报一次）
    seen: set[float] = set()
    unique: list[dict[str, Any]] = []
    for item in unsupported:
        if item["number"] in seen:
            continue
        seen.add(item["number"])
        unique.append(item)
    return {
        "checked_numbers": int(checked),
        "unsupported": unique[:20],
        "grounded": not unique,
        "allowed_numbers": len(allowed),
    }


def _parse_llm_payload(text: str, fallback: dict[str, Any]) -> dict[str, Any]:
    """LLM 返回必须是 JSON；解析不了就把它当作一段正文塞进 sections（并保留离线结论）."""
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and payload.get("sections"):
            out = dict(fallback)
            out.update({
                "headline": str(payload.get("headline") or fallback.get("headline") or ""),
                "confidence": float(payload.get("confidence", fallback.get("confidence", 0.0)) or 0.0),
                "sections": payload.get("sections"),
                "uncertainties": list(payload.get("uncertainties") or fallback.get("uncertainties") or []),
                "next_actions": list(payload.get("next_actions") or fallback.get("next_actions") or []),
            })
            return out
    except Exception:  # noqa: BLE001
        pass
    out = dict(fallback)
    out["sections"] = list(fallback.get("sections") or []) + [
        {"id": "llm", "title": "LLM 原始输出（非 JSON，未解析）", "text": text[:4000], "evidence": []}
    ]
    out["warnings"] = list(fallback.get("warnings") or []) + ["LLM 输出不是预期 JSON，已原样附在后面"]
    return out
