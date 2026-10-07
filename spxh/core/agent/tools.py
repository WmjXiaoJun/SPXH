"""代理可用的工具集：每一个都是对确定性管线的**薄封装**.

设计原则：
1. 工具**只读**（不改配置、不写文件），参数经过白名单校验；
2. 返回值**结构化**：summary 给人看、evidence 给模型与"数值落地检查"用；
3. 任何工具失败都返回 ok=false + 中文原因，绝不抛出去打断整个代理循环；
4. 证据条目统一成 {key, value, source, unit?, confidence?}，与 /api/narrate 的证据表同构 ——
   这样离线渲染器、数值落地检查、前端证据表都能直接复用。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

__all__ = ["ToolResult", "TOOLS", "run_tool", "tool_schemas", "build_evidence"]


def _api():
    """延迟导入 spxh.api：api 反过来要用 agent（循环导入），所以工具里按需取."""
    import spxh.api as api

    return api


def _item(key: str, value: Any, source: str, unit: str = "", confidence: Optional[float] = None,
          note: str = "") -> dict[str, Any]:
    item: dict[str, Any] = {"key": key, "value": value, "source": source}
    if unit:
        item["unit"] = unit
    if confidence is not None and np.isfinite(confidence):
        item["confidence"] = round(float(confidence), 4)
    if note:
        item["note"] = note
    return item


def _num(value: Any, digits: int = 4) -> Any:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(number):
        return None
    return round(number, digits)


@dataclass
class ToolResult:
    name: str
    args: dict[str, Any]
    ok: bool
    summary: str
    evidence: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    duration_ms: int = 0
    error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "args": self.args,
            "ok": bool(self.ok),
            "summary": self.summary,
            "duration_ms": int(self.duration_ms),
            "error": self.error,
            "evidence": self.evidence,
            "extra": self.extra,
        }


# --------------------------------------------------------------------- 各工具

def _tool_analyze(args: dict[str, Any]) -> ToolResult:
    payload = _api().analyze_path(str(args["path"]))
    # /api/analyze 把 snr / cfo / symbol_rate 放在**顶层**（各自是带 value/unit/confidence 的对象）
    rows = payload.get("estimates") or payload.get("results") or payload
    items: list[dict[str, Any]] = []
    if payload.get("sample_rate") is not None:
        items.append(_item("sample_rate", _num(payload["sample_rate"]), "M0", "Hz"))
    if payload.get("num_samples") is not None:
        items.append(_item("num_samples", int(payload["num_samples"]), "M0"))
    if payload.get("duration_s") is not None:
        items.append(_item("duration_s", _num(payload["duration_s"]), "M0", "s"))
    if payload.get("sps") is not None:
        items.append(_item("sps", _num(payload["sps"], 3), "M1", note="每符号采样点数"))
    occupied = payload.get("occupied") or {}
    if occupied.get("bandwidth") is not None:
        items.append(_item("occupied_bandwidth_hz", _num(occupied["bandwidth"]), "M1", "Hz"))
    peak = payload.get("peak") or {}
    if peak.get("frequency") is not None:
        items.append(_item("peak_frequency_hz", _num(peak["frequency"]), "M1", "Hz"))
    mapping = (("snr", "snr_db", "dB"), ("cfo", "cfo_hz", "Hz"), ("symbol_rate", "symbol_rate_hz", "Hz"))
    parts: list[str] = []
    for key, out_key, unit in mapping:
        entry = rows.get(key) or {}
        value = _num(entry.get("value"))
        if value is None:
            continue
        items.append(_item(out_key, value, "M1", unit, entry.get("confidence"), str(entry.get("method") or "")))
        parts.append("{} {} {}".format(out_key, value, unit))
    return ToolResult("analyze", args, True, "；".join(parts) or "参数估计完成",
                      {"signal": items}, extra={"path": payload.get("path")})


def _tool_spectrum(args: dict[str, Any]) -> ToolResult:
    payload = _api().spectrum_path(str(args["path"]))
    peaks = (payload.get("peaks") or [])[:4]
    items = [
        _item("peak_hz", _num(p.get("frequency_hz") if p.get("frequency_hz") is not None else p.get("frequency")),
              "M1", "Hz", note="功率谱峰")
        for p in peaks
        if (p.get("frequency_hz") is not None or p.get("frequency") is not None)
    ]
    items.append(_item("num_peaks", len(payload.get("peaks") or []), "M1"))
    text = "、".join("{:.1f} Hz".format(float(p["frequency_hz"])) for p in peaks if p.get("frequency_hz") is not None)
    return ToolResult("spectrum", args, True, ("谱峰：" + text) if text else "未发现显著谱峰",
                      {"extra": items})


def _tool_classify(args: dict[str, Any]) -> ToolResult:
    payload = _api().classify_path(str(args["path"]))
    items = [
        _item("modulation", payload.get("modulation"), "M2",
              confidence=payload.get("confidence")),
        _item("gate", payload.get("gate"), "M2", note="ml=机器学习判决 / fallback=物理规则 / reject=拒绝作答"),
    ]
    probabilities = payload.get("probabilities") or {}
    if probabilities:
        best = max(probabilities.items(), key=lambda kv: float(kv[1]))
        items.append(_item("top_probability", _num(best[1]), "M2", note="最可能类别：" + str(best[0])))
    if payload.get("ood_score") is not None:
        items.append(_item("ood_score", _num(payload["ood_score"]), "M2"))
    summary = "调制识别：{}（门控 {}，置信度 {}）".format(
        payload.get("modulation"), payload.get("gate"), _num(payload.get("confidence"), 3))
    return ToolResult("classify", args, True, summary, {"classification": items})


def _tool_demod(args: dict[str, Any]) -> ToolResult:
    payload = _api().demod_path(str(args["path"]))
    lock = payload.get("lock") or {}
    symbols = payload.get("symbols") or {}
    items = [
        _item("lock_state", lock.get("state"), "M3"),
        _item("timing_metric", _num(lock.get("timing_metric"), 3), "M3", note="定时环锁定度量"),
        _item("carrier_metric", _num(lock.get("carrier_metric"), 3), "M3", note="载波环锁定度量"),
        _item("locked_symbols", lock.get("locked_symbols"), "M3"),
        _item("num_symbols", symbols.get("count"), "M3"),
        _item("evm_percent", _num(payload.get("evm_percent")), "M3", "%"),
        _item("modulation", payload.get("modulation"), "M3"),
    ]
    ber = payload.get("ber") or {}
    if ber.get("bit_error_rate") is not None:
        items.append(_item("ber", _num(ber.get("bit_error_rate"), 6), "M3", note="与真值对比"))
        items.append(_item("compared_bits", ber.get("compared_bits"), "M3"))
    elif ber.get("ber") is not None:
        items.append(_item("ber", _num(ber.get("ber"), 6), "M3", note="与真值对比"))
    summary = "解调：锁定 {}，符号 {}，EVM {}%，BER {}".format(
        lock.get("state"), symbols.get("count"), _num(payload.get("evm_percent"), 2),
        _num(ber.get("bit_error_rate"), 6))
    return ToolResult("demod", args, True, summary, {"extra": items})


def _tool_frame(args: dict[str, Any]) -> ToolResult:
    blind = bool(args.get("blind", False))
    payload = _api().frame_path(str(args["path"]), blind=blind)
    summary = payload.get("summary") or {}
    sync = payload.get("sync") or {}
    items = [
        _item("frames", summary.get("frames"), "M5"),
        _item("frames_crc_ok", summary.get("frames_crc_ok"), "M5"),
        _item("payload_bytes_total", summary.get("payload_bytes_total"), "M5", "B"),
        _item("correlation_peak", _num(sync.get("correlation_peak"), 3), "M5"),
        _item("ambiguity_rotation_deg", _num(sync.get("ambiguity_rotation_deg"), 2), "M5", "deg"),
    ]
    if summary.get("payload_bytes_exact") is not None:
        items.append(_item("payload_bytes_exact", summary["payload_bytes_exact"], "M5", "B", note="与真值逐字节一致"))
    blind_info = sync.get("blind")
    if isinstance(blind_info, dict):
        items.append(_item("blind_length_recovered", blind_info.get("length_recovered"), "M5", note="帧头被打错后靠 CRC-8 恢复"))
        items.append(_item("blind_hypotheses_tried", blind_info.get("hypotheses_tried"), "M5"))
    text = "帧同步：解析 {} 帧，{} 帧 CRC 通过，载荷 {} 字节".format(
        summary.get("frames"), summary.get("frames_crc_ok"), summary.get("payload_bytes_total"))
    return ToolResult("frame", args, True, text, {"frame": items})


def _tool_ssca(args: dict[str, Any]) -> ToolResult:
    payload = _api().ssca_path(str(args["path"]))
    features = payload.get("features") or {}
    items = [
        _item("alpha_peak_hz", _num(features.get("symbol_rate_hz")), "M8", "Hz", note="alpha 轴最强峰 ≈ 符号速率"),
        _item("ridge_offset_hz", _num(features.get("cfo_hz")), "M8", "Hz", note="alpha!=0 脊线 ≈ 载频偏移"),
    ]
    peaks = (payload.get("peaks") or [])[:3]
    if peaks:
        items.append(_item("top_peaks", [[_num(p.get("alpha_hz"), 1), _num(p.get("freq_hz"), 1), _num(p.get("level_db"), 1)] for p in peaks],
                           "M8", note="[alpha Hz, f Hz, dB]"))
    estimate = payload.get("estimate") or {}
    if estimate:
        items.append(_item("estimate_symbol_rate_hz", _num(estimate.get("symbol_rate_hz")), "M1", "Hz", note="M1 的精确估计，用于与谱相关对照"))
    summary = "谱相关：alpha 峰 {} Hz（符号速率），脊线偏移 {} Hz".format(
        _num(features.get("symbol_rate_hz"), 1), _num(features.get("cfo_hz"), 1))
    return ToolResult("ssca", args, True, summary, {"cyclostationary": items})


def _tool_cases(args: dict[str, Any]) -> ToolResult:
    payload = _api().list_cases(str(args.get("dir") or "data/demo"))
    names = [str(c.get("name")) for c in (payload.get("cases") or [])][:40]
    items = [_item("case_count", len(payload.get("cases") or []), "M0"), _item("cases", names, "M0")]
    return ToolResult("cases", args, True, "共 {} 个案例".format(len(payload.get("cases") or [])), {"extra": items})


def _tool_decode(args: dict[str, Any]) -> ToolResult:
    """对一段信号跑"完整接收链"：同步 -> 帧头 CRC -> 译码（含 LLR 擦除联合译码）-> CRC 对账."""
    payload = _api().frame_path(str(args["path"]), blind=bool(args.get("blind", True)),
                                erasure_decoding=True)
    summary = payload.get("summary") or {}
    frames = payload.get("frames") or []
    corrected, uncorrectable, erased = 0, 0, 0
    for frame in frames:
        info = frame.get("fec") or {}
        corrected += int(info.get("corrected_symbols") or 0)
        uncorrectable += 1 if info.get("uncorrectable") else 0
        erased += int(info.get("erasures") or 0)
    items = [
        _item("frames", summary.get("frames"), "M5/M6"),
        _item("frames_crc_ok", summary.get("frames_crc_ok"), "M5/M6"),
        _item("corrected_symbols_total", corrected, "M8", note="RS 联合译码纠正的符号总数"),
        _item("uncorrectable_frames", uncorrectable, "M8", note="报不可纠的帧数（不猜，如实上报）"),
        _item("erasures_total", erased, "M8", note="由 |LLR| 门限自动给出的擦除数"),
    ]
    if summary.get("payload_bytes_exact") is not None:
        items.append(_item("payload_bytes_exact", summary["payload_bytes_exact"], "M5", "B"))
    text = "译码：解析 {} 帧，CRC 通过 {}，联合译码纠正 {} 个符号，{} 帧不可纠（擦除 {} 个）".format(
        summary.get("frames"), summary.get("frames_crc_ok"), corrected, uncorrectable, erased)
    return ToolResult("decode", args, True, text, {"frame": items})


def _tool_batch(args: dict[str, Any]) -> ToolResult:
    """批量跑多个案例，给出横向对照（做"多案例巡检"用）."""
    directory = str(args.get("dir") or "data/demo")
    limit = int(args.get("limit") or 5)
    limit = max(1, min(limit, 12))
    wanted = str(args.get("modulation") or "").strip().lower()
    listing = _api().list_cases(directory)
    cases = [c for c in (listing.get("cases") or []) if c.get("name")]
    if wanted:
        cases = [c for c in cases if str(c.get("modulation") or "").lower() == wanted]
    cases = cases[:limit]
    rows: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for case in cases:
        name = str(case.get("name"))
        files = case.get("files") or {}
        path = str(case.get("path") or files.get("sigmf_meta") or case.get("prefix") or "")
        if not path:
            path = directory.rstrip("/") + "/" + name + ".sigmf-meta"
        row: dict[str, Any] = {"case": name}
        try:
            analysis = _api().analyze_path(path)
            row["snr_db"] = _num((analysis.get("snr") or {}).get("value"), 2)
            row["symbol_rate_hz"] = _num((analysis.get("symbol_rate") or {}).get("value"), 1)
        except Exception as exc:  # noqa: BLE001
            row["analyze_error"] = "{}: {}".format(type(exc).__name__, str(exc)[:60])
        try:
            classification = _api().classify_path(path)
            row["modulation"] = classification.get("modulation")
            row["confidence"] = _num(classification.get("confidence"), 3)
            row["gate"] = classification.get("gate")
        except Exception as exc:  # noqa: BLE001
            row["classify_error"] = "{}: {}".format(type(exc).__name__, str(exc)[:60])
        try:
            frames = _api().frame_path(path)
            frame_summary = frames.get("summary") or {}
            row["frames"] = frame_summary.get("frames")
            row["frames_crc_ok"] = frame_summary.get("frames_crc_ok")
        except Exception as exc:  # noqa: BLE001
            row["frame_error"] = "{}: {}".format(type(exc).__name__, str(exc)[:60])
        rows.append(row)
        items.append(_item("case_row", row, "巡检", note="单案例一行：参数 + 制式 + 帧结果"))
    locked = sum(1 for r in rows if (r.get("frames") or 0) > 0)
    failed = sum(1 for r in rows if any(key.endswith("_error") for key in r))
    text = "批量巡检 {} 个案例：{} 个同步成功，{} 个环节报错（8PSK/16QAM 有已知的 M3 误码地板，帧结果可能为空）".format(
        len(rows), locked, failed)
    return ToolResult("batch", args, True, text, {"extra": items}, extra={"rows": rows})


TOOLS: dict[str, dict[str, Any]] = {
    "decode": {
        "title": "译码与联合译码",
        "description": "跑完整接收链并统计译码结果：RS 联合译码纠正了多少符号、哪些帧不可纠、"
                       "由 |LLR| 自动标出的擦除数（M6/M8）。用于判断「数据到底能不能取回来」。",
        "args": {"path": "信号文件路径（必填）", "blind": "布尔，是否开启盲编码/长度搜索（默认 true）"},
        "required": ["path"],
        "run": _tool_decode,
    },
    "batch": {
        "title": "批量巡检",
        "description": "对目录下多个案例批量跑参数估计 + 制式识别 + 帧同步，给出横向对照表（默认 5 个，最多 12 个）。"
                       "用于先摸清一批数据整体是什么情况。",
        "args": {"dir": "目录（默认 data/demo）", "limit": "最多跑几个案例（默认 5，上限 12）",
                 "modulation": "只看某种调制（可选，例如 qpsk；8PSK/16QAM 有已知 M3 局限）"},
        "required": [],
        "run": _tool_batch,
    },
    "analyze": {
        "title": "参数估计",
        "description": "盲估 Es/N0、载频偏移、符号速率（M1）。任何分析都应该先做这一步。",
        "args": {"path": "信号文件路径（必填）"},
        "required": ["path"],
        "run": _tool_analyze,
    },
    "classify": {
        "title": "调制识别",
        "description": "给出调制类型、置信度、门控（ml/fallback/reject）与 OOD 分数（M2）。",
        "args": {"path": "信号文件路径（必填）"},
        "required": ["path"],
        "run": _tool_classify,
    },
    "spectrum": {
        "title": "功率谱",
        "description": "Welch 功率谱与显著谱峰（M1），用于看载频与带宽轮廓。",
        "args": {"path": "信号文件路径（必填）"},
        "required": ["path"],
        "run": _tool_spectrum,
    },
    "demod": {
        "title": "解调",
        "description": "M3 定时/载波同步与判决：锁定状态、符号数、EVM、误码率（有真值时）。",
        "args": {"path": "信号文件路径（必填）"},
        "required": ["path"],
        "run": _tool_demod,
    },
    "frame": {
        "title": "帧同步与载荷",
        "description": "M5：Barker 同步、相位模糊、帧头 CRC-8、载荷提取与 CRC 校验。blind=true 时对编码/交织/长度做盲搜索。",
        "args": {"path": "信号文件路径（必填）", "blind": "布尔，是否开启盲恢复（默认 false）"},
        "required": ["path"],
        "run": _tool_frame,
    },
    "ssca": {
        "title": "谱相关",
        "description": "M8 双频平面：alpha 轴峰给符号速率、脊线给载频偏移，是最直观的循环平稳证据。",
        "args": {"path": "信号文件路径（必填）"},
        "required": ["path"],
        "run": _tool_ssca,
    },
    "cases": {
        "title": "案例列表",
        "description": "列出可用信号文件，便于在不确定时先看看有什么数据。",
        "args": {"dir": "目录（默认 data/demo）"},
        "required": [],
        "run": _tool_cases,
    },
}


def tool_schemas() -> list[dict[str, Any]]:
    """OpenAI function-calling 格式的工具声明."""
    schemas: list[dict[str, Any]] = []
    for name, spec in TOOLS.items():
        properties = {key: {"type": "string", "description": text} for key, text in spec["args"].items()}
        if name == "frame":
            properties["blind"] = {"type": "boolean", "description": spec["args"]["blind"]}
        if name == "decode":
            properties["blind"] = {"type": "boolean", "description": spec["args"]["blind"]}
        if name == "batch":
            properties["limit"] = {"type": "integer", "description": spec["args"]["limit"]}
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": spec["description"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(spec.get("required") or []),
                },
            },
        })
    return schemas


def run_tool(name: str, args: dict[str, Any], default_path: Optional[str] = None) -> ToolResult:
    """执行一个工具：白名单 + 参数校验 + 计时 + 异常兜底（绝不抛出去）."""
    clean: dict[str, Any] = {}
    started = time.time()
    spec = TOOLS.get(str(name))
    if spec is None:
        return ToolResult(str(name), {}, False, "未知工具", error="未知工具：" + str(name))
    for key in spec["args"]:
        if key in args and args[key] not in (None, ""):
            clean[key] = args[key]
    if "path" in spec["args"] and "path" not in clean and default_path:
        clean["path"] = default_path
    missing = [key for key in (spec.get("required") or []) if key not in clean]
    if missing:
        return ToolResult(str(name), clean, False, "参数不完整", error="缺少参数：" + "、".join(missing))
    try:
        result = spec["run"](clean)
    except Exception as exc:  # noqa: BLE001 - 工具失败要如实回报，不能打断循环
        return ToolResult(str(name), clean, False, "执行失败",
                          error="{}: {}".format(type(exc).__name__, str(exc)[:200]),
                          duration_ms=int((time.time() - started) * 1000))
    result.duration_ms = int((time.time() - started) * 1000)
    return result


def build_evidence(path: str) -> dict[str, Any]:
    """空证据表骨架（与 /api/narrate 的 collect_evidence 输出同构）."""
    return {
        "path": str(path),
        "signal": [],
        "classification": [],
        "frame": [],
        "cyclostationary": [],
        "extra": [],
        "pipeline_errors": [],
    }


def merge_evidence(evidence: dict[str, Any], result: ToolResult) -> None:
    """把一次工具调用的证据并入总表；工具失败记进 pipeline_errors."""
    if not result.ok:
        evidence["pipeline_errors"].append({"stage": result.name, "error": result.error or result.summary})
        return
    for section, items in (result.evidence or {}).items():
        target = evidence.setdefault(section, [])
        if isinstance(target, list):
            target.extend(items)
