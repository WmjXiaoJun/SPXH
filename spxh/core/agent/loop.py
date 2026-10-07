"""工具调用代理：让模型自己决定"下一步看什么"，但**始终有界、可审计、可离线**.

三条硬约束（这是"能用"和"危险"的分界线）：
1. **有界**：轮数、工具调用次数、墙钟时间三重上限，任一触顶就停并如实回报 stop_reason；
2. **事实只能来自工具**：模型不允许凭空给数字，最终报告要过"数值落地检查"；
3. **没有 LLM 也能跑**：离线确定性计划（analyze -> classify -> frame -> ssca）产出同结构结果，
   面板永远有东西看，也让"代理"这件事在没有 key 的环境里可测。
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

from spxh.core.agent.tools import TOOLS, ToolResult, build_evidence, merge_evidence, run_tool, tool_schemas
from spxh.core.report.llm_config import chat_completion, load_config
from spxh.core.report.narrator import (
    build_prompt,
    check_numeric_grounding,
    llm_info,
    render_offline,
)

__all__ = ["BUDGETS", "run_agent", "agent_tools_payload", "OFFLINE_PLAN"]

BUDGETS = {"max_rounds": 6, "max_tool_calls": 12, "timeout_s": 120}
_HARD_LIMITS = {"max_rounds": 12, "max_tool_calls": 24, "timeout_s": 300}

#: 离线（无 LLM）时的确定性计划：先把最基础的事实拿全，再去要"可解释性"证据
OFFLINE_PLAN: list[tuple[str, dict[str, Any]]] = [
    ("analyze", {}),
    ("classify", {}),
    ("frame", {"blind": True}),
    ("ssca", {}),
]


def agent_tools_payload() -> dict[str, Any]:
    return {
        "tools": [
            {
                "name": name,
                "title": spec["title"],
                "description": spec["description"],
                "args": spec["args"],
                "required": list(spec.get("required") or []),
            }
            for name, spec in TOOLS.items()
        ],
        "budgets": dict(BUDGETS),
        "hard_limits": dict(_HARD_LIMITS),
    }


def _system_prompt(goal: str, path: str, budgets: dict[str, Any]) -> str:
    tool_lines = []
    for name, spec in TOOLS.items():
        args = "、".join(spec["args"]) or "无"
        tool_lines.append("- {}：{}（参数：{}）".format(name, spec["description"], args))
    return (
        "你是射频频谱分析平台 SPXH 的分析代理。你**只能通过工具**获取事实，"
        "不允许凭想象给出任何数值。\n\n"
        "可用工具：\n" + "\n".join(tool_lines) + "\n\n"
        "工作方式：\n"
        "1. 需要事实就调用工具；一次可以调多个。调用工具时**不要**输出解释性长文。\n"
        "2. 默认先 analyze（参数）与 classify（制式），再根据目标选择 frame / ssca / demod / spectrum。\n"
        "3. 同一个工具 + 同一组参数**不要重复调用**（会被拒绝）；已经有答案就停下来给结论。\n"
        "4. 预算：最多 {} 轮、最多 {} 次工具调用、{} 秒。用完必须收尾。\n\n"
        "收尾时**只输出 JSON**（不要再调用工具）。JSON 要简洁：每个 section 的 text 不超过 120 字，\n"
        "只写结论与依据，不要把长长的思考过程写进去（写长了会被截断，平台只能退回离线渲染）。结构：\n"
        '{{"headline":"一句话结论","confidence":0.0,'
        '"sections":[{{"id":"signal|modulation|frame|cyclostationary","title":"标题","text":"中文分析",'
        '"evidence":[{{"key":"...","value":...,"unit":"...","source":"..."}}]}}],'
        '"uncertainties":["..."],"next_actions":["..."]}}\n'
        "正文里出现的每个数字都必须来自工具返回值。\n\n"
        "目标：{}\n信号文件：{}".format(
            budgets["max_rounds"], budgets["max_tool_calls"], budgets["timeout_s"],
            goal or "对这段信号做一次完整可解释的分析", path,
        )
    )


def _tool_message(call_id: str, result: ToolResult, limit: int = 4000) -> dict[str, Any]:
    payload = {
        "ok": result.ok,
        "summary": result.summary,
        "error": result.error,
        "evidence": result.evidence,
    }
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) > limit:
        text = text[:limit] + "…(截断)"
    return {"role": "tool", "tool_call_id": call_id, "content": text}


def _final_from_text(text: str, evidence: dict[str, Any], fallback_reason: str) -> tuple[dict[str, Any], list[str]]:
    """收尾：模型给了合法 JSON 就用它，否则退回离线渲染 + 把模型原话附成一节."""
    warnings: list[str] = []
    base = render_offline(evidence)
    text = str(text or "").strip()
    if text:
        try:
            payload = json.loads(text)
        except ValueError:
            payload = None
        if isinstance(payload, dict) and payload.get("sections"):
            base.update({
                "headline": str(payload.get("headline") or base.get("headline") or ""),
                "confidence": float(payload.get("confidence", base.get("confidence", 0.0)) or 0.0),
                "sections": payload.get("sections"),
                "uncertainties": list(payload.get("uncertainties") or base.get("uncertainties") or []),
                "next_actions": list(payload.get("next_actions") or base.get("next_actions") or []),
            })
            return base, warnings
        base["sections"] = list(base.get("sections") or []) + [
            {"id": "agent", "title": "代理收尾说明", "text": text[:4000], "evidence": []}
        ]
        warnings.append(fallback_reason)
    return base, warnings


def run_agent(
    path: str,
    goal: str = "",
    provider: str = "auto",
    max_rounds: Optional[int] = None,
    max_tool_calls: Optional[int] = None,
    timeout_s: Optional[float] = None,
) -> dict[str, Any]:
    """跑一次代理：有 LLM 就走工具调用循环，没有就走离线确定性计划."""
    budgets = {
        "max_rounds": min(int(max_rounds or BUDGETS["max_rounds"]), _HARD_LIMITS["max_rounds"]),
        "max_tool_calls": min(int(max_tool_calls or BUDGETS["max_tool_calls"]), _HARD_LIMITS["max_tool_calls"]),
        "timeout_s": min(float(timeout_s or BUDGETS["timeout_s"]), _HARD_LIMITS["timeout_s"]),
    }
    started = time.time()
    deadline = started + budgets["timeout_s"]
    evidence = build_evidence(path)
    rounds: list[dict[str, Any]] = []
    seen: set[str] = set()
    tool_call_count = 0
    warnings: list[str] = []
    used: list[str] = []
    prompt_preview = ""
    final_text = ""
    mode = "offline"
    model = "offline-deterministic-plan"
    stop_reason = "no_llm"

    cfg = load_config()
    want_llm = str(provider or "auto").lower() in ("auto", "llm")
    use_llm = want_llm and cfg.ready
    if want_llm and not cfg.ready:
        warnings.append("未配置可用的 LLM，已改用离线确定性计划"
                        + ("（provider=llm 被降级）" if str(provider).lower() == "llm" else ""))

    def _execute(name: str, args: dict[str, Any]) -> ToolResult:
        nonlocal tool_call_count
        key = name + "|" + json.dumps(args, sort_keys=True, ensure_ascii=False)
        if key in seen:
            result = ToolResult(name, args, False, "重复调用已被拒绝",
                                error="同一工具 + 同一参数重复调用会被拒绝，请基于已有结果继续")
        else:
            seen.add(key)
            result = run_tool(name, args, default_path=path)
        tool_call_count += 1
        if result.name not in used:
            used.append(result.name)
        merge_evidence(evidence, result)
        # 把"模型实际看到的那份 JSON"也留档：数值落地检查的白名单就该是它 ——
        # "模型只许引用它被展示过的数字"，而不是"只许引用我挑出来放进证据表的数字"。
        evidence.setdefault("shown", []).append(result.to_dict())
        return result

    if not use_llm:
        # ---------------- 离线确定性计划：同样受预算约束、同样产出结构化结果
        for name, extra_args in OFFLINE_PLAN:
            if tool_call_count >= budgets["max_tool_calls"] or time.time() > deadline:
                stop_reason = "max_tool_calls" if tool_call_count >= budgets["max_tool_calls"] else "timeout"
                break
            args = dict(extra_args)
            result = _execute(name, args)
            rounds.append({
                "index": len(rounds) + 1,
                "assistant_text": "离线计划：执行 " + TOOLS[name]["title"],
                "tool_calls": [result.to_dict()],
            })
        else:
            stop_reason = "no_llm"
    else:
        # ---------------- LLM 工具调用循环
        mode = "llm"
        model = cfg.model
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _system_prompt(goal, path, budgets)},
            {"role": "user", "content": "请开始分析：{}".format(goal or "给出这段信号的完整可解释结论")},
        ]
        prompt_preview = json.dumps(messages, ensure_ascii=False)[:1500]
        for index in range(1, budgets["max_rounds"] + 1):
            if time.time() > deadline:
                stop_reason = "timeout"
                break
            if tool_call_count >= budgets["max_tool_calls"]:
                stop_reason = "max_tool_calls"
                break
            try:
                reply = chat_completion(messages, tools=tool_schemas(), tool_choice="auto",
                                        max_tokens=max(int(cfg.max_tokens), 2000))
            except Exception as exc:  # noqa: BLE001 - LLM 出错要如实回报，不能让请求 5xx
                warnings.append("LLM 调用失败：{}".format(str(exc)[:200]))
                stop_reason = "llm_error"
                break
            message = reply.get("message") or {}
            calls = message.get("tool_calls") or []
            assistant_text = str(message.get("content") or "")
            round_row = {"index": index, "assistant_text": assistant_text, "tool_calls": []}
            rounds.append(round_row)
            if not calls:
                final_text = assistant_text
                stop_reason = "final"
                break
            messages.append({"role": "assistant", "content": assistant_text, "tool_calls": calls})
            for call in calls:
                function = (call or {}).get("function") or {}
                name = str(function.get("name") or "")
                try:
                    args = json.loads(function.get("arguments") or "{}")
                except ValueError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                if tool_call_count >= budgets["max_tool_calls"]:
                    stop_reason = "max_tool_calls"
                    break
                result = _execute(name, args)
                round_row["tool_calls"].append(result.to_dict())
                messages.append(_tool_message(str((call or {}).get("id") or name), result))
            if stop_reason == "max_tool_calls":
                break
        else:
            stop_reason = "max_rounds"

    final, extra_warnings = _final_from_text(final_text, evidence, "代理收尾不是合法 JSON，已用离线渲染兜底并附上原文")
    warnings.extend(extra_warnings)
    if not final.get("sections"):
        final = render_offline(evidence)
    grounding = check_numeric_grounding(final, evidence)
    if not grounding["grounded"]:
        warnings.append("代理报告里有 {} 个数字无法从工具结果溯源（已标出）".format(len(grounding["unsupported"])))

    payload: dict[str, Any] = {
        "path": path,
        "mode": mode,
        "model": model,
        "goal": goal,
        "stop_reason": stop_reason,
        "rounds": rounds,
        "final": final,
        "grounding": grounding,
        "evidence": evidence,
        "prompt_preview": prompt_preview or build_prompt(evidence)[:1500],
        "trace_summary": {
            "rounds": len(rounds),
            "tool_calls": tool_call_count,
            "tools_used": used,
            "elapsed_ms": int((time.time() - started) * 1000),
            "budget": budgets,
        },
        "warnings": warnings,
    }
    if mode == "llm":
        payload["llm"] = llm_info()
    return payload
