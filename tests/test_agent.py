"""M11 测试：工具调用代理（有界、去重、结构化、可离线）.

全部用**离线脚本化的假 LLM 服务**驱动，不打真 API：
假服务按脚本返回 tool_calls / 最终 JSON，用来精确验证循环的边界行为。
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from spxh.core.agent import run_agent, run_tool, tool_schemas
from spxh.core.report import llm_config as lc

DEMO = "data/demo/qpsk_conv_snr+10.0_000.sigmf-meta"


class _ScriptedLLM(BaseHTTPRequestHandler):
    """按脚本返回响应的 OpenAI 兼容服务；同时记录它收到的请求（用于断言）。"""

    script: list[dict] = []
    requests: list[dict] = []

    def log_message(self, *args):  # noqa: A003
        return

    def _json(self, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        self._json({"data": [{"id": "fake-tool-model"}]})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        _ScriptedLLM.requests.append(payload)
        index = min(len(_ScriptedLLM.requests) - 1, len(_ScriptedLLM.script) - 1)
        message = _ScriptedLLM.script[index] if _ScriptedLLM.script else {"content": "{}"}
        self._json({"model": "fake-tool-model", "choices": [{"message": message, "finish_reason": "stop"}]})


def _call(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


@pytest.fixture()
def scripted_llm(monkeypatch, tmp_path):
    """给一个脚本化的假 LLM，并把配置指到它（同时把配置文件隔离到 tmp）."""
    monkeypatch.setattr(lc, "config_path", lambda root=None: tmp_path / "llm_config.json")
    for name in (lc.ENV_BASE, lc.ENV_KEY, lc.ENV_MODEL):
        monkeypatch.delenv(name, raising=False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedLLM)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def configure(script: list[dict]):
        _ScriptedLLM.script = script
        _ScriptedLLM.requests = []
        lc.save_config({"base_url": "http://127.0.0.1:{}/v1".format(port),
                        "api_key": "sk-fake-1234", "model": "fake-tool-model"})

    try:
        yield configure
    finally:
        server.shutdown()
        server.server_close()


def test_offline_plan_runs_without_llm(monkeypatch, tmp_path):
    """没有可用 LLM 时走离线确定性计划，结构完全一致（面板永远有东西看）."""
    monkeypatch.setattr(lc, "config_path", lambda root=None: tmp_path / "llm_config.json")
    monkeypatch.delenv(lc.ENV_KEY, raising=False)
    report = run_agent(DEMO, goal="离线看看", provider="offline")
    assert report["mode"] == "offline"
    assert report["stop_reason"] == "no_llm"
    assert report["trace_summary"]["tools_used"] == ["analyze", "classify", "frame", "ssca"]
    assert report["final"]["headline"]
    assert report["grounding"]["grounded"] is True


def test_agent_loop_executes_tool_calls_then_finishes(scripted_llm):
    scripted_llm([
        {"content": "", "tool_calls": [_call("analyze", {"path": DEMO}, "c1")]},
        {"content": json.dumps({
            "headline": "参数与谱相关一致",
            "confidence": 0.8,
            "sections": [{"id": "signal", "title": "信号", "text": "符号速率 1000.0 Hz，alpha 峰 1000.0 Hz。",
                          "evidence": []}],
            "uncertainties": [], "next_actions": ["看帧结构"],
        }, ensure_ascii=False)},
    ])
    report = run_agent(DEMO, goal="核查参数", provider="auto")
    assert report["mode"] == "llm"
    assert report["stop_reason"] == "final"
    assert report["trace_summary"]["tool_calls"] == 1
    assert report["trace_summary"]["rounds"] == 2
    assert report["rounds"][0]["tool_calls"][0]["name"] == "analyze"
    assert report["rounds"][0]["tool_calls"][0]["ok"] is True
    assert report["final"]["headline"] == "参数与谱相关一致"
    # 工具声明确实发给了模型（function calling 生效）
    assert _ScriptedLLM.requests[0]["tools"]
    assert any(tool["function"]["name"] == "analyze" for tool in _ScriptedLLM.requests[0]["tools"])


def test_agent_stops_at_max_rounds(scripted_llm):
    """模型一直调工具也要停：轮数上限必须硬生效."""
    always = {"content": "", "tool_calls": [_call("analyze", {"path": DEMO}, "c1")]}
    other = {"content": "", "tool_calls": [_call("classify", {"path": DEMO}, "c2")]}
    scripted_llm([always, other, always, other, always, other, always, other])
    report = run_agent(DEMO, provider="auto", max_rounds=3)
    assert report["stop_reason"] in ("max_rounds", "max_tool_calls")
    assert report["trace_summary"]["rounds"] <= 3
    assert report["trace_summary"]["tool_calls"] <= 3


def test_agent_rejects_duplicate_tool_calls(scripted_llm):
    """同一工具 + 同一参数重复调用会被拒绝（防止模型空转烧钱）."""
    same_args = {"path": DEMO}
    scripted_llm([
        {"content": "", "tool_calls": [_call("analyze", same_args, "c1")]},
        {"content": "", "tool_calls": [_call("analyze", same_args, "c2")]},
        {"content": json.dumps({"headline": "ok", "confidence": 0.5, "sections": [
            {"id": "signal", "title": "信号", "text": "完成", "evidence": []}],
            "uncertainties": [], "next_actions": []}, ensure_ascii=False)},
    ])
    report = run_agent(DEMO, provider="auto", max_rounds=4)
    second_round = report["rounds"][1]["tool_calls"][0]
    assert second_round["ok"] is False
    assert "重复调用" in (second_round["error"] or "")


def test_agent_respects_max_tool_calls(scripted_llm):
    calls = [_call("analyze", {"path": DEMO}, "c{}".format(i)) for i in range(4)]
    scripted_llm([{"content": "", "tool_calls": calls}] * 3)
    report = run_agent(DEMO, provider="auto", max_rounds=6, max_tool_calls=2)
    assert report["stop_reason"] == "max_tool_calls"
    assert report["trace_summary"]["tool_calls"] == 2


def test_agent_reports_llm_error_without_raising(monkeypatch, tmp_path):
    """LLM 出错要如实回报（stop_reason=llm_error），不能抛异常也不能 5xx."""
    monkeypatch.setattr(lc, "config_path", lambda root=None: tmp_path / "llm_config.json")
    monkeypatch.delenv(lc.ENV_KEY, raising=False)
    lc.save_config({"base_url": "http://127.0.0.1:9/v1", "api_key": "sk-x-1234", "model": "m"})
    report = run_agent(DEMO, provider="llm", timeout_s=5)
    assert report["stop_reason"] == "llm_error"
    assert any("LLM 调用失败" in w for w in report["warnings"])
    assert report["final"]["headline"]      # 仍然给出离线兜底的结论


def test_agent_falls_back_when_model_output_is_not_json(scripted_llm):
    scripted_llm([{"content": "我随便说一句，不是 JSON。"}])
    report = run_agent(DEMO, provider="auto")
    assert report["stop_reason"] == "final"
    assert any("不是合法 JSON" in w for w in report["warnings"])
    assert any(section["id"] == "agent" for section in report["final"]["sections"])


def test_tool_runner_validates_arguments_and_unknown_tools():
    unknown = run_tool("nope", {})
    assert unknown.ok is False and "未知工具" in unknown.error
    missing = run_tool("frame", {})
    assert missing.ok is False and "缺少参数" in missing.error
    # 工具失败不抛异常，也不打断循环
    bad_path = run_tool("analyze", {"path": "data/demo/definitely-missing.sigmf-meta"})
    assert bad_path.ok is False and bad_path.error


def test_tool_schemas_are_valid_function_definitions():
    schemas = tool_schemas()
    assert {s["function"]["name"] for s in schemas} >= {"analyze", "classify", "frame", "ssca"}
    for schema in schemas:
        assert schema["type"] == "function"
        assert schema["function"]["description"]
        properties = schema["function"]["parameters"]["properties"]
        assert schema["function"]["parameters"]["type"] == "object"
        assert properties, "每个工具都要声明参数"
        # 除了 cases（只看目录）和 batch（按目录批量巡检）以外，工具都必须能定位到信号文件
        if schema["function"]["name"] not in {"cases", "batch"}:
            assert "path" in properties


def test_agent_grounding_flags_invented_numbers(scripted_llm):
    """模型编数时必须在 grounding 里被标出来（这是"不许编"的落地检查）."""
    scripted_llm([
        {"content": json.dumps({
            "headline": "结论",
            "confidence": 0.5,
            "sections": [{"id": "signal", "title": "信号", "text": "符号速率 1234.5 Hz，纯属虚构。", "evidence": []}],
            "uncertainties": [], "next_actions": [],
        }, ensure_ascii=False)},
    ])
    report = run_agent(DEMO, provider="auto")
    assert report["grounding"]["grounded"] is False
    assert any(abs(item["number"] - 1234.5) < 1e-6 for item in report["grounding"]["unsupported"])
    assert any("无法从工具结果溯源" in w for w in report["warnings"])
