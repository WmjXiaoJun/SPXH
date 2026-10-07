"""M10 测试：LLM 配置（自定义设置）—— 脱敏、校验、优先级、连通性测试、POST 接口."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from spxh.core.report import llm_config as lc


class _FakeLLM(BaseHTTPRequestHandler):
    """假的 OpenAI 兼容服务：用于**离线**验证"测试连接"与模型列表."""

    model = "qwen2.5:7b"
    received: list[dict] = []

    def log_message(self, *args):  # noqa: A003
        return

    def do_GET(self):  # noqa: N802
        body = json.dumps({"data": [{"id": "qwen2.5:7b"}, {"id": "llama3:8b"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        _FakeLLM.received.append({"auth": self.headers.get("Authorization"), "body": payload})
        if not (self.headers.get("Authorization") or "").startswith("Bearer sk-"):
            self.send_response(401)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = json.dumps({"model": payload.get("model", self.model),
                           "choices": [{"message": {"content": "ok"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def fake_llm():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeLLM)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    try:
        yield "http://127.0.0.1:{}/v1".format(port)
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture()
def isolated_config(tmp_path, monkeypatch):
    """把配置文件指到临时目录，别碰仓库里的真实配置."""
    monkeypatch.setattr(lc, "config_path", lambda root=None: tmp_path / "llm_config.json")
    for name in (lc.ENV_BASE, lc.ENV_KEY, lc.ENV_MODEL):
        monkeypatch.delenv(name, raising=False)
    return tmp_path / "llm_config.json"


def test_defaults_when_nothing_configured(isolated_config):
    cfg = lc.load_config()
    assert cfg.base_url == lc.DEFAULT_BASE_URL
    assert cfg.model == lc.DEFAULT_MODEL
    assert cfg.ready is False
    masked = lc.masked_config(cfg)
    assert masked["api_key_set"] is False and masked["api_key_hint"] is None
    assert masked["source"]["base_url"] == "default"


def test_save_and_mask_never_leaks_key(isolated_config):
    cfg = lc.save_config({"base_url": "http://127.0.0.1:9/v1", "api_key": "sk-secret-abcdef123456",
                          "model": "local-model", "temperature": 0.7})
    masked = lc.masked_config(cfg)
    blob = json.dumps(masked, ensure_ascii=False)
    assert "sk-secret" not in blob            # 明文绝不能出现在接口视图里
    assert masked["api_key_hint"] == "****3456"
    assert masked["api_key_set"] is True
    assert masked["ready"] is True
    assert masked["source"]["api_key"].startswith("file")   # file 或 file(dpapi)
    # 重新读盘也要一致（配置真的落盘了）
    again = lc.masked_config(lc.load_config())
    assert again["base_url"] == "http://127.0.0.1:9/v1"
    assert again["api_key_hint"] == "****3456"


def test_empty_api_key_does_not_overwrite(isolated_config):
    lc.save_config({"api_key": "sk-keep-me-1234"})
    lc.save_config({"api_key": ""})            # 前端没改密钥时不该把它清掉
    assert lc.load_config().api_key.endswith("1234")
    lc.save_config({"clear_api_key": True})    # 显式清除才清
    assert lc.load_config().api_key == ""


@pytest.mark.parametrize("bad,field", [
    ({"base_url": "ftp://x"}, "base_url"),
    ({"base_url": "example.com"}, "base_url"),
    ({"temperature": 5}, "temperature"),
    ({"temperature": -1}, "temperature"),
    ({"max_tokens": 0}, "max_tokens"),
    ({"max_tokens": 999999}, "max_tokens"),
    ({"timeout_s": 0}, "timeout_s"),
    ({"model": "  "}, "model"),
    ({"unknown_key": 1}, "unknown_key"),
])
def test_invalid_values_are_rejected(isolated_config, bad, field):
    with pytest.raises(ValueError):
        lc.save_config(bad)


def test_env_used_when_file_missing(isolated_config, monkeypatch):
    monkeypatch.setenv(lc.ENV_BASE, "http://env.example/v1")
    monkeypatch.setenv(lc.ENV_MODEL, "env-model")
    monkeypatch.setenv(lc.ENV_KEY, "sk-env-9999")
    cfg = lc.load_config()
    assert cfg.base_url == "http://env.example/v1"
    assert cfg.model == "env-model"
    assert cfg.source["base_url"] == "env"
    assert lc.masked_config(cfg)["env"][lc.ENV_KEY] == "已设置"


def test_file_beats_env(isolated_config, monkeypatch):
    monkeypatch.setenv(lc.ENV_MODEL, "env-model")
    lc.save_config({"model": "file-model"})
    cfg = lc.load_config()
    assert cfg.model == "file-model"
    assert cfg.source["model"] == "file"


def test_overrides_win_but_do_not_persist(isolated_config, fake_llm):
    lc.save_config({"base_url": "http://127.0.0.1:1/v1", "api_key": "sk-file-1111"})
    cfg = lc.load_config(base_url=fake_llm, api_key="sk-temp-2222")
    assert cfg.base_url == fake_llm and cfg.source["base_url"] == "override"
    # 临时覆盖不能写回文件
    assert lc.load_config().base_url == "http://127.0.0.1:1/v1"


def test_list_models_and_test_connection(isolated_config, fake_llm):
    lc.save_config({"base_url": fake_llm, "api_key": "sk-test-1234567890abcd", "model": "qwen2.5:7b"})
    models = lc.list_models()
    assert models["models"] == ["qwen2.5:7b", "llama3:8b"]
    assert models["error"] is None
    result = lc.test_connection()
    assert result["ok"] is True
    assert result["reply"] == "ok"
    assert result["latency_ms"] >= 0
    assert "sk-test" not in json.dumps(result, ensure_ascii=False)
    assert result["checked"]["api_key_hint"] == "****abcd"


def test_test_connection_reports_failures_in_chinese(isolated_config, fake_llm):
    lc.save_config({"base_url": fake_llm, "api_key": "bad-key", "model": "qwen2.5:7b"})
    result = lc.test_connection()
    assert result["ok"] is False
    assert "401" in result["error"] or "密钥" in result["error"]
    # 完全没有密钥时也要给明确原因
    lc.save_config({"clear_api_key": True})
    assert "API Key" in lc.test_connection()["error"]


def test_narrate_uses_config_and_falls_back_offline(isolated_config, fake_llm):
    from spxh.core.report import narrate

    path = "data/demo/qpsk_conv_snr+10.0_000.sigmf-meta"
    # 没配置 -> 离线（确定性）
    offline = narrate(path, provider="auto")
    assert offline["mode"] == "offline"
    # 配好假服务 -> auto 走 LLM，并带上脱敏的 llm 信息
    lc.save_config({"base_url": fake_llm, "api_key": "sk-test-1234567890abcd", "model": "qwen2.5:7b"})
    online = narrate(path, provider="auto")
    assert online["mode"] == "llm"
    assert online["llm"]["model"] == "qwen2.5:7b"
    assert "sk-test" not in json.dumps(online, ensure_ascii=False)
    # enabled=false -> 即使有 key 也离线
    lc.save_config({"enabled": False})
    assert narrate(path, provider="llm")["mode"] == "offline"


def test_numeric_grounding_flags_invented_numbers():
    from spxh.core.report import check_numeric_grounding

    evidence = {"signal": [{"key": "snr_db", "value": 9.76, "source": "M1"}], "frame": []}
    good = {"sections": [{"id": "signal", "text": "Es/N0 约 9.76 dB，与估计一致。", "evidence": []}]}
    bad = {"sections": [{"id": "signal", "text": "Es/N0 约 12.5 dB，符号速率 1234.5 Hz。", "evidence": []}]}
    assert check_numeric_grounding(good, evidence)["grounded"] is True
    flagged = check_numeric_grounding(bad, evidence)
    assert flagged["grounded"] is False
    numbers = sorted(item["number"] for item in flagged["unsupported"])
    assert numbers == [12.5, 1234.5]


def test_offline_report_is_fully_grounded():
    """离线渲染的数字必须 100% 能在证据里溯源（等于给渲染器做自证）."""
    from spxh.core.report import narrate

    report = narrate("data/demo/qpsk_conv_snr+10.0_000.sigmf-meta", provider="offline")
    grounding = report.get("grounding") or {}
    assert grounding.get("checked_numbers", 0) > 0
    assert grounding.get("grounded") is True, grounding.get("unsupported")


def test_empty_update_is_a_noop(isolated_config):
    """空 body / 空字段不能让配置被"物化落盘"（探测式请求不该改变来源）."""
    import spxh.api as api

    path = isolated_config
    assert not path.exists()
    before = api.llm_config()
    assert before["source"]["base_url"] == "default"
    # 空 body
    after = api.llm_config({})
    assert after["source"]["base_url"] == "default"
    assert not path.exists(), "空 body 不应创建配置文件"
    # 只有空字符串/null 的 body
    api.llm_config({"base_url": "", "model": None, "clear_api_key": False})
    assert not path.exists()
    # 真的要改的时候才落盘
    api.llm_config({"model": "changed-model"})
    assert path.exists()
    assert api.llm_config()["model"] == "changed-model"


def test_config_path_and_model_error_are_display_friendly(isolated_config, monkeypatch):
    import spxh.api as api

    masked = api.llm_config()
    assert not masked["config_path"].startswith("D:")     # 默认给相对路径
    assert masked["config_path"].endswith("llm_config.json")
    assert masked["config_path_abs"]

    # 错误必须是单行短理由（服务商常回多行 JSON）
    class FakeResponseError(Exception):
        pass

    broken = lc.load_config(base_url="http://127.0.0.1:9/v1", api_key="sk-x")
    result = lc.list_models(broken)
    assert result["error"] and "\n" not in result["error"] and len(result["error"]) <= 200


class _FakeDeepSeek(BaseHTTPRequestHandler):
    """模拟 DeepSeek 的两个关键行为：完整 endpoint 也能工作；reasoner 不支持 JSON 模式."""

    def log_message(self, *args):  # noqa: A003
        return

    def _json(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        self._json(200, {"object": "list", "data": [{"id": "deepseek-chat"}, {"id": "deepseek-reasoner"}]})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        if not (self.headers.get("Authorization") or "").startswith("Bearer sk-"):
            self._json(401, {"error": {"message": "Authentication Fails"}})
            return
        if "response_format" in payload and "reasoner" in str(payload.get("model", "")):
            self._json(400, {"error": {"message": "response_format is not supported by deepseek-reasoner"}})
            return
        self._json(200, {"model": payload.get("model", "deepseek-chat"),
                         "choices": [{"message": {"content": "ok"}}]})


@pytest.fixture()
def fake_deepseek():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeDeepSeek)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    try:
        yield "http://127.0.0.1:{}/v1".format(port)
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("raw,expected", [
    ("https://api.deepseek.com", "https://api.deepseek.com"),
    ("https://api.deepseek.com/", "https://api.deepseek.com"),
    ("https://api.deepseek.com/v1", "https://api.deepseek.com/v1"),
    ("https://api.deepseek.com/chat/completions", "https://api.deepseek.com"),
    ("https://api.deepseek.com/v1/chat/completions/", "https://api.deepseek.com/v1"),
    ("https://api.deepseek.com/v1/models", "https://api.deepseek.com/v1"),
])
def test_base_url_normalisation(raw, expected):
    """最常见的配置错误是把完整接口地址粘进 Base URL，这里必须自动纠正."""
    assert lc.normalize_base_url(raw) == expected
    assert lc.save_config({"base_url": raw}).base_url == expected


def test_deepseek_preset_present():
    presets = {item["id"]: item for item in lc.PRESETS}
    assert "deepseek" in presets
    assert presets["deepseek"]["base_url"] == "https://api.deepseek.com"
    assert presets["deepseek"]["model"] == "deepseek-chat"
    assert "ollama" in presets


def test_deepseek_test_connection_and_json_fallback(isolated_config, fake_deepseek):
    """DeepSeek：测试连接要成功；reasoner 不支持 JSON 模式时要能自动降级重试."""
    from spxh.core.report import call_llm, narrate

    lc.save_config({"base_url": fake_deepseek, "api_key": "sk-deepseek-test-0001", "model": "deepseek-chat"})
    result = lc.test_connection()
    assert result["ok"] is True
    assert "deepseek-chat" in (result.get("models") or [])
    assert result["checked"]["chat_url"].endswith("/chat/completions")
    assert "sk-deepseek" not in json.dumps(result, ensure_ascii=False)

    assert call_llm("只输出 JSON", model="deepseek-chat").text == "ok"
    assert call_llm("只输出 JSON", model="deepseek-reasoner").text == "ok"

    lc.save_config({"model": "deepseek-reasoner"})
    report = narrate("data/demo/qpsk_conv_snr+10.0_000.sigmf-meta", provider="auto")
    assert report["mode"] == "llm"


def test_model_hint_when_name_is_wrong(isolated_config, fake_deepseek):
    """模型名写错时给出可用模型清单，而不是只有一句 HTTP 400."""
    lc.save_config({"base_url": fake_deepseek, "api_key": "sk-x-1234", "model": "deepseek-v3"})
    result = lc.test_connection()
    assert result["ok"] is True
    assert "deepseek-chat" in result["hint"]


def test_api_key_is_encrypted_at_rest_on_windows(isolated_config):
    """落盘保护：Windows 上密钥应以 DPAPI 密文存储，**文件里不得出现明文**."""
    from spxh.core.report import secrets

    path = isolated_config
    lc.save_config({"api_key": "sk-plain-should-never-appear-123456"})
    raw = path.read_text(encoding="utf-8")
    if secrets.available():
        import json as _json

        payload = _json.loads(raw)
        assert "api_key" not in payload, "加密可用时不应再写明文 api_key"
        assert payload.get("api_key_enc")
        assert "sk-plain-should-never-appear" not in raw, "配置文件里绝不能出现明文密钥"
        assert lc.load_config().api_key == "sk-plain-should-never-appear-123456"
        assert lc.masked_config()["api_key_protected"] is True
        assert lc.masked_config()["source"]["api_key"] == "file(dpapi)"
    else:
        assert "api_key" in raw and lc.masked_config()["api_key_protected"] is False


def test_api_key_falls_back_to_plaintext_with_flag(isolated_config, monkeypatch):
    """DPAPI 不可用时退回明文，但必须如实标注（不许假装已加密）."""
    monkeypatch.setattr(lc, "protect", lambda text: (None, False))
    lc.save_config({"api_key": "sk-fallback-0001"})
    masked = lc.masked_config()
    assert masked["api_key_set"] is True
    assert masked["api_key_protected"] is False
    assert lc.load_config().api_key == "sk-fallback-0001"


def test_clear_key_removes_both_forms(isolated_config):
    lc.save_config({"api_key": "sk-to-be-removed"})
    lc.save_config({"clear_api_key": True})
    raw = isolated_config.read_text(encoding="utf-8")
    assert "api_key" not in raw and "api_key_enc" not in raw
    assert lc.load_config().api_key == ""
