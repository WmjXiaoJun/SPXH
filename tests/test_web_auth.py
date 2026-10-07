"""M11 测试：访问令牌（默认不打扰本机，暴露时必须带令牌）."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from spxh.web.auth import AuthConfig, is_loopback, resolve_token
from spxh.web.server import create_server


def test_default_loopback_needs_no_token():
    auth = AuthConfig(token=None, host="127.0.0.1")
    assert auth.required is False
    assert auth.allows({}) is True
    status = auth.status()
    assert status["auth_required"] is False and status["loopback_only"] is True
    assert "不需要令牌" in status["hint"]


def test_token_enables_auth():
    auth = AuthConfig(token="secret-token", host="127.0.0.1")
    assert auth.required is True
    assert auth.allows({"Authorization": "Bearer secret-token"}) is True
    assert auth.allows({"X-SPXH-Token": "secret-token"}) is True
    assert auth.allows({"Authorization": "Bearer wrong"}) is False
    assert auth.allows({}) is False
    assert auth.is_public("/api/health") is True


def test_non_loopback_requires_token_even_without_one_configured():
    """监听非回环地址却没设令牌 -> 一律拒绝（不能把配置接口裸奔在局域网上）."""
    auth = AuthConfig(token=None, host="0.0.0.0")
    assert auth.required is True
    assert auth.allows({"Authorization": "Bearer anything"}) is False
    assert "拒绝" in auth.status()["hint"]


@pytest.mark.parametrize("host,expected", [
    ("127.0.0.1", True), ("127.0.0.5", True), ("localhost", True), ("::1", True),
    ("192.168.1.10", False), ("0.0.0.0", False), ("example.com", False),
])
def test_is_loopback(host, expected):
    assert is_loopback(host) is expected


def test_resolve_token_variants(monkeypatch):
    monkeypatch.delenv("SPXH_WEB_TOKEN", raising=False)
    assert resolve_token(None) is None
    assert resolve_token("fixed-token") == "fixed-token"
    monkeypatch.setenv("SPXH_WEB_TOKEN", "env-token")
    assert resolve_token(None) == "env-token"
    generated = resolve_token("auto")
    assert generated and generated != "auto" and len(generated) > 16


def test_http_layer_returns_401_and_honours_public_paths():
    """真起一个带令牌的服务：受保护接口 401，探活与 auth/status 保持开放."""
    server = create_server(host="127.0.0.1", port=0, token="tok-123", quiet=True)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def get(path: str, token: str | None = None):
        request = urllib.request.Request("http://127.0.0.1:{}{}".format(port, path))
        if token:
            request.add_header("Authorization", "Bearer " + token)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    try:
        assert get("/api/health")[0] == 200                 # 探活开放
        status, payload = get("/api/auth/status")
        assert status == 200 and payload["auth_required"] is True
        code, payload = get("/api/llm/config")              # 配置接口需要令牌
        assert code == 401 and "令牌" in payload["error"]
        code, payload = get("/api/llm/config", token="tok-123")
        assert code == 200 and "presets" in payload
        assert get("/api/agent/tools", token="tok-123")[0] == 200
    finally:
        server.shutdown()
        server.server_close()
