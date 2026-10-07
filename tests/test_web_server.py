"""本地 Web 服务测试（真实起服务 + HTTP 请求）."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from spxh.web.server import create_server


@pytest.fixture
def server(workspace):
    httpd = create_server(host="127.0.0.1", port=0, model_dir=str(workspace / "models"))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield "http://127.0.0.1:{}".format(httpd.server_address[1])
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def _get(url: str):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _get_status(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def test_health_endpoint(server):
    status, payload = _get(server + "/api/health")
    assert status == 200
    assert payload["ok"] is True


def test_cases_endpoint(server):
    status, payload = _get(server + "/api/cases?dir=data")
    assert status == 200
    assert payload["cases"][0]["name"] == "case"


def test_analyze_endpoint(server):
    status, payload = _get(server + "/api/analyze?path=data/case.sigmf-meta&nperseg=512")
    assert status == 200
    assert "symbol_rate" in payload
    assert payload["comparison"]["modulation"] == "qpsk"


def test_features_and_classify_endpoints(server):
    status, features = _get(server + "/api/features?path=data/case.sigmf-meta")
    assert status == 200
    assert len(features["values"]) == 23
    status, classification = _get(server + "/api/classify?path=data/case.sigmf-meta")
    assert status == 200
    assert classification["gate"] in ("ml", "fallback", "reject")


def test_missing_parameter_is_400(server):
    assert _get_status(server + "/api/analyze") == 400


def test_unknown_endpoint_is_404(server):
    assert _get_status(server + "/api/nope") == 404
    assert _get_status(server + "/nope") == 404


def test_path_traversal_is_403(server):
    assert _get_status(server + "/api/analyze?path=../../../etc/passwd") == 403


def test_static_index_serves_or_reports_missing(server):
    status = _get_status(server + "/")
    assert status in (200, 503)


def test_static_traversal_is_blocked(server):
    assert _get_status(server + "/static/../../pyproject.toml") in (403, 404)
