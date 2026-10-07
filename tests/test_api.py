"""服务层（spxh.api）测试：契约字段、路径安全、各接口可用性."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import spxh.api as api


def test_health_reports_estimator(workspace):
    payload = api.health(model_dir=str(workspace / "models"))
    assert payload["ok"] is True
    assert payload["estimator"] == "spxh-m1"
    json.dumps(payload)


def test_list_cases_finds_generated_case(workspace):
    payload = api.list_cases("data")
    assert len(payload["cases"]) == 1
    case = payload["cases"][0]
    assert case["name"] == "case"
    assert case["has_truth"] is True
    assert case["modulation"] == "qpsk"
    assert case["num_samples"] > 0
    json.dumps(payload)


def test_analyze_matches_truth_sidecar(workspace):
    payload = api.analyze_path("data/case.sigmf-meta")
    assert payload["comparison"] is not None
    assert abs(payload["comparison"]["cfo_error_hz"]) < 30.0
    assert abs(payload["comparison"]["symbol_rate_rel_error"]) < 5e-3
    assert "symbol_rate" in payload and "cfo" in payload and "snr" in payload
    json.dumps(payload)


def test_spectrum_contract(workspace):
    payload = api.spectrum_path("data/case.sigmf-meta", points=256)
    assert len(payload["freqs"]) == 256
    assert len(payload["psd_db"]) == 256
    assert len(payload["residual_db"]) == 256
    assert payload["cfo_true"] is not None
    assert min(payload["residual_db"]) >= -60.5
    json.dumps(payload)


def test_waterfall_contract(workspace):
    payload = api.waterfall_path("data/case.sigmf-meta", nperseg=128, max_frames=50, max_bins=32)
    rows, cols = payload["shape"]
    assert len(payload["magnitude_db"]) == rows
    assert len(payload["magnitude_db"][0]) == cols
    assert len(payload["freqs"]) == rows
    assert len(payload["times"]) == cols
    assert rows <= 32 and cols <= 50
    json.dumps(payload)


def test_constellation_contract(workspace):
    payload = api.constellation_path("data/case.sigmf-meta", max_points=400)
    assert payload["count"] == len(payload["i"]) == len(payload["q"])
    assert payload["source"] == "estimated"
    assert payload["cluster_count"] >= 1
    json.dumps(payload)


def test_features_contract(workspace):
    payload = api.features_path("data/case.sigmf-meta")
    assert len(payload["names"]) == 23 == len(payload["values"])
    assert set(payload["groups"]) == {"cumulant", "spectral", "timefreq", "constellation", "cyclostationary"}
    assert abs(payload["symbol_rate"] - 1000.0) < 10.0
    json.dumps(payload)


def test_classify_falls_back_without_model(workspace):
    payload = api.classify_path("data/case.sigmf-meta", model_dir=str(workspace / "models"))
    assert payload["gate"] in ("fallback", "reject")
    assert payload["modulation"] in ("qpsk", None)
    assert payload["truth"]["modulation"] == "qpsk"
    json.dumps(payload)


def test_path_traversal_is_rejected(workspace):
    with pytest.raises(PermissionError):
        api.resolve_path("../../etc/passwd")
    with pytest.raises(PermissionError):
        api.resolve_path(str(Path(workspace).parent / "outside.cf32"))


def test_missing_file_is_reported(workspace):
    with pytest.raises(FileNotFoundError):
        api.resolve_path("data/does-not-exist.sigmf-meta")
