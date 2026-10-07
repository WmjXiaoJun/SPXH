"""M1 analyze 阶段整合测试."""
from __future__ import annotations

import json

import pytest

from spxh.core.estimate import analyze_signal, compare_with_truth
from spxh.core.types import MODULATIONS


def test_analyze_end_to_end_qpsk(synth_case):
    signal, truth = synth_case(modulation="qpsk", snr_db=10.0, cfo_hz=43.7, symbol_rate=1234.5)
    result = analyze_signal(signal)
    assert abs(result.cfo.value - truth.cfo_hz) <= 0.015 * truth.symbol_rate
    assert abs(result.symbol_rate.value - truth.symbol_rate) / truth.symbol_rate < 2e-3
    assert abs(result.snr.value - truth.snr_db) < 0.6
    assert result.sps == pytest.approx(truth.sps, rel=0.01)
    assert result.warnings == []


@pytest.mark.parametrize("modulation", MODULATIONS)
def test_analyze_smoke_for_all_modulations(synth_case, modulation):
    signal, truth = synth_case(modulation=modulation, snr_db=20.0)
    result = analyze_signal(signal)
    assert abs(result.symbol_rate.value - truth.symbol_rate) / truth.symbol_rate < 2e-3
    payload = result.to_dict()
    json.dumps(payload)  # 必须可 JSON 序列化（GUI/报表要直接用）
    assert payload["symbol_rate"]["value"] > 0


def test_analyze_warns_on_short_record(synth_case):
    signal, _ = synth_case(num_frames=1, payload_len=32)
    result = analyze_signal(signal)
    assert any("Welch" in warning or "过短" in warning for warning in result.warnings)


def test_analyze_flags_low_confidence_at_low_snr(synth_case):
    signal, _ = synth_case(modulation="16qam", snr_db=0.0, num_frames=4, payload_len=64)
    result = analyze_signal(signal)
    assert result.symbol_rate.confidence < 0.35
    assert any("符号速率" in warning for warning in result.warnings)


def test_compare_with_truth_reports_errors(synth_case):
    signal, truth = synth_case(snr_db=20.0, cfo_hz=-88.0, symbol_rate=1000.0)
    result = analyze_signal(signal)
    comparison = compare_with_truth(result, truth)
    assert set(
        [
            "cfo_error_hz",
            "symbol_rate_rel_error",
            "sps_true",
            "sps_est",
            "es_n0_error_db",
            "cfo_confidence",
            "symbol_rate_confidence",
        ]
    ).issubset(comparison)
    assert abs(comparison["cfo_error_hz"]) <= 0.015 * truth.symbol_rate
    assert abs(comparison["es_n0_error_db"]) < 0.6
    assert comparison["sps_true"] == truth.sps


def test_analyze_summary_text_is_human_readable(synth_case):
    signal, _ = synth_case()
    text = analyze_signal(signal).summary()
    assert "采样率" in text
    assert "cfo" in text
    assert "symbol_rate" in text
