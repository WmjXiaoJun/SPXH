"""M1 CLI（spectrum / analyze）测试."""
from __future__ import annotations

import json
from pathlib import Path

from spxh.cli.main import main


def _generate(tmp_path, capsys):
    prefix = tmp_path / "case"
    assert main([
        "generate", "--mod", "qpsk", "--snr", "12", "--cfo", "40", "--symbol-rate", "1200",
        "--frames", "8", "--payload-len", "128", "--out", str(prefix), "--json",
    ]) == 0
    capsys.readouterr()  # 丢弃 generate 的 JSON 输出，避免污染后续断言
    return prefix


def test_spectrum_command(tmp_path, capsys):
    prefix = _generate(tmp_path, capsys)
    assert main(["spectrum", str(prefix) + ".sigmf-meta", "--out", str(tmp_path / "spec"), "--waterfall", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["noise_floor_psd"] > 0
    assert Path(payload["psd_csv"]).exists()
    assert Path(payload["waterfall_npz"]).exists()
    assert payload["waterfall_shape"][0] > 0


def test_analyze_command_with_truth_sidecar(tmp_path, capsys):
    prefix = _generate(tmp_path, capsys)
    assert main(["analyze", str(prefix) + ".sigmf-meta", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "comparison" in payload
    comparison = payload["comparison"]
    assert abs(comparison["cfo_error_hz"]) <= 25.0
    assert abs(comparison["symbol_rate_rel_error"]) < 5e-3
    assert abs(comparison["es_n0_error_db"]) < 0.6


def test_analyze_command_plot(tmp_path, capsys):
    prefix = _generate(tmp_path, capsys)
    plot_path = tmp_path / "spectrum.png"
    assert main(["analyze", str(prefix) + ".sigmf-meta", "--plot", str(plot_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["plot"]).exists()
    assert plot_path.stat().st_size > 1000
