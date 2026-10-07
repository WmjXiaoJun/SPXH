"""CLI 测试."""
from __future__ import annotations

import json
from pathlib import Path

from spxh.cli.main import main


def test_mods_command(capsys):
    assert main(["mods"]) == 0
    out = capsys.readouterr().out
    for modulation in ("bpsk", "qpsk", "8psk", "16qam", "2fsk", "4fsk"):
        assert modulation in out


def test_generate_then_info(tmp_path, capsys):
    prefix = tmp_path / "case"
    assert main([
        "generate", "--mod", "qpsk", "--snr", "12", "--out", str(prefix), "--json",
        "--frames", "2", "--payload-len", "32",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["written"]["sigmf_meta"]).exists()
    assert Path(payload["written"]["truth_json"]).exists()
    assert payload["ground_truth"]["num_frames"] == 2

    assert main(["info", payload["written"]["sigmf_meta"], "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["ok"] is True
    assert info["ground_truth"]["modulation"] == "qpsk"


def test_generate_with_impairments_and_bundle(tmp_path):
    prefix = tmp_path / "impaired"
    assert main([
        "generate", "--mod", "16qam", "--snr", "15", "--cfo-ratio", "0.01",
        "--phase-noise", "0.05", "--iq-gain-db", "0.5", "--formats", "sigmf,truth,bundle",
        "--out", str(prefix), "--json",
    ]) == 0
    assert Path(str(prefix) + ".spxh.npz").exists()
    assert Path(str(prefix) + ".truth.json").exists()


def test_dataset_grid(tmp_path):
    out = tmp_path / "grid"
    assert main([
        "dataset", "--out", str(out), "--mods", "qpsk,2fsk", "--snrs", "0,20",
        "--frames", "1", "--payload-len", "16", "--json",
    ]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["num_cases"] == 4
    for case in manifest["cases"]:
        assert Path(case["files"]["sigmf_meta"]).exists()
        assert Path(case["files"]["sigmf_data"]).exists()
        assert Path(case["files"]["truth_json"]).exists()


def test_info_missing_file_returns_error(tmp_path, capsys):
    assert main(["info", str(tmp_path / "missing.cf32"), "--json"]) == 1
    info = json.loads(capsys.readouterr().out)
    assert info["exists"] is False
