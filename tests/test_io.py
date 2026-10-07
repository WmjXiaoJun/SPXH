"""多格式 I/O 测试."""
from __future__ import annotations

import json

import numpy as np
import pytest

from spxh.core.io import (
    infer_datatype,
    load_signal,
    read_bundle,
    read_raw,
    read_sigmf,
    read_wav,
    save_signal,
    write_bundle,
    write_raw,
    write_sigmf,
    write_wav,
)
from spxh.core.io.api import describe_file
from spxh.core.synth import SynthConfig, synthesize
from spxh.core.types import Signal


def _demo_signal(fs: float = 8000.0, n: int = 512) -> Signal:
    # 峰值缩放到 0.25，保证整型格式（ci16/rf32）不会因为满量程裁剪而丢信息
    rng = np.random.default_rng(21)
    samples = 0.25 * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    return Signal(samples=samples, sample_rate=fs, center_freq=433e6, meta={"origin": "test"})


def test_infer_datatype():
    assert infer_datatype("x.cf32") == "cf32"
    assert infer_datatype("x.cs16") == "ci16"
    assert infer_datatype("x.wav") is None


def test_raw_cf32_roundtrip(tmp_path):
    signal = _demo_signal()
    path = tmp_path / "a.cf32"
    write_raw(path, signal)
    loaded = read_raw(path, sample_rate=signal.sample_rate)
    assert np.allclose(loaded.samples, signal.samples, atol=1e-6)
    assert loaded.meta["is_real"] is False


def test_raw_ci16_quantization_error_is_small(tmp_path):
    signal = _demo_signal()
    path = tmp_path / "a.ci16"
    write_raw(path, signal, datatype="ci16")
    loaded = read_raw(path, datatype="ci16", sample_rate=signal.sample_rate)
    error = np.max(np.abs(loaded.samples - signal.samples))
    assert error < 1e-4


def test_raw_real_format_has_zero_imaginary(tmp_path):
    signal = _demo_signal()
    path = tmp_path / "a.rf32"
    write_raw(path, signal, datatype="rf32")
    loaded = read_raw(path, sample_rate=signal.sample_rate)
    assert np.allclose(loaded.samples.imag, 0.0)
    assert loaded.meta["is_real"] is True


def test_raw_requires_explicit_sample_rate(tmp_path):
    path = tmp_path / "a.cf32"
    write_raw(path, _demo_signal())
    with pytest.raises(ValueError, match="sample_rate"):
        read_raw(path)


def test_wav_iq_roundtrip(tmp_path):
    signal = _demo_signal(fs=48000.0)
    path = tmp_path / "a.wav"
    write_wav(path, signal)
    loaded = read_wav(path)
    assert loaded.sample_rate == pytest.approx(signal.sample_rate)
    assert loaded.num_samples == signal.num_samples
    assert np.allclose(loaded.samples, signal.samples, atol=1e-7)


def test_wav_real_becomes_analytic(tmp_path):
    signal = _demo_signal(fs=48000.0)
    path = tmp_path / "real.wav"
    write_wav(path, signal, mode="real")
    loaded = read_wav(path, analytic=True)
    assert loaded.num_samples == signal.num_samples
    assert not np.allclose(loaded.samples.imag, 0.0)


@pytest.mark.parametrize("datatype", ["cf32_le", "ci16_le", "rf32_le"])
def test_sigmf_roundtrip(tmp_path, datatype):
    signal = _demo_signal()
    paths = write_sigmf(tmp_path / "b", signal, datatype=datatype, description="unit-test")
    assert paths["meta"].endswith(".sigmf-meta")
    assert paths["data"].endswith(".sigmf-data")
    loaded = read_sigmf(paths["meta"])
    assert loaded.sample_rate == pytest.approx(signal.sample_rate)
    assert loaded.center_freq == pytest.approx(signal.center_freq)
    tolerance = 1e-3 if datatype == "ci16_le" else 1e-6
    if datatype.startswith("rf"):
        # 实信号格式天然丢弃虚部
        assert np.allclose(loaded.samples.imag, 0.0)
        assert np.allclose(loaded.samples.real, signal.samples.real, atol=tolerance)
    else:
        assert np.allclose(loaded.samples, signal.samples, atol=tolerance)


def test_sigmf_carries_ground_truth_and_annotations(tmp_path):
    signal, truth = synthesize(SynthConfig(modulation="qpsk", snr_db=25.0, num_frames=2, payload_len=16))
    paths = write_sigmf(tmp_path / "c", signal, truth=truth)
    meta = json.loads(open(paths["meta"], encoding="utf-8").read())
    assert meta["global"]["core:datatype"] == "cf32_le"
    assert len(meta["annotations"]) == 2
    assert "spxh:ground_truth" in meta


def test_bundle_roundtrip_with_truth(tmp_path):
    signal, truth = synthesize(SynthConfig(modulation="16qam", snr_db=30.0, num_frames=1, payload_len=8))
    path = write_bundle(tmp_path / "bundle", signal, truth)
    loaded_signal, loaded_truth = read_bundle(path)
    assert np.allclose(loaded_signal.samples, signal.samples)
    assert loaded_signal.sample_rate == pytest.approx(signal.sample_rate)
    assert loaded_truth is not None
    assert np.array_equal(loaded_truth.bits, truth.bits)
    assert loaded_truth.frames[0].payload == truth.frames[0].payload


def test_save_and_load_signal_dispatch(tmp_path):
    signal, truth = synthesize(SynthConfig(modulation="8psk", snr_db=30.0, num_frames=1, payload_len=8))
    written = save_signal(tmp_path / "case", signal, truth=truth, formats=("sigmf", "truth", "bundle"))
    assert set(["sigmf_meta", "truth_json", "bundle"]).issubset(written.keys())

    from_sigmf = load_signal(written["sigmf_meta"])
    from_bundle = load_signal(written["bundle"])
    assert np.allclose(from_sigmf.samples, signal.samples)
    assert np.allclose(from_bundle.samples, signal.samples)

    info = describe_file(written["sigmf_meta"])
    assert info["ok"] is True
    assert info["signal"]["num_samples"] == signal.num_samples
    assert info["ground_truth"]["modulation"] == "8psk"


def test_load_signal_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_signal(tmp_path / "nope.cf32")
