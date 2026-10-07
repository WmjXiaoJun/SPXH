"""pytest 全局配置：把仓库根目录加入 sys.path，并提供合成用例/工作区夹具."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spxh.core.io import save_signal  # noqa: E402
from spxh.core.synth import SynthConfig, synthesize  # noqa: E402


@pytest.fixture
def synth_case():
    """生成合成信号：默认 QPSK、Rs=1234.5 Hz（非整数，避免正好落在 FFT 栅格上）."""

    def _make(**kwargs):
        params = dict(
            modulation="qpsk",
            symbol_rate=1234.5,
            sps=8,
            snr_db=20.0,
            cfo_hz=43.7,
            num_frames=8,
            payload_len=128,
            seed=7,
        )
        params.update(kwargs)
        return synthesize(SynthConfig(**params))

    return _make


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """临时工作区：内含一个带真值的 SigMF 案例，并把 api 的工作区指向它."""
    root = tmp_path / "ws"
    (root / "data").mkdir(parents=True)
    monkeypatch.setenv("SPXH_WORKSPACE", str(root))
    signal, truth = synthesize(
        SynthConfig(
            modulation="qpsk",
            symbol_rate=1000.0,
            sps=8,
            snr_db=15.0,
            cfo_hz=20.0,
            num_frames=4,
            payload_len=64,
            seed=3,
        )
    )
    save_signal(root / "data" / "case", signal, truth=truth, formats=("sigmf", "truth"))
    return root
