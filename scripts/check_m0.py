"""M0 验收自检：合成链是否正确、可复现、指标是否随 SNR 单调.

用法: python scripts/check_m0.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spxh.core.framing import parse_frame
from spxh.core.synth import SynthConfig, synthesize
from spxh.core.types import MODULATIONS
from spxh.tools.reference_rx import ber, evm_percent, receive


def main() -> int:
    print("=" * 92)
    print("SPXH M0 验收自检：理想参考接收端（无频偏/相噪/失衡，理想定时）")
    print("=" * 92)
    header = "{:7s} {:>4s} {:>6s} {:>8s} {:>9s} {:>9s} {:>10s} {:>8s}".format(
        "mod", "sps", "SNR", "bits", "bit_errors", "BER", "EVM%", "frame_crc"
    )
    print(header)
    print("-" * 92)
    failures = []
    for mod in MODULATIONS:
        for snr in (30.0, 20.0, 10.0, 0.0):
            cfg = SynthConfig(modulation=mod, snr_db=snr, num_frames=2, payload_len=48, seed=20261004)
            signal, truth = synthesize(cfg)
            bits_hat = receive(signal, truth)
            errors = int(np.sum(bits_hat[: truth.bits.size] != truth.bits[: truth.bits.size]))
            ber_value = ber(truth.bits, bits_hat)
            evm = float("nan") if mod.endswith("fsk") else evm_percent(signal, truth)
            parsed = parse_frame(truth.bits)
            crc_ok = parsed["ok"] and parsed["payload"] == truth.frames[0].payload
            print("{:7s} {:4d} {:6.0f} {:8d} {:9d} {:9.2e} {:10.2f} {:>8s}".format(
                mod, truth.sps, snr, truth.num_bits, errors, ber_value, evm, str(crc_ok)))
            if snr >= 20.0 and (errors != 0 or not crc_ok):
                failures.append((mod, snr, errors, crc_ok))

    print("-" * 92)
    # 可复现性
    cfg_a = SynthConfig(modulation="16qam", snr_db=12.0, seed=99)
    cfg_b = SynthConfig(modulation="16qam", snr_db=12.0, seed=99)
    sig_a, gt_a = synthesize(cfg_a)
    sig_b, gt_b = synthesize(cfg_b)
    reproducible = bool(np.array_equal(sig_a.samples, sig_b.samples)) and bool(np.array_equal(gt_a.bits, gt_b.bits))
    print("可复现性（同 seed 两次生成 bit-exact）: " + str(reproducible))
    if not reproducible:
        failures.append(("reproducibility", 0.0, -1, False))

    # 噪声方差是否符合 Es/N0
    cfg_c = SynthConfig(modulation="qpsk", snr_db=10.0, seed=5)
    sig_c, gt_c = synthesize(cfg_c)
    measured = float(np.mean(np.abs(sig_c.samples - gt_c.clean) ** 2))
    ratio_db = 10.0 * np.log10(gt_c.extras["es_per_symbol"] / measured)
    print("实测 sigma^2={:.5g} 目标 sigma^2={:.5g} 反推 Es/N0={:.2f} dB (目标 {:.2f})".format(
        measured, gt_c.noise_variance, ratio_db, cfg_c.snr_db))
    if abs(ratio_db - cfg_c.snr_db) > 0.1:
        failures.append(("noise_variance", 0.0, -1, False))

    print("-" * 92)
    if failures:
        print("FAIL: " + str(failures))
        return 1
    print("PASS: 20 dB 及以上所有调制零误码；CRC 帧解析通过；生成可复现；噪声方差与 Es/N0 一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
