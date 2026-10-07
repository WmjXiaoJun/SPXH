"""训练调制分类器（M2）.

用法:
    python scripts/train_classifier.py --out models --per-combo 24 --snrs -5,0,5,10,20

流程: 合成随机参数数据集 -> 提取五域 23 维特征 -> 训练/校准/OOD -> 保存模型与报告。
随机化参数（符号速率、sps、滚降、频偏、相噪、载荷长度）是为了避免分类器记住某一种配置。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spxh.core.classify.model import train_classifier  # noqa: E402
from spxh.core.features import FEATURE_NAMES, FeatureConfig, extract_features  # noqa: E402
from spxh.core.synth import SynthConfig, synthesize  # noqa: E402
from spxh.core.types import MODULATIONS  # noqa: E402

SYMBOL_RATES = (800.0, 977.3, 1000.0, 1234.5, 1500.0)
SPSS = (4, 8)
ROLLOFFS = (0.25, 0.35, 0.5)
PAYLOAD_LENS = (32, 64, 128)
FRAME_COUNTS = (3, 4, 6)


def build_config(rng: np.random.Generator, modulation: str, snr_db: float, seed: int) -> SynthConfig:
    symbol_rate = float(rng.choice(SYMBOL_RATES))
    sps = int(rng.choice(SPSS))
    rolloff = float(rng.choice(ROLLOFFS))
    cfo_ratio = float(rng.uniform(-0.03, 0.03))
    phase_noise = float(rng.choice([0.0, 0.0, 0.01, 0.05]))
    return SynthConfig(
        modulation=modulation,
        symbol_rate=symbol_rate,
        sps=sps,
        sample_rate=sps * symbol_rate,
        snr_db=float(snr_db),
        cfo_hz=cfo_ratio * symbol_rate,
        phase_noise_rms_rad=phase_noise,
        rolloff=rolloff,
        span_symbols=16,
        payload_len=int(rng.choice(PAYLOAD_LENS)),
        num_frames=int(rng.choice(FRAME_COUNTS)),
        iq_gain_db=float(rng.choice([0.0, 0.0, 0.3])),
        iq_phase_deg=float(rng.choice([0.0, 0.0, 2.0])),
        seed=int(seed),
        keep_clean=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="训练 SPXH 调制分类器")
    parser.add_argument("--out", default="models", help="模型输出目录")
    parser.add_argument("--per-combo", type=int, default=24, help="每个 (调制, SNR) 组合的样本数")
    parser.add_argument("--snrs", default="-5,0,5,10,20", help="SNR 网格 (dB)")
    parser.add_argument("--mods", default="all", help="调制列表或 all")
    parser.add_argument("--test-frac", type=float, default=0.25, help="测试集比例")
    parser.add_argument("--seed", type=int, default=20261004, help="随机种子")
    parser.add_argument("--trees", type=int, default=300, help="随机森林树数")
    parser.add_argument("--threshold", type=float, default=0.5, help="置信度门限（单测/实测定为 0.5，见 check_m2.py）")
    parser.add_argument("--report", default="models/training_report.json", help="训练报告输出路径")
    args = parser.parse_args()

    modulations = list(MODULATIONS) if str(args.mods).lower() == "all" else [m.strip() for m in str(args.mods).split(",") if m.strip()]
    snrs = [float(s) for s in str(args.snrs).split(",") if s.strip()]
    feature_config = FeatureConfig()

    rng = np.random.default_rng(int(args.seed))
    features: list[np.ndarray] = []
    labels: list[str] = []
    snr_estimates: list[float] = []
    case_index = 0
    started = time.time()
    total = len(modulations) * len(snrs) * int(args.per_combo)

    print("开始生成数据集：{} 个案例（{} 调制 x {} SNR x {} 样本）".format(total, len(modulations), len(snrs), args.per_combo))
    failures = 0
    for modulation in modulations:
        for snr_db in snrs:
            for _ in range(int(args.per_combo)):
                case_index += 1
                config = build_config(rng, modulation, snr_db, seed=int(rng.integers(1, 2 ** 31 - 1)))
                try:
                    signal, _truth = synthesize(config)
                    feature_set = extract_features(signal, config=feature_config)
                    features.append(feature_set.vector())
                    labels.append(modulation)
                    # 分区要用**估计出来的** SNR（推理时也只有估计值可用），不能用真值
                    snr_estimates.append(float(feature_set.evidence.get("es_n0_db", 0.0)))
                except Exception as exc:  # noqa: BLE001 - 单个样本失败不应该中断训练
                    failures += 1
                    print("  [warn] 案例失败 {}/{}: {}".format(modulation, snr_db, exc))
                if case_index % 50 == 0:
                    elapsed = time.time() - started
                    print("  进度 {}/{}  已用 {:.1f}s  预计剩余 {:.1f}s".format(
                        case_index, total, elapsed, elapsed / case_index * (total - case_index)))
                    sys.stdout.flush()

    if not features:
        print("没有可用样本，训练中止")
        return 1
    x = np.asarray(features, dtype=np.float64)
    print("特征矩阵 {}，失败 {} 个".format(x.shape, failures))

    classifier, metrics = train_classifier(
        x,
        labels,
        snr_db=snr_estimates,
        feature_names=FEATURE_NAMES,
        feature_config=feature_config,
        test_size=float(args.test_frac),
        seed=int(args.seed),
        n_estimators=int(args.trees),
        threshold=float(args.threshold),
    )
    written = classifier.save(args.out)

    report = {
        "written": written,
        "num_cases": int(x.shape[0]),
        "failures": failures,
        "modulations": modulations,
        "snrs": snrs,
        "per_combo": int(args.per_combo),
        "feature_config": feature_config.to_dict(),
        "features": list(FEATURE_NAMES),
        "elapsed_s": round(time.time() - started, 1),
        "metrics": {k: v for k, v in metrics.items() if k not in ("confidences", "correctness")},
        "confidences": metrics["confidences"],
        "correctness": metrics["correctness"],
    }
    report_path = Path(str(args.report))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("")
    print("=" * 84)
    print("训练完成：总体准确率 {:.4f}  总体 ECE {:.4f}  训练样本 {}  测试样本 {}".format(
        metrics["accuracy"], metrics["ece"], classifier.metadata.train_size, classifier.metadata.test_size))
    for expert in metrics["experts"]:
        bound = expert["snr_max"]
        label = "SNR<=inf" if bound is None else "SNR<={:.0f}dB".format(bound)
        print("  [{}] n_train={:5d} acc={:.4f} ECE={:.4f} OOD门限={:.2f}".format(
            label, expert.get("num_test", 0), expert["accuracy"], expert["ece"], expert.get("ood_threshold", float("nan"))))
        print("    各类召回：" + ", ".join("{}={:.3f}".format(k, v) for k, v in expert["per_class_recall"].items()))
        print("    混淆矩阵（行=真值，列=预测，顺序 {}）：".format(expert["confusion_labels"]))
        for row_label, row in zip(expert["confusion_labels"], expert["confusion_matrix"]):
            print("      {:6s} ".format(row_label) + " ".join("{:5d}".format(int(v)) for v in row))
    if "feature_importance" in metrics:
        top = sorted(metrics["feature_importance"].items(), key=lambda kv: -kv[1])[:8]
        print("Top 特征重要度：" + ", ".join("{}={:.3f}".format(k, v) for k, v in top))
    print("模型: " + written["model"])
    print("报告: " + str(report_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
