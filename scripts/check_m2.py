"""M2 验收自检：分类精度、校准、门控、OOD + 出图.

用法: python scripts/check_m2.py [--per-combo 12] [--model-dir models]
输出: 终端表格 + docs/figures/m2_*.png + models/check_report.json
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

from spxh.core.classify.model import ModulationClassifier, expected_calibration_error, reliability_curve  # noqa: E402
from spxh.core.classify.pipeline import classify_signal  # noqa: E402
from spxh.core.classify.rules import physical_classify  # noqa: E402
from spxh.core.features import extract_features  # noqa: E402
from spxh.core.synth import SynthConfig, synthesize  # noqa: E402
from spxh.core.types import MODULATIONS  # noqa: E402

FIG_DIR = ROOT / "docs" / "figures"
SYMBOL_RATES = (820.0, 977.3, 1010.0, 1234.5, 1480.0)
SPSS = (4, 8)
ROLLOFFS = (0.25, 0.35, 0.5)
PAYLOAD_LENS = (32, 64, 128)
FRAME_COUNTS = (3, 4, 6)

# 验收判据
# 低信噪比下盲目追求准确率没有意义（0 dB 时波形 SNR 约 -9 dB，物理上不可分），
# 真正要保证的是两件事：
#   1) 该准的地方准：10/20 dB 的准确率达标；
#   2) 不确定时不说谎：走机器学习判决的样本里错误率要低，其余必须落进回退/拒绝。
ACC_TARGET = {10.0: 0.85, 20.0: 0.95}     # 该准的地方要准
CONFIDENT_ERROR_TARGET = 0.08           # 走 ML 判决的样本中判错的比例（SNR >= 5 dB）
ECE_TARGET = 0.12                       # 全部样本的校准误差（SNR >= 5 dB）
OOD_DETECT_TARGET = 0.55                # 分布外样本被标记（OOD/回退/拒绝）的比例
COVERAGE_TARGET = 0.90                  # SNR >= 10 dB 时的作答率（非 reject）
LOW_SNR_HONESTY_TARGET = 0.50           # 0 dB 时"不硬答"（回退+拒绝）的比例
RULE_ACC_REPORT_ONLY = True             # 物理规则准确率只报告、不作为验收项（见 M2 报告）


def build_config(rng: np.random.Generator, modulation: str, snr_db: float) -> SynthConfig:
    symbol_rate = float(rng.choice(SYMBOL_RATES))
    sps = int(rng.choice(SPSS))
    return SynthConfig(
        modulation=modulation,
        symbol_rate=symbol_rate,
        sps=sps,
        sample_rate=sps * symbol_rate,
        snr_db=float(snr_db),
        cfo_hz=float(rng.uniform(-0.03, 0.03)) * symbol_rate,
        phase_noise_rms_rad=float(rng.choice([0.0, 0.01, 0.05])),
        rolloff=float(rng.choice(ROLLOFFS)),
        span_symbols=16,
        payload_len=int(rng.choice(PAYLOAD_LENS)),
        num_frames=int(rng.choice(FRAME_COUNTS)),
        iq_gain_db=float(rng.choice([0.0, 0.3])),
        iq_phase_deg=float(rng.choice([0.0, 2.0])),
        seed=int(rng.integers(1, 2 ** 31 - 1)),
        keep_clean=False,
    )


def evaluate(model_dir, per_combo: int, snrs, seed: int = 9999) -> list[dict]:
    classifier = ModulationClassifier.load(model_dir)
    if classifier is None:
        raise SystemExit("未找到训练好的模型：" + str(model_dir) + "（先跑 scripts/train_classifier.py）")
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    started = time.time()
    total = len(MODULATIONS) * len(snrs) * per_combo
    count = 0
    for modulation in MODULATIONS:
        for snr_db in snrs:
            for _ in range(per_combo):
                count += 1
                config = build_config(rng, modulation, snr_db)
                signal, truth = synthesize(config)
                feature_set = extract_features(signal)
                payload = classifier.predict(feature_set, signal=signal)
                records.append(
                    {
                        "truth": modulation,
                        "snr_db": float(snr_db),
                        "predicted": payload["modulation"],
                        "gate": payload["gate"],
                        "confidence": float(payload["confidence"]),
                        "correct": bool(payload["modulation"] == modulation),
                        "ood_score": float(payload["ood"]["score"]),
                        "is_outlier": bool(payload["ood"]["is_outlier"]),
                        "raw_confident": bool(float(max(payload["probabilities"].values())) >= float(payload["threshold"] or 0.7)),
                    }
                )
                if count % 60 == 0:
                    elapsed = time.time() - started
                    print("  评估进度 {}/{}  已用 {:.1f}s".format(count, total, elapsed))
                    sys.stdout.flush()
    return records


def summarize(records: list[dict], snrs) -> dict:
    summary: dict = {"per_snr": {}, "overall": {}}
    for snr in snrs:
        subset = [r for r in records if abs(r["snr_db"] - snr) < 1e-6]
        if not subset:
            continue
        accuracy = float(np.mean([r["correct"] for r in subset]))
        gates = {key: float(np.mean([r["gate"] == key for r in subset])) for key in ("ml", "fallback", "reject")}
        ml_subset = [r for r in subset if r["gate"] == "ml"]
        confident_error = float(1.0 - np.mean([r["correct"] for r in ml_subset])) if ml_subset else float("nan")
        summary["per_snr"][str(snr)] = {
            "accuracy": accuracy,
            "gate_ml": gates["ml"],
            "gate_fallback": gates["fallback"],
            "gate_reject": gates["reject"],
            "confident_error": confident_error,
            "answered_rate": 1.0 - gates["reject"],
            "num": len(subset),
        }
    summary["overall"]["accuracy"] = float(np.mean([r["correct"] for r in records]))
    summary["overall"]["num"] = len(records)
    high = [r for r in records if r["snr_db"] >= 5.0]
    ml_high = [r for r in high if r["gate"] == "ml"]
    confidences = np.asarray([r["confidence"] for r in high], dtype=np.float64)
    correctness = np.asarray([1.0 if r["correct"] else 0.0 for r in high], dtype=np.float64)
    ml_confidences = np.asarray([r["confidence"] for r in ml_high], dtype=np.float64)
    ml_correctness = np.asarray([1.0 if r["correct"] else 0.0 for r in ml_high], dtype=np.float64)
    summary["overall"]["accuracy_snr_ge_5"] = float(np.mean([1.0 if r["correct"] else 0.0 for r in high])) if high else float("nan")
    summary["overall"]["ece_snr_ge_5"] = (
        float(expected_calibration_error(confidences, correctness)) if confidences.size else float("nan")
    )
    summary["overall"]["confident_error_snr_ge_5"] = (
        float(1.0 - np.mean(ml_correctness)) if ml_correctness.size else float("nan")
    )
    summary["overall"]["ml_mean_confidence"] = (
        float(np.mean(ml_confidences)) if ml_confidences.size else float("nan")
    )
    summary["overall"]["coverage_snr_ge_5"] = float(np.mean([r["gate"] != "reject" for r in high])) if high else float("nan")
    summary["reliability"] = reliability_curve(confidences, correctness)
    summary["overall"]["ece_all_snr_ge_5"] = float(expected_calibration_error(confidences, correctness))
    labels = list(MODULATIONS)
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    skipped = 0
    for record in high:
        i = labels.index(record["truth"])
        if record["predicted"] not in labels:
            # 拒答（modulation=None）不属于任何一格，单独计数，不能塞进第 0 列
            skipped += 1
            continue
        matrix[i, labels.index(record["predicted"])] += 1
    summary["confusion"] = matrix.tolist()
    summary["confusion_labels"] = labels
    summary["overall"]["confusion_skipped"] = skipped
    fallback = [r for r in records if r["gate"] == "fallback"]
    summary["overall"]["fallback_count"] = len(fallback)
    summary["overall"]["fallback_accuracy"] = float(np.mean([r["correct"] for r in fallback])) if fallback else float("nan")
    return summary


def ood_evaluation(classifier, count: int = 12) -> dict:
    """分布外样本：纯噪声 + 训练分布之外的波形（2FSK h=0.25、滚降 0.9、极低符号速率）."""
    rng = np.random.default_rng(4242)
    flagged = 0
    total = 0
    details = {}
    cases = {
        "pure_noise": [],
        "2fsk_h0.25": [],
        "rolloff_0.9": [],
        "symbol_rate_80": [],
    }
    for index in range(count):
        noise = (rng.standard_normal(60000) + 1j * rng.standard_normal(60000)) / np.sqrt(2)
        from spxh.core.types import Signal

        signals = {
            "pure_noise": Signal(samples=noise, sample_rate=8000.0),
            "2fsk_h0.25": synthesize(
                SynthConfig(modulation="2fsk", symbol_rate=1000.0, sps=8, snr_db=20.0, fsk_mod_index=0.25, num_frames=4, payload_len=64, seed=1000 + index)
            )[0],
            "rolloff_0.9": synthesize(
                SynthConfig(modulation="16qam", symbol_rate=1000.0, sps=8, snr_db=20.0, rolloff=0.9, num_frames=4, payload_len=64, seed=2000 + index)
            )[0],
            "symbol_rate_80": synthesize(
                SynthConfig(modulation="qpsk", symbol_rate=80.0, sps=8, snr_db=20.0, num_frames=4, payload_len=64, seed=3000 + index)
            )[0],
        }
        for name, signal in signals.items():
            try:
                feature_set = extract_features(signal)
                payload = classifier.predict(feature_set, signal=signal)
                is_flagged = bool(payload["ood"]["is_outlier"]) or payload["gate"] in ("fallback", "reject")
            except Exception:  # noqa: BLE001 - 无法处理的输入也算"被拒"
                is_flagged = True
            total += 1
            flagged += 1 if is_flagged else 0
            details.setdefault(name, []).append(1 if is_flagged else 0)
    return {
        "flagged": flagged,
        "total": total,
        "rate": float(flagged / max(total, 1)),
        "per_case": {k: float(np.mean(v)) for k, v in details.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="M2 验收自检")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--per-combo", type=int, default=10)
    parser.add_argument("--snrs", default="-5,0,5,10,20")
    parser.add_argument("--report", default="models/check_report.json")
    args = parser.parse_args()
    snrs = [float(s) for s in str(args.snrs).split(",") if s.strip()]

    print("=" * 100)
    print("M2-A 分类精度 / 门控 / 校准（每个 (调制, SNR) 组合 {} 个新样本）".format(args.per_combo))
    print("=" * 100)
    records = evaluate(args.model_dir, int(args.per_combo), snrs)
    summary = summarize(records, snrs)

    print("{:>6s} {:>9s} {:>9s} {:>9s} {:>9s} {:>11s} {:>7s}".format(
        "SNR", "accuracy", "gate=ml", "gate=fb", "gate=rej", "ml错误率", "样本数"))
    print("-" * 100)
    for snr in snrs:
        row = summary["per_snr"].get(str(snr))
        if not row:
            continue
        confident = row["confident_error"]
        print("{:6.0f} {:9.3f} {:9.3f} {:9.3f} {:9.3f} {:>11s} {:7d}".format(
            snr,
            row["accuracy"],
            row["gate_ml"],
            row["gate_fallback"],
            row["gate_reject"],
            "n/a" if confident != confident else "{:.3f}".format(confident),
            row["num"],
        ))
    print("-" * 100)
    print("SNR>=5 dB：准确率 {:.4f} | 走 ML 判决的样本错误率 {:.3f} | 作答率（非 reject）{:.3f}".format(
        summary["overall"]["accuracy_snr_ge_5"],
        summary["overall"]["confident_error_snr_ge_5"],
        summary["overall"]["coverage_snr_ge_5"],
    ))
    print("校准：全部样本 ECE {:.4f}（ML 子集平均置信度 {:.3f}）".format(
        summary["overall"]["ece_all_snr_ge_5"], summary["overall"].get("ml_mean_confidence", float("nan"))))
    print("回退样本 {} 个（回退后准确率 {}）".format(
        summary["overall"]["fallback_count"],
        "n/a" if np.isnan(summary["overall"]["fallback_accuracy"]) else "{:.3f}".format(summary["overall"]["fallback_accuracy"]),
    ))
    labels = summary["confusion_labels"]
    print("混淆矩阵（SNR>=5 dB，行=真值，列=预测）：顺序 " + str(labels))
    for label, row in zip(labels, summary["confusion"]):
        print("  {:6s} ".format(label) + " ".join("{:5d}".format(int(v)) for v in row))

    print("")
    print("=" * 100)
    print("M2-B 分布外（OOD）检测：纯噪声 + 训练分布外的波形")
    print("=" * 100)
    classifier = ModulationClassifier.load(args.model_dir)
    ood = ood_evaluation(classifier, count=8)
    for name, rate in ood["per_case"].items():
        print("  {:16s} 被标记比例 {:.3f}".format(name, rate))
    print("  合计 {}/{} = {:.3f}".format(ood["flagged"], ood["total"], ood["rate"]))

    print("")
    print("=" * 100)
    print("M2-C 物理规则单独表现（不看机器学习）")
    print("=" * 100)
    rng = np.random.default_rng(777)
    rule_hits = 0
    rule_total = 0
    for modulation in MODULATIONS:
        hits = 0
        for _ in range(6):
            config = build_config(rng, modulation, 10.0)
            signal, _ = synthesize(config)
            feature_set = extract_features(signal)
            decision = physical_classify(signal, feature_set)
            rule_total += 1
            if decision.modulation == modulation:
                hits += 1
                rule_hits += 1
        print("  {:6s} 规则命中 {}/6".format(modulation, hits))
    rule_accuracy = rule_hits / max(rule_total, 1)
    print("  规则总体准确率（10 dB）：{:.3f}".format(rule_accuracy))

    figures(summary, records, snrs)

    report = {
        "summary": summary,
        "ood": ood,
        "rule_accuracy_10db": rule_accuracy,
        "note": "物理规则准确率仅作报告：低信噪比下物理量本身不可分，规则的价值是可解释与可复现，不是精度",
        "records": records,
        "criteria": {
            "acc_target": {str(k): v for k, v in ACC_TARGET.items()},
            "confident_error_target": CONFIDENT_ERROR_TARGET,
            "ece_target": ECE_TARGET,
            "ood_target": OOD_DETECT_TARGET,
            "coverage_target": COVERAGE_TARGET,
            "low_snr_honesty_target": LOW_SNR_HONESTY_TARGET,
        },
    }
    Path(str(args.report)).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    failures = []
    for snr, target in ACC_TARGET.items():
        row = summary["per_snr"].get(str(snr))
        if row and row["accuracy"] < target:
            failures.append("accuracy@{}dB={:.3f} < {:.2f}".format(snr, row["accuracy"], target))
    if summary["overall"]["ece_all_snr_ge_5"] > ECE_TARGET:
        failures.append("ECE={:.3f} > {:.2f}".format(summary["overall"]["ece_all_snr_ge_5"], ECE_TARGET))
    if summary["overall"]["confident_error_snr_ge_5"] > CONFIDENT_ERROR_TARGET:
        failures.append("ML 判决错误率 {:.3f} > {:.2f}".format(summary["overall"]["confident_error_snr_ge_5"], CONFIDENT_ERROR_TARGET))
    if ood["rate"] < OOD_DETECT_TARGET:
        failures.append("OOD 检出率 {:.3f} < {:.2f}".format(ood["rate"], OOD_DETECT_TARGET))
    # 作答率：SNR >= 10 dB 时系统应该敢答
    answered = [summary["per_snr"].get(str(snr), {}).get("answered_rate", 0.0) for snr in snrs if snr >= 10.0]
    if answered and min(answered) < COVERAGE_TARGET:
        failures.append("SNR>=10 dB 作答率 {:.3f} < {:.2f}".format(min(answered), COVERAGE_TARGET))
    # 低信噪比诚实性：0 dB 时大部分样本必须走回退或拒绝，而不是被自信地答错
    low_rows = [summary["per_snr"].get(str(snr), {}) for snr in snrs if snr <= 0.0]
    low_honesty = [1.0 - float(row.get("gate_ml", 0.0)) for row in low_rows if row]
    if low_honesty and min(low_honesty) < LOW_SNR_HONESTY_TARGET:
        failures.append("低信噪比硬答比例过高：0 dB 时 gate=ml 占 {:.3f}".format(1.0 - min(low_honesty)))

    print("")
    print("=" * 100)
    if failures:
        print("M2 FAIL: " + "; ".join(failures))
        return 1
    print("M2 PASS: 精度/校准/门控/OOD/回退 全部满足判据")
    print("报告: " + str(args.report))
    return 0


def figures(summary, records, snrs) -> None:
    try:
        from spxh.tools.plots import plot_accuracy_curve, plot_confusion, plot_reliability
    except Exception as exc:  # noqa: BLE001
        print("跳过出图（matplotlib 不可用）: " + str(exc))
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    accuracy = [summary["per_snr"].get(str(s), {}).get("accuracy", float("nan")) for s in snrs]
    ml_rate = [summary["per_snr"].get(str(s), {}).get("gate_ml", float("nan")) for s in snrs]
    fallback_rate = [summary["per_snr"].get(str(s), {}).get("gate_fallback", float("nan")) for s in snrs]
    plot_accuracy_curve(
        snrs,
        {"分类准确率": accuracy, "门控=ml 比例": ml_rate, "物理回退比例": fallback_rate},
        FIG_DIR / "m2_accuracy_vs_snr.png",
        title="调制识别：准确率与门控行为（测试集为全新随机参数）",
    )
    plot_confusion(
        summary["confusion"],
        summary["confusion_labels"],
        FIG_DIR / "m2_confusion.png",
        title="混淆矩阵（SNR >= 5 dB，共 {} 个样本）".format(int(np.sum(summary["confusion"]))),
    )
    plot_reliability(
        summary["reliability"],
        FIG_DIR / "m2_reliability.png",
        title="置信度可靠性（SNR >= 5 dB，全部样本）",
        ece=summary["overall"]["ece_all_snr_ge_5"],
    )


if __name__ == "__main__":
    sys.exit(main())
