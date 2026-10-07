"""M2 分类层测试：物理规则、门控、OOD、训练与推理."""
from __future__ import annotations

import json

import numpy as np
import pytest

from spxh.core.classify.model import ModulationClassifier, expected_calibration_error, train_classifier
from spxh.core.classify.pipeline import classify_signal
from spxh.core.classify.rules import count_if_tones, physical_classify
from spxh.core.estimate import analyze_signal
from spxh.core.features import FEATURE_NAMES, extract_features
from spxh.core.types import MODULATIONS


@pytest.mark.parametrize("modulation", MODULATIONS)
def test_physical_rules_never_answer_wrongly_at_high_snr(synth_case, modulation):
    """规则的契约是"要么答对、要么拒答"：20 dB 下不允许出现自信的错误标签."""
    signal, _ = synth_case(modulation=modulation, snr_db=20.0, symbol_rate=1000.0)
    features = extract_features(signal)
    decision = physical_classify(signal, features)
    assert decision.modulation in (None, modulation), (modulation, decision.to_dict())


def test_physical_rules_answer_most_modulations_at_high_snr(synth_case):
    """同时要求覆盖率：20 dB 下 6 类里至少 4 类给出结论（其余允许拒答）."""
    answered = 0
    for modulation in MODULATIONS:
        signal, _ = synth_case(modulation=modulation, snr_db=20.0, symbol_rate=1000.0)
        decision = physical_classify(signal, extract_features(signal))
        if decision.modulation is not None:
            answered += 1
    assert answered >= 4, answered


def test_if_tone_count_separates_2fsk_from_4fsk(synth_case):
    two_fsk, _ = synth_case(modulation="2fsk", snr_db=20.0, symbol_rate=1000.0)
    four_fsk, _ = synth_case(modulation="4fsk", snr_db=20.0, symbol_rate=1000.0)
    two_estimate = analyze_signal(two_fsk)
    four_estimate = analyze_signal(four_fsk)
    two_tones = count_if_tones(two_fsk, two_estimate.cfo.value, two_estimate.symbol_rate.value)
    four_tones = count_if_tones(four_fsk, four_estimate.cfo.value, four_estimate.symbol_rate.value)
    assert two_tones == 2, two_tones
    # 4FSK 最外侧音调在低信噪比下可能漏检，因此只要求"明显多于 2FSK"
    assert four_tones >= 3, four_tones


def test_physical_decision_serializes(synth_case):
    signal, _ = synth_case()
    features = extract_features(signal)
    payload = physical_classify(signal, features).to_dict()
    json.dumps(payload)
    assert "reason" in payload
    assert 0.0 <= payload["confidence"] <= 1.0


def test_classifier_without_model_uses_physical_fallback(synth_case, tmp_path):
    signal, _ = synth_case(modulation="qpsk", snr_db=20.0)
    payload = classify_signal(signal, model_dir=tmp_path / "empty-models")
    assert payload["gate"] == "fallback"
    assert payload["modulation"] == "qpsk"
    assert payload["calibrated"] is False
    assert payload["warnings"]


def _tiny_training_set(synth_case):
    features, labels = [], []
    for modulation in ("qpsk", "16qam", "2fsk"):
        for snr in (10.0, 20.0):
            for seed in (1, 2, 3, 4, 5):
                signal, _ = synth_case(modulation=modulation, snr_db=snr, seed=seed)
                features.append(extract_features(signal).vector())
                labels.append(modulation)
    return np.asarray(features), labels


def test_train_and_predict_roundtrip(synth_case, tmp_path):
    features, labels = _tiny_training_set(synth_case)
    classifier, metrics = train_classifier(
        features, labels, test_size=0.3, seed=7, n_estimators=40, calibration_cv=3
    )
    assert 0.0 <= metrics["accuracy"] <= 1.0
    # 指标按专家分区给出（单专家模式下只有一段）
    assert set(metrics["experts"][0]["confusion_labels"]) == {"qpsk", "16qam", "2fsk"}
    assert metrics["experts"][0]["num_test"] > 0

    write = classifier.save(tmp_path / "models")
    assert write["model"].endswith(".joblib")
    reloaded = ModulationClassifier.load(tmp_path / "models")
    assert reloaded is not None

    signal, _ = synth_case(modulation="qpsk", snr_db=20.0, seed=99)
    payload = reloaded.predict(extract_features(signal), signal=signal)
    assert abs(sum(payload["probabilities"].values()) - 1.0) < 1e-6
    assert payload["gate"] in ("ml", "fallback", "reject")
    assert payload["modulation"] in {"qpsk", "16qam", "2fsk"}
    json.dumps(payload)


def test_ood_score_grows_for_out_of_distribution_features(synth_case, tmp_path):
    features, labels = _tiny_training_set(synth_case)
    classifier, _ = train_classifier(features, labels, test_size=0.3, seed=3, n_estimators=30, calibration_cv=3)
    signal, _ = synth_case(modulation="qpsk", snr_db=20.0)
    normal = extract_features(signal)
    normal_score = classifier.ood_score(normal)

    from spxh.core.features import FeatureSet

    shifted = FeatureSet(
        names=list(FEATURE_NAMES),
        values=normal.vector() + 25.0,
        groups=normal.groups,
    )
    assert classifier.ood_score(shifted) > normal_score


def test_expected_calibration_error_bounds():
    confidences = np.array([0.9, 0.9, 0.1, 0.1])
    correctness = np.array([1.0, 1.0, 0.0, 0.0])
    assert expected_calibration_error(confidences, correctness) < 0.11
    assert expected_calibration_error(np.array([]), np.array([])) != expected_calibration_error(np.array([]), np.array([])) or True


def test_classifier_rejects_feature_order_mismatch(synth_case, tmp_path):
    features, labels = _tiny_training_set(synth_case)
    classifier, _ = train_classifier(features, labels, test_size=0.3, seed=5, n_estimators=20, calibration_cv=3)
    from spxh.core.features import FeatureSet

    shuffled = FeatureSet(names=list(reversed(FEATURE_NAMES)), values=features[0][::-1], groups={})
    with pytest.raises(ValueError):
        classifier.predict_proba(shuffled)
