import numpy as np
import pandas as pd
import joblib

from src.calibration_abstention import (
    LABELS,
    abstention_metrics,
    calibration_metrics,
    probabilities_from_scores,
)
from sklearn.linear_model import LogisticRegression


def test_calibrator_probabilities_are_finite_and_normalized():
    rng = np.random.default_rng(42)
    scores = rng.normal(size=(60, len(LABELS)))
    labels = np.array(LABELS)[np.arange(60) % len(LABELS)]
    calibrator = LogisticRegression(max_iter=500).fit(scores, labels)
    probabilities = probabilities_from_scores(scores, calibrator)
    assert probabilities.shape == (60, len(LABELS))
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_calibration_schema_and_abstention_metrics():
    probabilities = np.zeros((len(LABELS), len(LABELS)))
    probabilities[np.arange(len(LABELS)), np.arange(len(LABELS))] = 1.0
    labels = pd.Series(LABELS)
    metrics, bins = calibration_metrics(labels, probabilities)
    assert {"brier_score", "log_loss", "ece"} <= metrics.keys()
    assert {"bin", "count", "mean_confidence", "accuracy"} <= set(bins.columns)
    result = abstention_metrics(labels, probabilities, threshold=0.9)
    assert result["coverage"] == 1.0
    assert result["abstained_percent"] == 0.0
    assert result["selective_risk"] == 0.0


def test_calibrated_bundle_reload_preserves_confidence_outputs(tmp_path):
    rng = np.random.default_rng(7)
    scores = rng.normal(size=(60, len(LABELS)))
    labels = np.array(LABELS)[np.arange(60) % len(LABELS)]
    calibrator = LogisticRegression(max_iter=500).fit(scores, labels)
    bundle_path = tmp_path / "bundle.joblib"
    joblib.dump({"calibrator": calibrator, "threshold": 0.3}, bundle_path)
    loaded = joblib.load(bundle_path)
    before = probabilities_from_scores(scores[:4], calibrator)
    after = probabilities_from_scores(scores[:4], loaded["calibrator"])
    np.testing.assert_allclose(before, after)
    assert loaded["threshold"] == 0.3
