"""Focused tests for model contracts and robust input handling."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_ROOT))

from config import LABELS
from model_loader import LoadedModel, predict


def _fake_tfidf(probabilities=None) -> LoadedModel:
    class Vectorizer:
        def transform(self, values):
            return values

    class Classifier:
        classes_ = LABELS

        def decision_function(self, values):
            return [[float(index) for index in range(len(LABELS))]]

    class Calibrator:
        def predict_proba(self, values):
            raw = probabilities or [index + 1 for index in range(len(LABELS))]
            total = sum(raw)
            return [[value / total for value in raw]]

    return LoadedModel(
        "tfidf", "TF-IDF + LinearSVC",
        {"vectorizer": Vectorizer(), "classifier": Classifier(), "calibrator": Calibrator(), "labels": LABELS, "threshold": 0.2932},
        {"score_type": "calibrated probability", "labels": LABELS, "threshold": 0.2932},
        0.0,
    )


def test_tfidf_probability_contract():
    result = predict(_fake_tfidf(), "नेपाली समाचार")
    assert result["label"] == "World"
    assert result["score_label"] == "calibrated probability"
    assert math.isclose(sum(score for _, score in result["top3"]), (18 + 19 + 20) / sum(range(1, 21)))


def test_tfidf_uncertain_flag_uses_validated_threshold():
    low_confidence = _fake_tfidf([1] * len(LABELS))
    high_confidence = _fake_tfidf([100] + [1] * (len(LABELS) - 1))
    assert predict(low_confidence, "synthetic input")["uncertain"] is True
    assert predict(high_confidence, "synthetic input")["uncertain"] is False


def test_empty_english_and_long_text_do_not_crash():
    for text in ("", "This is English text.", "नेपाली " * 2000):
        result = predict(_fake_tfidf(), text)
        assert result["label"] in LABELS


def test_twenty_category_inputs_have_valid_contract():
    for label in LABELS:
        result = predict(_fake_tfidf(), f"{label} सम्बन्धी समाचार")
        assert result["label"] in LABELS
