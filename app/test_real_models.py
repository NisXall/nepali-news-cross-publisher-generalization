"""Integration checks for the saved demo artifacts."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import LABELS
from model_loader import load_all_models, predict


@pytest.fixture(scope="session")
def models():
    return load_all_models()


def test_available_models_return_valid_labels_and_scores(models):
    expected_score_types = {
        "tfidf": "calibrated probability",
        "minilm": "margin score, not a probability",
        "xlmr": "softmax score, not calibrated",
    }
    for model in models.values():
        assert not model.error, f"{model.display_name} failed to load: {model.error}"
        assert model.key in expected_score_types
        assert model.metadata["score_type"] == expected_score_types[model.key]
        for label in LABELS:
            result = predict(model, f"{label} सम्बन्धी नेपाली समाचार")
            assert result["label"] in LABELS
            if result["score_sum"] is not None:
                assert result["score_sum"] == pytest.approx(1.0, abs=1e-5)


def test_edge_inputs_do_not_crash(models):
    for model in models.values():
        assert not model.error, f"{model.display_name} failed to load: {model.error}"
        for text in ("", "This is English text.", "नेपाली " * 2000):
            assert predict(model, text)["label"] in LABELS
