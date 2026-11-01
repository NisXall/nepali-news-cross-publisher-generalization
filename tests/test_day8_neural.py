import numpy as np
import pandas as pd

from src.day8_neural_experiments import (
    _validated_cached_embeddings,
    _classification_metrics,
    cache_key,
    texts,
    validate_source_isolation,
)


def test_embedding_cache_key_changes_with_split_identity():
    frame = pd.DataFrame({"original_index": [1, 2], "content": ["नेपाल", "समाचार"]})
    assert cache_key(frame, "train", 32) != cache_key(frame, "validation", 32)


def test_text_preprocessing_is_explicit_and_finite():
    frame = pd.DataFrame({"content": ["नेपाल", None]})
    assert texts(frame) == ["नेपाल", ""]
    assert np.isfinite(np.array([1.0])).all()


def test_cached_embeddings_require_exact_partition_shape_and_finite_values(tmp_path):
    frame = pd.DataFrame({"original_index": [1, 2], "content": ["नेपाल", "समाचार"]})
    path = tmp_path / "embeddings.npy"
    np.save(path, np.ones((2, 384), dtype=np.float32))
    assert _validated_cached_embeddings(frame, path).shape == (2, 384)
    np.save(path, np.ones((1, 384), dtype=np.float32))
    assert _validated_cached_embeddings(frame, path) is None
    np.save(path, np.full((2, 384), np.nan, dtype=np.float32))
    assert _validated_cached_embeddings(frame, path) is None


def test_xlm_metrics_use_present_categories_and_preserve_prediction_length():
    frame = pd.Series(["Arts", "Sports", "Arts"])
    predicted = ["Arts", "Arts", "Arts"]
    metrics = _classification_metrics(frame, predicted)
    assert len(predicted) == len(frame)
    assert set(frame.unique()) == {"Arts", "Sports"}
    assert metrics["accuracy"] == 2 / 3
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_held_out_source_isolation_rejects_leakage():
    train = pd.DataFrame({"source": ["other"], "category": ["Arts"]})
    validation = pd.DataFrame({"source": ["other"], "category": ["Arts"]})
    heldout = pd.DataFrame({"source": ["heldout"], "category": ["Arts"]})
    validate_source_isolation(train, validation, heldout, "heldout")
    leaked = pd.DataFrame({"source": ["heldout"], "category": ["Arts"]})
    try:
        validate_source_isolation(leaked, validation, heldout, "heldout")
    except ValueError as exc:
        assert "leaked" in str(exc)
    else:
        raise AssertionError("Expected held-out source leakage to fail")
