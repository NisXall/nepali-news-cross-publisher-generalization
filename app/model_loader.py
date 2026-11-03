"""Model loading and inference helpers for the demo."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from config import (
    LABELS, MINILM_MODEL_ID, MINILM_BUNDLE, TFIDF_BUNDLE,
    MODEL_REPO_ID, XLMR_MODEL_DIR,
)


@dataclass
class LoadedModel:
    key: str
    display_name: str
    model: Any
    metadata: dict[str, Any]
    loaded_seconds: float
    error: str | None = None


def _load_tfidf() -> tuple[Any, dict[str, Any]]:
    import joblib

    bundle = joblib.load(TFIDF_BUNDLE)
    required = {"vectorizer", "classifier", "calibrator", "labels", "threshold"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"TF-IDF bundle is missing keys: {sorted(missing)}")
    calibrator = bundle["calibrator"]
    if not hasattr(calibrator, "multi_class"):
        calibrator.multi_class = "auto"
    return bundle, {
        "score_type": "calibrated probability",
        "labels": tuple(str(x) for x in bundle["labels"]),
        "threshold": float(bundle["threshold"]),
    }


def _load_minilm() -> tuple[Any, dict[str, Any]]:
    import joblib
    from sentence_transformers import SentenceTransformer

    bundle = joblib.load(MINILM_BUNDLE)
    classifier = bundle.get("model")
    if classifier is None:
        raise ValueError("MiniLM bundle has no saved classifier")
    encoder = SentenceTransformer(MINILM_MODEL_ID)
    return (encoder, classifier), {
        "score_type": "margin score, not a probability",
        "labels": tuple(str(x) for x in classifier.classes_),
    }


def _load_xlmr() -> tuple[Any, dict[str, Any]]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_source = str(XLMR_MODEL_DIR) if XLMR_MODEL_DIR.is_dir() else MODEL_REPO_ID
    local_only = XLMR_MODEL_DIR.is_dir()
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=local_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        local_files_only=local_only,
        low_cpu_mem_usage=True,
    )
    labels = tuple(model.config.id2label.get(i, LABELS[i]) for i in range(len(LABELS)))
    return (tokenizer, model), {
        "score_type": "softmax score, not calibrated",
        "labels": labels,
        "max_length": 256,
    }


def load_all_models() -> dict[str, LoadedModel]:
    loaders = (
        ("tfidf", "TF-IDF + LinearSVC", _load_tfidf),
        ("minilm", "MiniLM + LinearSVC", _load_minilm),
        ("xlmr", "XLM-R matched-A", _load_xlmr),
    )
    loaded: dict[str, LoadedModel] = {}
    for key, display_name, loader in loaders:
        started = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("default")
                model, metadata = loader()
            loaded[key] = LoadedModel(
                key, display_name, model, metadata, time.perf_counter() - started
            )
        except Exception as exc:
            loaded[key] = LoadedModel(
                key, display_name, None, {}, time.perf_counter() - started, str(exc)
            )
    return loaded


def _top_three(labels: tuple[str, ...], scores: np.ndarray) -> list[tuple[str, float]]:
    indices = np.argsort(scores)[::-1][:3]
    return [(labels[int(index)], float(scores[int(index)])) for index in indices]


def predict(loaded: LoadedModel, text: str) -> dict[str, Any]:
    if loaded.error or loaded.model is None:
        raise RuntimeError(loaded.error or f"{loaded.display_name} is unavailable")
    started = time.perf_counter()
    labels = loaded.metadata["labels"]
    if loaded.key == "tfidf":
        bundle = loaded.model
        features = bundle["vectorizer"].transform([text])
        margins = bundle["classifier"].decision_function(features)
        probabilities = np.asarray(bundle["calibrator"].predict_proba(margins)[0])
        top = _top_three(labels, probabilities)
        confidence = float(probabilities[np.argmax(probabilities)])
        return {
            "label": top[0][0],
            "confidence": confidence,
            "score_label": loaded.metadata["score_type"],
            "score_sum": float(probabilities.sum()),
            "top3": top,
            "uncertain": confidence < loaded.metadata["threshold"],
            "latency": time.perf_counter() - started,
        }
    if loaded.key == "minilm":
        encoder, classifier = loaded.model
        embedding = encoder.encode([text], normalize_embeddings=True)
        margins = np.asarray(classifier.decision_function(embedding))[0]
        top = _top_three(labels, margins)
        return {
            "label": top[0][0],
            "confidence": float(top[0][1]),
            "score_label": loaded.metadata["score_type"],
            "score_sum": None,
            "top3": top,
            "uncertain": False,
            "latency": time.perf_counter() - started,
        }
    tokenizer, model = loaded.model
    import torch

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=loaded.metadata["max_length"]
    )
    with torch.inference_mode():
        scores = torch.softmax(model(**inputs).logits[0], dim=-1).cpu().numpy()
    top = _top_three(labels, scores)
    return {
        "label": top[0][0],
        "confidence": float(top[0][1]),
        "score_label": loaded.metadata["score_type"],
        "score_sum": float(scores.sum()),
        "top3": top,
        "uncertain": False,
        "latency": time.perf_counter() - started,
    }
