"""Day 4 sentence-embedding feasibility and baseline helpers."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PREPROCESSING_VERSION = "content-as-stored-v1"


def dependency_status() -> dict[str, bool]:
    return {
        "sentence_transformers": importlib.util.find_spec("sentence_transformers") is not None,
    }


def cache_key(document_id: object, text: str) -> str:
    payload = f"{MODEL_NAME}|{PREPROCESSING_VERSION}|{document_id}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def smoke_test() -> dict:
    status = dependency_status()
    if not status["sentence_transformers"]:
        return {
            "status": "blocked",
            "reason": "sentence-transformers is not installed",
            "model": MODEL_NAME,
        }
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(
            MODEL_NAME,
            model_kwargs={"local_files_only": True},
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return {
            "status": "blocked",
            "reason": "embedding model is not cached locally; no network download attempted",
            "model": MODEL_NAME,
            "error_type": type(exc).__name__,
            "error": str(exc)[:300],
        }
    vector = model.encode(["नेपालको समाचार परीक्षण पाठ हो।"], normalize_embeddings=True)
    return {
        "status": "ready",
        "model": MODEL_NAME,
        "vector_dimension": int(vector.shape[1]),
    }


def run_baseline(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    cache_dir: Path | None = None,
    batch_size: int = 32,
) -> tuple[pd.DataFrame, dict]:
    """Encode fixed partitions and evaluate a train-only classifier.

    This function intentionally raises an explicit ImportError when the
    optional embedding dependency is unavailable.
    """

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    cache_dir = cache_dir or ROOT / "results" / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def encode(frame: pd.DataFrame) -> object:
        key = hashlib.sha256(
            (MODEL_NAME + PREPROCESSING_VERSION + ",".join(frame["original_index"].astype(str))).encode()
        ).hexdigest()[:16]
        path = cache_dir / f"{key}.npy"
        if path.exists():
            import numpy as np

            return np.load(path)
        vectors = model.encode(
            frame["content"].fillna("").tolist(),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        import numpy as np

        np.save(path, vectors)
        return vectors

    started = time.perf_counter()
    x_train = encode(train)
    x_validation = encode(validation)
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(x_train, train["category"])
    predicted = classifier.predict(x_validation)
    metrics = {
        "model": MODEL_NAME,
        "vector_dimension": int(x_train.shape[1]),
        "accuracy": accuracy_score(validation["category"], predicted),
        "macro_f1": f1_score(validation["category"], predicted, average="macro"),
        "weighted_f1": f1_score(validation["category"], predicted, average="weighted"),
        "seconds": time.perf_counter() - started,
    }
    predictions = validation[["original_index", "source", "category"]].copy()
    predictions["predicted_category"] = predicted
    return predictions, metrics


def write_status(path: Path | None = None) -> dict:
    result = {"dependencies": dependency_status(), "smoke_test": smoke_test()}
    path = path or ROOT / "results" / "tables" / "day4_embedding_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(write_status(), indent=2))
