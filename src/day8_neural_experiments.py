"""Bounded, leakage-safe neural experiments for the Nepali benchmark."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
CACHE = ROOT / "results" / "cache"
MODELS = ROOT / "results" / "models"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PREPROCESSING_VERSION = "content-v1"
SEED = 42
XLM_MODEL = "xlm-roberta-base"
XLM_MAX_LENGTH = 256
XLM_LEARNING_RATE = 2e-5
XLM_BATCH_SIZE = 4
XLM_GRADIENT_ACCUMULATION = 4
XLM_EPOCHS = 1
ELIGIBILITY_RULE = "at least 1,000 documents and at least 5 observed categories"


def load_partitions() -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(DATA / f"{name}.parquet") for name in ("train", "validation", "test")}


def texts(frame: pd.DataFrame) -> list[str]:
    return frame["content"].fillna("").astype(str).tolist()


def cache_key(frame: pd.DataFrame, split: str, batch_size: int) -> str:
    identifiers = ",".join(map(str, frame["original_index"].tolist()))
    payload = "|".join((EMBEDDING_MODEL, PREPROCESSING_VERSION, "content", split, str(batch_size), identifiers))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def encode_cached(frame: pd.DataFrame, split: str, model, batch_size: int = 32) -> tuple[np.ndarray, dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = cache_key(frame, split, batch_size)
    path = CACHE / f"embeddings_{split}_{key}.npy"
    started = time.perf_counter()
    if path.exists():
        embeddings = np.load(path, mmap_mode="r")
        return np.asarray(embeddings), {"cache": "hit", "path": str(path), "seconds": time.perf_counter() - started}
    embeddings = model.encode(texts(frame), batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding output contains non-finite values")
    np.save(path, embeddings.astype("float32"))
    return embeddings, {"cache": "miss", "path": str(path), "seconds": time.perf_counter() - started}


def _validated_cached_embeddings(
    frame: pd.DataFrame,
    path: Path,
    expected_dimension: int = 384,
) -> np.ndarray | None:
    """Return a cache only when its shape and numeric contents match the partition."""
    try:
        embeddings = np.load(path, mmap_mode="r")
    except (OSError, ValueError):
        return None
    if embeddings.ndim != 2 or embeddings.shape != (len(frame), expected_dimension):
        return None
    if not np.isfinite(embeddings).all():
        return None
    return np.asarray(embeddings)


def encode_full_cached(
    frame: pd.DataFrame,
    split: str,
    model: Any,
    batch_size: int = 128,
    expected_dimension: int = 384,
) -> tuple[np.ndarray, dict]:
    """Encode one complete fixed partition, reusing only a valid exact cache."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = cache_key(frame, split, batch_size)
    path = CACHE / f"embeddings_{split}_{key}.npy"
    started = time.perf_counter()
    if path.exists():
        embeddings = _validated_cached_embeddings(frame, path, expected_dimension)
        if embeddings is not None:
            return embeddings, {
                "cache": "hit",
                "path": str(path),
                "rows": len(frame),
                "dimension": int(embeddings.shape[1]),
                "seconds": time.perf_counter() - started,
            }
    embeddings = model.encode(
        texts(frame),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if embeddings.shape != (len(frame), expected_dimension) or not np.isfinite(embeddings).all():
        raise ValueError(f"Unexpected embedding output shape or values: {embeddings.shape}")
    np.save(path, embeddings.astype("float32"))
    return embeddings, {
        "cache": "miss",
        "path": str(path),
        "rows": len(frame),
        "dimension": int(embeddings.shape[1]),
        "seconds": time.perf_counter() - started,
    }


def evaluate_embeddings(train: pd.DataFrame, validation: pd.DataFrame, x_train: np.ndarray, x_validation: np.ndarray) -> pd.DataFrame:
    rows = []
    for name, classifier in (
        ("LogisticRegression", LogisticRegression(max_iter=200, solver="lbfgs", random_state=SEED)),
        ("LinearSVC", LinearSVC(C=1.0)),
    ):
        started = time.perf_counter()
        classifier.fit(x_train, train["category"])
        predicted = classifier.predict(x_validation)
        rows.append({
            "model": f"{EMBEDDING_MODEL} + {name}",
            "split": "fixed validation",
            "accuracy": accuracy_score(validation["category"], predicted),
            "macro_f1": f1_score(validation["category"], predicted, average="macro", zero_division=0),
            "weighted_f1": f1_score(validation["category"], predicted, average="weighted", zero_division=0),
            "classifier_seconds": time.perf_counter() - started,
            "embedding_dimension": x_train.shape[1],
        })
    return pd.DataFrame(rows)


def _metric_row(y_true: pd.Series, predicted: np.ndarray, classifier: str, seconds: float) -> dict:
    return {
        "model": f"{EMBEDDING_MODEL} + {classifier}",
        "classifier": classifier,
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predicted, average="weighted", zero_division=0)),
        "classifier_seconds": seconds,
    }


def run_full_embedding_benchmark(batch_size: int = 128) -> dict:
    """Run the complete fixed-partition embedding benchmark.

    Both classifiers are selected using validation macro-F1. The test
    partition is scored exactly once, using only the selected classifier.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    partitions = load_partitions()
    split_metadata = json.loads((TABLES / "day3_split_metadata.json").read_text(encoding="utf-8"))
    expected_sizes = split_metadata["partition_sizes"]
    actual_sizes = {name: len(frame) for name, frame in partitions.items()}
    if actual_sizes != expected_sizes:
        raise ValueError(f"Fixed partition sizes changed: expected {expected_sizes}, got {actual_sizes}")
    if split_metadata["grouped_split_duplicate_audit"]["groups_crossing_partitions"] != 0:
        raise ValueError("Fixed partition metadata reports duplicate groups crossing partitions")
    if any(not frame["original_index"].is_unique for frame in partitions.values()):
        raise ValueError("A fixed partition contains duplicate original_index values")
    if set(partitions["train"]["original_index"]) & set(partitions["validation"]["original_index"]):
        raise ValueError("Train and validation partitions overlap")
    if set(partitions["train"]["original_index"]) & set(partitions["test"]["original_index"]):
        raise ValueError("Train and test partitions overlap")
    if set(partitions["validation"]["original_index"]) & set(partitions["test"]["original_index"]):
        raise ValueError("Validation and test partitions overlap")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    embeddings = {}
    cache_status = {}
    for name, frame in partitions.items():
        embeddings[name], cache_status[name] = encode_full_cached(
            frame, name, model, batch_size=batch_size
        )

    validation_rows = []
    fitted = {}
    for classifier_name, classifier in (
        ("LogisticRegression", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED)),
        ("LinearSVC", LinearSVC(C=1.0)),
    ):
        fit_started = time.perf_counter()
        classifier.fit(embeddings["train"], partitions["train"]["category"])
        predicted = classifier.predict(embeddings["validation"])
        fitted[classifier_name] = classifier
        validation_rows.append(
            _metric_row(
                partitions["validation"]["category"],
                predicted,
                classifier_name,
                time.perf_counter() - fit_started,
            )
        )
    validation_metrics = pd.DataFrame(validation_rows).sort_values(
        ["macro_f1", "accuracy", "classifier"], ascending=[False, False, True]
    ).reset_index(drop=True)
    selected_name = str(validation_metrics.iloc[0]["classifier"])
    selected_classifier = fitted[selected_name]

    test_started = time.perf_counter()
    test_predicted = selected_classifier.predict(embeddings["test"])
    test_metrics = _metric_row(
        partitions["test"]["category"],
        test_predicted,
        selected_name,
        time.perf_counter() - test_started,
    )

    TABLES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    validation_path = TABLES / "sentence_embedding_validation_metrics.csv"
    final_metrics_path = TABLES / "sentence_embedding_final_test_metrics.json"
    final_predictions_path = TABLES / "sentence_embedding_final_test_predictions.csv"
    status_path = TABLES / "sentence_embedding_benchmark_status.json"
    bundle_path = MODELS / "sentence_embedding_reference_bundle.joblib"
    validation_metrics.to_csv(validation_path, index=False)
    pd.DataFrame({
        "original_index": partitions["test"]["original_index"],
        "source": partitions["test"]["source"],
        "category": partitions["test"]["category"],
        "predicted_category": test_predicted,
    }).to_csv(final_predictions_path, index=False)
    final_metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "model": selected_classifier,
            "embedding_model": EMBEDDING_MODEL,
            "preprocessing_version": PREPROCESSING_VERSION,
            "selected_by": "fixed validation macro-F1",
            "seed": SEED,
        },
        bundle_path,
    )
    result = {
        "status": "FINAL_COMPLETED",
        "model": EMBEDDING_MODEL,
        "model_revision": "main",
        "preprocessing_version": PREPROCESSING_VERSION,
        "input_field": "content",
        "normalization": "fill missing content with empty string; cast to string; no additional normalization",
        "seed": SEED,
        "partition_sizes": actual_sizes,
        "embedding_dimension": 384,
        "normalize_embeddings": True,
        "batch_size": batch_size,
        "device": device,
        "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "package_versions": {
            "sentence-transformers": importlib.metadata.version("sentence-transformers"),
            "torch": importlib.metadata.version("torch"),
            "scikit-learn": importlib.metadata.version("scikit-learn"),
        },
        "cache": cache_status,
        "validation_metrics_path": str(validation_path),
        "selected_classifier": selected_name,
        "selection_metric": "macro_f1",
        "selection_value": float(validation_metrics.iloc[0]["macro_f1"]),
        "test_evaluation_count": 1,
        "test_metrics_path": str(final_metrics_path),
        "test_predictions_path": str(final_predictions_path),
        "model_bundle_path": str(bundle_path),
        "total_seconds": time.perf_counter() - started,
        "warning": "Pilot artifacts remain separate; this is the final full fixed-partition embedding benchmark.",
    }
    status_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_embedding_pilot(limit: int = 1000) -> dict:
    import torch
    from sentence_transformers import SentenceTransformer

    partitions = load_partitions()
    train, _ = train_test_split(
        partitions["train"],
        train_size=limit,
        random_state=SEED,
        stratify=partitions["train"]["category"],
    )
    validation, _ = train_test_split(
        partitions["validation"],
        train_size=max(200, limit // 4),
        random_state=SEED,
        stratify=partitions["validation"]["category"],
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    smoke = model.encode(["नेपालको समाचार परीक्षण पाठ हो।"], convert_to_numpy=True, normalize_embeddings=True)
    if smoke.shape != (1, 384) or not np.isfinite(smoke).all():
        raise ValueError(f"Unexpected embedding smoke output: {smoke.shape}")
    x_train, train_cache = encode_cached(train, f"train_pilot_{limit}", model)
    x_validation, validation_cache = encode_cached(validation, f"validation_pilot_{limit}", model)
    metrics = evaluate_embeddings(train, validation, x_train, x_validation)
    TABLES.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(TABLES / "day8_embedding_pilot.csv", index=False)
    result = {
        "status": "PILOT_COMPLETED",
        "model": EMBEDDING_MODEL,
        "preprocessing_version": PREPROCESSING_VERSION,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "embedding_dimension": int(x_train.shape[1]),
        "device": device,
        "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "smoke_shape": list(smoke.shape),
        "train_cache": train_cache,
        "validation_cache": validation_cache,
        "metrics_path": str(TABLES / "day8_embedding_pilot.csv"),
        "warning": "Pilot uses prefixes of fixed partitions and is not a full benchmark.",
    }
    (TABLES / "day8_embedding_status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _condition_frames(condition: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Return train, validation, test for a predeclared evaluation condition."""
    from src.classical_benchmark import eligible_sources

    clean = pd.read_parquet(DATA / "np20ng_clean.parquet")
    partitions = load_partitions()
    if condition == "B":
        return (
            partitions["train"],
            partitions["validation"],
            partitions["test"],
            {"condition": "B", "description": "fixed duplicate-aware grouped split"},
        )
    if condition == "A":
        train, remainder = train_test_split(
            clean, test_size=0.4, random_state=SEED, stratify=clean["category"]
        )
        validation, test = train_test_split(
            remainder, test_size=0.5, random_state=SEED + 1, stratify=remainder["category"]
        )
        return (
            train.reset_index(drop=True),
            validation.reset_index(drop=True),
            test.reset_index(drop=True),
            {
                "condition": "A",
                "description": "ordinary stratified random split; duplicate leakage permitted",
                "outer_seed": SEED,
                "inner_seed": SEED + 1,
            },
        )
    raise ValueError(f"Unknown non-source condition: {condition}")


def _source_condition_frames(source: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    from src.classical_benchmark import eligible_sources

    clean = pd.read_parquet(DATA / "np20ng_clean.parquet")
    eligible = eligible_sources(clean)
    if source not in eligible:
        raise ValueError(f"Source is not eligible under {ELIGIBILITY_RULE}: {source}")
    test = clean.loc[clean["source"].eq(source)].copy()
    source_excluded = clean.loc[clean["source"].ne(source)].copy()
    train, validation = train_test_split(
        source_excluded,
        test_size=0.2,
        random_state=SEED,
        stratify=source_excluded["category"],
    )
    validate_source_isolation(train, validation, test, source)
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
        {
            "condition": "C",
            "description": "leave-one-source-out; held-out publisher excluded from train and validation",
            "held_out_source": source,
            "eligibility_rule": ELIGIBILITY_RULE,
            "validation_seed": SEED,
        },
    )


def validate_source_isolation(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    heldout: pd.DataFrame,
    source: str,
) -> None:
    if train["source"].eq(source).any() or validation["source"].eq(source).any():
        raise ValueError(f"Held-out source leaked into training or validation: {source}")
    if set(heldout["source"].unique()) != {source}:
        raise ValueError(f"Held-out evaluation contains unexpected sources: {source}")


def _stratified_limit(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or limit >= len(frame):
        return frame.reset_index(drop=True)
    selected, _ = train_test_split(
        frame,
        train_size=limit,
        random_state=SEED,
        stratify=frame["category"],
    )
    return selected.reset_index(drop=True)


def _classification_metrics(y_true: pd.Series, predicted: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, labels=sorted(y_true.unique()), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predicted, average="weighted", zero_division=0)),
    }


def _xlm_predict(model, tokenizer, frame: pd.DataFrame, label_names: list[str], device, batch_size: int) -> tuple[list[str], np.ndarray]:
    import torch

    model.eval()
    predictions: list[str] = []
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            batch = frame.iloc[start:start + batch_size]
            encoded = tokenizer(
                texts(batch),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=XLM_MAX_LENGTH,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            predictions.extend(label_names[index] for index in logits.argmax(dim=-1).cpu().tolist())
    return predictions, np.asarray(predictions, dtype=object)


def _save_xlm_evaluation(
    condition: str,
    frame: pd.DataFrame,
    predicted: list[str],
    label_names: list[str],
    prefix: str,
) -> None:
    from sklearn.metrics import classification_report, confusion_matrix

    TABLES.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        frame["category"], predicted, labels=label_names, output_dict=True, zero_division=0
    )
    pd.DataFrame(
        [
            {
                "condition": condition,
                "category": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": int(report[label]["support"]),
            }
            for label in label_names
        ]
    ).to_csv(TABLES / f"{prefix}_per_class.csv", index=False)
    pd.DataFrame(confusion_matrix(frame["category"], predicted, labels=label_names), index=label_names, columns=label_names).to_csv(
        TABLES / f"{prefix}_confusion.csv"
    )
    pd.DataFrame(
        {
            "original_index": frame["original_index"],
            "source": frame["source"],
            "category": frame["category"],
            "predicted_category": predicted,
        }
    ).to_csv(TABLES / f"{prefix}_predictions.csv", index=False)


def run_xlm_condition(
    condition: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    condition_metadata: dict,
    output_prefix: str,
    train_limit: int | None = None,
    evaluation_limit: int | None = None,
    epochs: int = XLM_EPOCHS,
) -> dict:
    """Train and evaluate one XLM-R condition with validation checkpoint selection."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = _stratified_limit(train, train_limit)
    if evaluation_limit is not None:
        validation = validation.sample(
            n=min(evaluation_limit, len(validation)), random_state=SEED
        ).reset_index(drop=True)
        test = test.sample(n=min(evaluation_limit, len(test)), random_state=SEED).reset_index(drop=True)
    label_names = sorted(train["category"].unique())
    label_to_id = {label: index for index, label in enumerate(label_names)}
    if set(validation["category"]) - set(label_names) or set(test["category"]) - set(label_names):
        raise ValueError("Evaluation contains a category absent from the training label mapping")
    tokenizer = AutoTokenizer.from_pretrained(XLM_MODEL, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        XLM_MODEL,
        num_labels=len(label_names),
        id2label={index: label for label, index in label_to_id.items()},
        label2id=label_to_id,
        local_files_only=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=XLM_LEARNING_RATE)
    checkpoint_dir = ROOT / "results" / "checkpoints" / output_prefix
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    losses = []
    validation_history = []
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    best_macro_f1 = -1.0
    best_epoch = None
    for epoch in range(epochs):
        model.train()
        shuffled = train.sample(frac=1.0, random_state=SEED + epoch).reset_index(drop=True)
        optimizer.zero_grad(set_to_none=True)
        for batch_number, start in enumerate(range(0, len(shuffled), XLM_BATCH_SIZE)):
            batch = shuffled.iloc[start:start + XLM_BATCH_SIZE]
            encoded = tokenizer(
                texts(batch),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=XLM_MAX_LENGTH,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            target = torch.tensor([label_to_id[label] for label in batch["category"]], dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                output = model(**encoded, labels=target)
                loss = output.loss / XLM_GRADIENT_ACCUMULATION
            loss.backward()
            if (batch_number + 1) % XLM_GRADIENT_ACCUMULATION == 0 or start + XLM_BATCH_SIZE >= len(shuffled):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()) * XLM_GRADIENT_ACCUMULATION)
        validation_predicted, _ = _xlm_predict(model, tokenizer, validation, label_names, device, XLM_BATCH_SIZE)
        validation_metrics = _classification_metrics(validation["category"], validation_predicted)
        validation_history.append({"epoch": epoch + 1, **validation_metrics})
        checkpoint_path = checkpoint_dir / f"epoch-{epoch + 1}"
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch + 1
            model.save_pretrained(checkpoint_dir / "best")
            tokenizer.save_pretrained(checkpoint_dir / "best")
    best_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir / "best", local_files_only=True).to(device)
    test_predicted, _ = _xlm_predict(best_model, tokenizer, test, label_names, device, XLM_BATCH_SIZE)
    test_metrics = _classification_metrics(test["category"], test_predicted)
    _save_xlm_evaluation(condition, test, test_predicted, label_names, f"{output_prefix}_test")
    validation_predicted, _ = _xlm_predict(best_model, tokenizer, validation, label_names, device, XLM_BATCH_SIZE)
    _save_xlm_evaluation(condition, validation, validation_predicted, label_names, f"{output_prefix}_validation")
    result = {
        **condition_metadata,
        "status": "COMPLETED" if evaluation_limit is None else "BOUNDED_COMPLETED",
        "model": XLM_MODEL,
        "model_revision": "local cached revision; exact commit unavailable",
        "text_field": "content",
        "preprocessing_version": PREPROCESSING_VERSION,
        "label_names": label_names,
        "seed": SEED,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "evaluation_limit": evaluation_limit,
        "train_categories": int(train["category"].nunique()),
        "validation_categories": int(validation["category"].nunique()),
        "test_categories": int(test["category"].nunique()),
        "max_length": XLM_MAX_LENGTH,
        "learning_rate": XLM_LEARNING_RATE,
        "epochs": epochs,
        "batch_size": XLM_BATCH_SIZE,
        "gradient_accumulation": XLM_GRADIENT_ACCUMULATION,
        "effective_batch_size": XLM_BATCH_SIZE * XLM_GRADIENT_ACCUMULATION,
        "mixed_precision": "bf16" if use_amp else "disabled",
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "package_versions": {
            "transformers": importlib.metadata.version("transformers"),
            "torch": importlib.metadata.version("torch"),
        },
        "mean_training_loss": float(np.mean(losses)),
        "validation_history": validation_history,
        "best_epoch": best_epoch,
        "validation_metrics": _classification_metrics(validation["category"], validation_predicted),
        "test_metrics": test_metrics,
        "checkpoint": str(checkpoint_dir / "best"),
        "runtime_seconds": time.perf_counter() - started,
        "protocol_note": "Repository has exact normalized duplicate grouping but no implemented Jaccard-0.8 MinHash/LSH near-duplicate safeguard; no such safeguard is claimed here.",
        "warning": (
            "Training and evaluation were intentionally capped by the available compute budget; this is not a full-data result."
            if evaluation_limit is not None
            else "This is a full condition result; test labels were not used for checkpoint selection."
        ),
    }
    (TABLES / f"{output_prefix}_status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_xlm_benchmark(
    conditions: tuple[str, ...] = ("A", "B"),
    train_limit: int | None = None,
    evaluation_limit: int | None = None,
    epochs: int = XLM_EPOCHS,
) -> list[dict]:
    """Run selected XLM-R conditions; C is invoked separately per eligible source."""
    results = []
    for condition in conditions:
        if condition == "C":
            from src.classical_benchmark import eligible_sources

            clean = pd.read_parquet(DATA / "np20ng_clean.parquet")
            for source in eligible_sources(clean):
                train, validation, test, metadata = _source_condition_frames(source)
                results.append(run_xlm_condition(
                    "C", train, validation, test, metadata,
                    f"xlmr_C_{source}", train_limit=train_limit, epochs=epochs,
                    evaluation_limit=evaluation_limit,
                ))
        else:
            train, validation, test, metadata = _condition_frames(condition)
            results.append(run_xlm_condition(
                condition, train, validation, test, metadata,
                f"xlmr_{condition}", train_limit=train_limit, epochs=epochs,
                evaluation_limit=evaluation_limit,
            ))
    (TABLES / "xlmr_benchmark_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def xlm_audit_and_smoke(limit: int = 8, max_length: int = 256) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = "xlm-roberta-base"
    partitions = load_partitions()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sample = partitions["train"].head(limit)
    encoded = tokenizer(texts(sample), return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    labels = torch.tensor([0] * len(sample), dtype=torch.long)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=20)
    output = model(**encoded, labels=labels)
    result = {
        "status": "SMOKE_COMPLETED",
        "model": model_name,
        "max_length": max_length,
        "batch_rows": len(sample),
        "input_shape": list(encoded["input_ids"].shape),
        "logits_shape": list(output.logits.shape),
        "loss_finite": bool(torch.isfinite(output.loss).item()),
        "cuda_available": bool(torch.cuda.is_available()),
        "transformers_version": importlib.metadata.version("transformers"),
        "torch_version": importlib.metadata.version("torch"),
        "warning": "This is a forward/loss smoke test, not fine-tuning or a benchmark result.",
    }
    (TABLES / "day8_xlmr_smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def xlm_token_length_audit(sample_size: int = 2000, max_length: int = 256) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    partitions = load_partitions()
    result = {"model": "xlm-roberta-base", "max_length": max_length, "sample_size": sample_size}
    for name, frame in partitions.items():
        sample = frame.sample(n=min(sample_size, len(frame)), random_state=SEED)
        lengths = [
            len(tokenizer.encode(value, add_special_tokens=True, truncation=False))
            for value in texts(sample)
        ]
        result[name] = {
            "rows": len(lengths),
            "p50": int(np.percentile(lengths, 50)),
            "p90": int(np.percentile(lengths, 90)),
            "p95": int(np.percentile(lengths, 95)),
            "p99": int(np.percentile(lengths, 99)),
            "max": int(max(lengths)),
            "truncation_rate": float(np.mean(np.asarray(lengths) > max_length)),
        }
    (TABLES / "day8_xlmr_token_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def xlm_training_pilot(
    train_rows: int = 256,
    validation_rows: int = 64,
    max_length: int = 128,
    batch_size: int = 4,
    epochs: int = 1,
) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    partitions = load_partitions()
    train, _ = train_test_split(
        partitions["train"], train_size=train_rows, random_state=SEED,
        stratify=partitions["train"]["category"],
    )
    validation, _ = train_test_split(
        partitions["validation"], train_size=validation_rows, random_state=SEED,
        stratify=partitions["validation"]["category"],
    )
    labels = sorted(partitions["train"]["category"].unique())
    label_to_id = {label: index for index, label in enumerate(labels)}
    model = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-base", num_labels=len(labels), id2label={i: label for label, i in label_to_id.items()},
        label2id=label_to_id,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    started = time.perf_counter()
    model.train()
    losses = []
    for epoch in range(epochs):
        shuffled = train.sample(frac=1.0, random_state=SEED + epoch)
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled.iloc[start:start + batch_size]
            encoded = tokenizer(
                texts(batch), return_tensors="pt", padding=True, truncation=True, max_length=max_length
            ).to(device)
            target = torch.tensor(
                [label_to_id[label] for label in batch["category"]], dtype=torch.long, device=device
            )
            output = model(**encoded, labels=target)
            output.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(output.loss.detach().cpu()))
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(validation), batch_size):
            batch = validation.iloc[start:start + batch_size]
            encoded = tokenizer(
                texts(batch), return_tensors="pt", padding=True, truncation=True, max_length=max_length
            ).to(device)
            predictions.extend(model(**encoded).logits.argmax(dim=-1).cpu().tolist())
    predicted_labels = [labels[index] for index in predictions]
    result = {
        "status": "PILOT_COMPLETED",
        "model": "xlm-roberta-base",
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "max_length": max_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 2e-5,
        "mean_training_loss": float(np.mean(losses)),
        "validation_accuracy": float(accuracy_score(validation["category"], predicted_labels)),
        "validation_macro_f1": float(f1_score(validation["category"], predicted_labels, average="macro", zero_division=0)),
        "seconds": time.perf_counter() - started,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if device.type == "cuda" else None
        ),
        "warning": "This is a bounded pilot, not a full benchmark or final model.",
    }
    (TABLES / "day8_xlmr_training_pilot.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--full-embedding-benchmark", action="store_true")
    parser.add_argument("--xlm-benchmark", action="store_true")
    parser.add_argument("--conditions", default="A,B")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--evaluation-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=XLM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.full_embedding_benchmark:
        print(json.dumps(run_full_embedding_benchmark(batch_size=args.batch_size), indent=2))
    elif args.xlm_benchmark:
        conditions = tuple(value.strip().upper() for value in args.conditions.split(",") if value.strip())
        print(json.dumps(
            run_xlm_benchmark(
                conditions=conditions,
                train_limit=args.train_limit,
                evaluation_limit=args.evaluation_limit,
                epochs=args.epochs,
            ),
            indent=2,
        ))
    else:
        print(json.dumps(run_embedding_pilot(), indent=2))
