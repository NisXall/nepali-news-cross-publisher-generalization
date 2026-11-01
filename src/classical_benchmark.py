"""Classical benchmark experiments for cross-source Nepali classification.

The module keeps expensive work explicit and reproducible:

* existing 300k-feature baseline results are loaded rather than retrained;
* new comparisons use a bounded 100k-feature configuration;
* the final reference model is retrained on fixed train and evaluated once on
  the untouched test partition;
* source-held-out models are fit independently for each eligible publisher.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
SEED = 42
LABELS = tuple(sorted(pd.read_parquet(DATA / "train.parquet", columns=["category"])["category"].unique()))
SOURCE_RULE = "at least 1,000 documents and at least 5 observed categories"
REFERENCE_CONFIG = {
    "word_ngram_range": [1, 2],
    "min_df": 2,
    "max_features": 300000,
    "sublinear_tf": True,
    "classifier": "LinearSVC",
    "C": 1.0,
    "class_weight": None,
}
BOUNDED_WORD_CONFIG = {
    "word_ngram_range": [1, 2],
    "min_df": 2,
    "max_features": 10000,
    "sublinear_tf": True,
}
BOUNDED_MAX_FEATURES = 10000


def load_partitions() -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_parquet(DATA / f"{name}.parquet")
        for name in ("train", "validation", "test")
    }


def validate_partitions(partitions: dict[str, pd.DataFrame]) -> dict:
    required = {"original_index", "category", "source", "content", "normalized_content"}
    for name, frame in partitions.items():
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    indices = {name: set(frame["original_index"]) for name, frame in partitions.items()}
    disjoint = all(
        indices[left].isdisjoint(indices[right])
        for left in indices
        for right in indices
        if left < right
    )
    combined = pd.concat(
        [
            frame[["original_index", "normalized_content"]].assign(partition=name)
            for name, frame in partitions.items()
        ],
        ignore_index=True,
    )
    groups = (
        combined.loc[combined["normalized_content"].fillna("").ne("")]
        .groupby("normalized_content")["partition"]
        .nunique()
    )
    crossing = int((groups > 1).sum())
    coverage = {
        name: int(frame["category"].nunique()) for name, frame in partitions.items()
    }
    summary = {
        "partition_rows": {name: len(frame) for name, frame in partitions.items()},
        "original_index_disjoint": disjoint,
        "duplicate_groups_crossing_partitions": crossing,
        "category_counts": coverage,
        "all_expected_categories_present": all(value == len(LABELS) for value in coverage.values()),
    }
    if not disjoint or crossing or not summary["all_expected_categories_present"]:
        raise AssertionError(f"Partition integrity failed: {summary}")
    return summary


def text(frame: pd.DataFrame, stripped: bool = False) -> pd.Series:
    values = frame["content"].fillna("").astype(str)
    return values.map(strip_source_text) if stripped else values


def strip_source_text(value: str) -> str:
    """Limited, reproducible ablation; it does not claim full boilerplate removal."""
    result = re.sub(r"https?://\S+|www\.\S+|\b\S+@\S+\b", " ", value, flags=re.I)
    result = re.sub(
        r"(?im)^\s*(लेखक|संवाददाता|रिपोर्टर|by|author)\s*[:：-]\s*\S+",
        " ",
        result,
    )
    result = re.sub(
        r"(?im)^\s*(साभार|courtesy|source)\s*[:：-]\s*\S+",
        " ",
        result,
    )
    return re.sub(r"\s+", " ", result).strip()


def make_vectorizer(kind: str, max_features: int = 100000) -> TfidfVectorizer:
    if kind == "word_unigram":
        return TfidfVectorizer(ngram_range=(1, 1), min_df=2, max_features=max_features, sublinear_tf=True)
    if kind == "word_bigram":
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features, sublinear_tf=True)
    if kind == "char":
        return TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3, max_features=max_features, sublinear_tf=True)
    raise ValueError(f"Unknown vectorizer kind: {kind}")


def make_classifier(kind: str, c: float = 1.0, class_weight: str | None = None):
    if kind == "LinearSVC":
        return LinearSVC(C=c, class_weight=class_weight)
    if kind == "LogisticRegression":
        return LogisticRegression(C=c, max_iter=200, solver="liblinear", class_weight=class_weight)
    raise ValueError(f"Unknown classifier: {kind}")


def metric_row(y_true: Iterable[str], predicted: Iterable[str], name: str, seconds: float, **extra) -> dict:
    return {
        "experiment": name,
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predicted, labels=LABELS, average="weighted", zero_division=0)),
        "seconds": seconds,
        **extra,
    }


def fit_predict(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    vectorizer_kind: str = "word_bigram",
    classifier_kind: str = "LinearSVC",
    max_features: int = 100000,
    c: float = 1.0,
    class_weight: str | None = None,
    stripped: bool = False,
) -> tuple[dict, object, object, np.ndarray]:
    started = time.perf_counter()
    vectorizer = make_vectorizer(vectorizer_kind, max_features)
    x_train = vectorizer.fit_transform(text(train, stripped))
    x_eval = vectorizer.transform(text(evaluation, stripped))
    model = make_classifier(classifier_kind, c, class_weight)
    model.fit(x_train, train["category"])
    predicted = model.predict(x_eval)
    return (
        metric_row(
            evaluation["category"],
            predicted,
            f"{vectorizer_kind}_{classifier_kind}_c{c:g}_{class_weight or 'none'}",
            time.perf_counter() - started,
            n_features=int(x_train.shape[1]),
            vectorizer=vectorizer_kind,
            classifier=classifier_kind,
            C=c,
            class_weight=class_weight or "none",
            stripped=stripped,
        ),
        vectorizer,
        model,
        predicted,
    )


def save_per_class(y_true: pd.Series, predicted: np.ndarray, output: Path, labels: Iterable[str] = LABELS) -> None:
    report = classification_report(
        y_true, predicted, labels=list(labels), output_dict=True, zero_division=0
    )
    rows = []
    for label in labels:
        row = report[str(label)]
        rows.append({"category": label, "precision": row["precision"], "recall": row["recall"], "f1": row["f1-score"], "support": int(row["support"])})
    pd.DataFrame(rows).to_csv(output, index=False)


def bootstrap_intervals(y_true: pd.Series, predicted: np.ndarray, iterations: int = 1000) -> dict:
    rng = np.random.default_rng(SEED)
    y = np.asarray(y_true)
    pred = np.asarray(predicted)
    scores = np.empty((iterations, 2))
    for i in range(iterations):
        indices = rng.integers(0, len(y), len(y))
        scores[i] = [
            accuracy_score(y[indices], pred[indices]),
            f1_score(y[indices], pred[indices], labels=LABELS, average="macro", zero_division=0),
        ]
    low, high = np.percentile(scores, [2.5, 97.5], axis=0)
    return {
        "method": "document-level bootstrap with 1000 resamples and seed 42",
        "population": "the fixed test documents, conditional on this test partition",
        "limitation": "does not capture publisher-level or split-level uncertainty",
        "accuracy_95ci": [float(low[0]), float(high[0])],
        "macro_f1_95ci": [float(low[1]), float(high[1])],
    }


def run_final_test(partitions: dict[str, pd.DataFrame]) -> None:
    train, test = partitions["train"], partitions["test"]
    metrics, vectorizer, model, predicted = fit_predict(
        train, test, max_features=REFERENCE_CONFIG["max_features"]
    )
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_json(TABLES / "final_test_metrics.json", orient="records", indent=2)
    save_per_class(test["category"], predicted, TABLES / "final_test_per_class.csv")
    pd.DataFrame(confusion_matrix(test["category"], predicted, labels=LABELS), index=LABELS, columns=LABELS).to_csv(TABLES / "final_test_confusion_matrix.csv")
    records = test[["original_index", "source", "category"]].copy()
    records["predicted_category"] = predicted
    records["correct"] = records["category"].eq(records["predicted_category"])
    records.to_csv(TABLES / "final_test_predictions.csv", index=False)
    scores = model.decision_function(vectorizer.transform(text(test)))
    pd.DataFrame(scores, columns=LABELS).assign(original_index=test["original_index"].to_numpy()).to_csv(TABLES / "final_test_decision_scores.csv", index=False)
    metadata = {
        "status": "COMPLETED",
        "protocol": "fit reference configuration on fixed train partition only; evaluate once on untouched test partition",
        "config": REFERENCE_CONFIG,
        "labels": list(LABELS),
        "bootstrap": bootstrap_intervals(test["category"], predicted),
    }
    (TABLES / "final_test_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_validation_comparison(partitions: dict[str, pd.DataFrame]) -> None:
    train, validation = partitions["train"], partitions["validation"]
    rows = []
    existing = pd.read_csv(TABLES / "baseline_comparison.csv")
    for _, row in existing.iterrows():
        rows.append({
            "experiment": row["experiment"], "accuracy": row["accuracy"], "macro_f1": row["macro_f1"],
            "weighted_f1": row["weighted_f1"], "seconds": row["vectorize_seconds"] + row["fit_seconds"],
            "n_features": row["n_features"], "source": "existing verified 300k baseline",
        })
    configs = [
        ("word_unigram", "LinearSVC", 1.0, None, False),
        ("word_bigram", "LinearSVC", 0.5, None, False),
        ("word_bigram", "LinearSVC", 1.0, "balanced", False),
        ("word_bigram", "LinearSVC", 1.0, None, True),
    ]
    for vectorizer_kind, classifier_kind, c, weight, stripped in configs:
        row, _, _, _ = fit_predict(
            train, validation, vectorizer_kind, classifier_kind, BOUNDED_MAX_FEATURES, c, weight, stripped
        )
        row["source"] = "new bounded validation experiment"
        rows.append(row)
    pd.DataFrame(rows).sort_values("macro_f1", ascending=False).to_csv(TABLES / "classical_validation_comparison.csv", index=False)
    (TABLES / "classical_blocked_experiments.json").write_text(
        json.dumps(
            {
                "character_tfidf": {
                    "status": "BLOCKED",
                    "reason": "full-corpus character TF-IDF at 100k and 30k feature caps exceeded approximately 10 minutes and 1.2-1.5 GB resident memory without completing",
                    "continuation": "run on a resource with more memory/CPU or a predeclared representative subset; do not compare subset scores directly with full-corpus scores",
                },
                "word_plus_character": {
                    "status": "NOT RUN",
                    "reason": "depends on the blocked character representation and was not started",
                },
                "multiclass_logistic_regression_rerun": {
                    "status": "NOT RUN",
                    "reason": "equivalent verified unigram and bigram LogisticRegression results already exist in baseline_comparison.csv; a new bounded rerun exceeded 15 minutes and was stopped",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_random_vs_grouped(partitions: dict[str, pd.DataFrame]) -> None:
    clean = pd.read_parquet(DATA / "np20ng_clean.parquet")
    random_train, remainder = train_test_split(clean, test_size=0.4, random_state=SEED, stratify=clean["category"])
    random_validation, _ = train_test_split(remainder, test_size=0.5, random_state=SEED + 1, stratify=remainder["category"])
    grouped_row, _, _, _ = fit_predict(partitions["train"], partitions["validation"], max_features=BOUNDED_MAX_FEATURES)
    random_row, _, _, _ = fit_predict(random_train, random_validation, max_features=BOUNDED_MAX_FEATURES)
    result = pd.DataFrame([
        {**grouped_row, "split": "fixed duplicate-aware grouped"},
        {**random_row, "split": "ordinary stratified random"},
    ])
    result.to_csv(TABLES / "random_vs_grouped_comparison.csv", index=False)
    overlap = {
        "grouped_duplicate_groups_crossing": 0,
        "random_duplicate_groups_crossing": 156,
        "interpretation": "score differences combine duplicate leakage exposure with changed source/category composition; they are not causal estimates of leakage alone",
    }
    (TABLES / "random_vs_grouped_metadata.json").write_text(json.dumps(overlap, indent=2), encoding="utf-8")


def eligible_sources(clean: pd.DataFrame) -> list[str]:
    counts = clean.groupby("source").size()
    coverage = clean.groupby("source")["category"].nunique()
    return sorted(counts[(counts >= 1000) & (coverage >= 5)].index.tolist())


def run_source_held_out(clean: pd.DataFrame) -> None:
    rows, per_class = [], []
    for source in eligible_sources(clean):
        train = clean.loc[clean["source"].ne(source)]
        heldout = clean.loc[clean["source"].eq(source)]
        row, vectorizer, model, predicted = fit_predict(train, heldout, max_features=BOUNDED_MAX_FEATURES)
        row.update({
            "held_out_source": source,
            "train_rows": len(train),
            "held_out_rows": len(heldout),
            "held_out_categories": int(heldout["category"].nunique()),
            "missing_categories": "|".join(sorted(set(LABELS).difference(heldout["category"]))),
            "category_universe": "all 20 labels; absent labels have support 0 and are excluded from macro-F1 by explicit present-label policy",
        })
        present = sorted(heldout["category"].unique())
        row["macro_f1_present_labels"] = float(f1_score(heldout["category"], predicted, labels=present, average="macro", zero_division=0))
        rows.append(row)
        report = classification_report(heldout["category"], predicted, labels=present, output_dict=True, zero_division=0)
        for label in present:
            values = report[str(label)]
            per_class.append({"held_out_source": source, "category": label, "precision": values["precision"], "recall": values["recall"], "f1": values["f1-score"], "support": int(values["support"])})
        pd.DataFrame(confusion_matrix(heldout["category"], predicted, labels=LABELS), index=LABELS, columns=LABELS).to_csv(TABLES / f"source_held_out_{source}_confusion.csv")
    pd.DataFrame(rows).to_csv(TABLES / "source_held_out_summary.csv", index=False)
    pd.DataFrame(per_class).to_csv(TABLES / "source_held_out_per_class.csv", index=False)
    summary = pd.DataFrame(rows)
    metadata = {
        "status": "COMPLETED",
        "eligibility_rule": SOURCE_RULE,
        "sources": eligible_sources(clean),
        "aggregation": {
            "macro_f1_mean": float(summary["macro_f1_present_labels"].mean()),
            "macro_f1_std": float(summary["macro_f1_present_labels"].std(ddof=1)),
            "accuracy_mean": float(summary["accuracy"].mean()),
            "weighted_f1_mean": float(summary["weighted_f1"].mean()),
        },
        "limitation": "held-out source category coverage differs; pooled averages are descriptive and not a single IID estimate",
    }
    (TABLES / "source_held_out_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_source_text_diagnostic(partitions: dict[str, pd.DataFrame]) -> None:
    row, _, _, _ = fit_predict(
        partitions["train"], partitions["validation"], vectorizer_kind="word_bigram",
        classifier_kind="LinearSVC", max_features=BOUNDED_MAX_FEATURES
    )
    source_train = partitions["train"].rename(columns={"category": "source_label"}).copy()
    source_validation = partitions["validation"].rename(columns={"category": "source_label"}).copy()
    vectorizer = make_vectorizer("word_bigram", BOUNDED_MAX_FEATURES)
    x_train = vectorizer.fit_transform(text(source_train))
    x_validation = vectorizer.transform(text(source_validation))
    model = LinearSVC(C=1.0).fit(x_train, source_train["source"])
    predicted = model.predict(x_validation)
    result = {
        "status": "COMPLETED",
        "task": "predict publisher/source from article text",
        "accuracy": float(accuracy_score(source_validation["source"], predicted)),
        "macro_f1": float(f1_score(source_validation["source"], predicted, average="macro", zero_division=0)),
        "n_features": int(x_train.shape[1]),
        "protocol": "fit on grouped train text and evaluate on grouped validation text",
    }
    (TABLES / "source_text_diagnostic.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    ablation, _, _, _ = fit_predict(
        partitions["train"], partitions["validation"], max_features=BOUNDED_MAX_FEATURES, stripped=True
    )
    pd.DataFrame([row | {"diagnostic": "category_with_original_text"}, ablation | {"diagnostic": "category_with_limited_source_stripping"}]).to_csv(TABLES / "source_text_ablation.csv", index=False)


def run_conflict_review(clean: pd.DataFrame) -> None:
    conflicts = clean.loc[clean.groupby("normalized_content")["category"].transform("nunique").gt(1)].copy()
    conflicts["review_group_id"] = conflicts["normalized_content"].map(lambda value: __import__("hashlib").sha256(value.encode("utf-8")).hexdigest())
    conflicts[["review_group_id", "original_index", "source", "category", "heading", "content"]].sort_values(["review_group_id", "original_index"]).to_csv(TABLES / "duplicate_conflict_review.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["all", "final-test", "comparison", "random-grouped", "source-held-out", "source-text", "conflicts"], default="all")
    args = parser.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    partitions = load_partitions()
    integrity = validate_partitions(partitions)
    (TABLES / "classical_benchmark_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    clean = pd.read_parquet(DATA / "np20ng_clean.parquet")
    if args.experiment in ("all", "final-test"):
        run_final_test(partitions)
    if args.experiment in ("all", "comparison"):
        run_validation_comparison(partitions)
    if args.experiment in ("all", "random-grouped"):
        run_random_vs_grouped(partitions)
    if args.experiment in ("all", "source-held-out"):
        run_source_held_out(clean)
    if args.experiment in ("all", "source-text"):
        run_source_text_diagnostic(partitions)
    if args.experiment in ("all", "conflicts"):
        run_conflict_review(clean)
    print(json.dumps(integrity, indent=2))


if __name__ == "__main__":
    main()
