"""Leakage-safe calibration and abstention for the frozen reference classifier."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss

from src.classical_benchmark import LABELS, make_vectorizer, text


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
MODELS = ROOT / "results" / "models"
SEED = 42
TARGET_SELECTIVE_RISK = 0.20


def load_partitions() -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_parquet(DATA / f"{name}.parquet")
        for name in ("train", "validation", "test")
    }


def probabilities_from_scores(scores: np.ndarray, calibrator: LogisticRegression) -> np.ndarray:
    probabilities = calibrator.predict_proba(scores)
    if probabilities.shape[1] != len(LABELS) or not np.isfinite(probabilities).all():
        raise ValueError("Calibrator returned invalid probability output")
    return probabilities


def calibration_metrics(
    y_true: pd.Series, probabilities: np.ndarray, bins: int = 10
) -> tuple[dict, pd.DataFrame]:
    y_ids = pd.Categorical(y_true, categories=LABELS).codes
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predicted == y_ids).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    weighted_gap = 0.0
    for index in range(bins):
        mask = (confidence >= edges[index]) & (
            confidence <= edges[index + 1]
            if index == bins - 1
            else confidence < edges[index + 1]
        )
        count = int(mask.sum())
        if count:
            mean_confidence = float(confidence[mask].mean())
            accuracy = float(correct[mask].mean())
            weighted_gap += count / len(confidence) * abs(mean_confidence - accuracy)
        else:
            mean_confidence = None
            accuracy = None
        rows.append(
            {
                "bin": index,
                "lower": edges[index],
                "upper": edges[index + 1],
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    metrics = {
        "accuracy": float(accuracy_score(y_true, [LABELS[i] for i in predicted])),
        "macro_f1": float(
            f1_score(y_true, [LABELS[i] for i in predicted], labels=LABELS, average="macro", zero_division=0)
        ),
        "brier_score": float(
            np.mean([brier_score_loss((y_ids == i).astype(int), probabilities[:, i]) for i in range(len(LABELS))])
        ),
        "log_loss": float(log_loss(y_true, probabilities, labels=list(LABELS))),
        "ece": float(weighted_gap),
        "binning": "10 equal-width confidence bins",
    }
    return metrics, pd.DataFrame(rows)


def select_abstention_threshold(
    y_true: pd.Series, probabilities: np.ndarray, target_risk: float = TARGET_SELECTIVE_RISK
) -> dict:
    predicted = probabilities.argmax(axis=1)
    y_ids = pd.Categorical(y_true, categories=LABELS).codes
    confidence = probabilities.max(axis=1)
    candidates = np.unique(np.r_[0.0, confidence])
    feasible = []
    for threshold in candidates:
        accepted = confidence >= threshold
        if not accepted.any():
            continue
        coverage = float(accepted.mean())
        risk = float(np.mean(predicted[accepted] != y_ids[accepted]))
        feasible.append((risk <= target_risk, coverage, -risk, float(threshold), risk))
    valid = [row for row in feasible if row[0]]
    if not valid:
        selected = min(feasible, key=lambda row: (row[4], -row[1]))
        criterion = "minimum validation selective risk fallback; target was unattainable"
    else:
        selected = max(valid, key=lambda row: (row[1], row[2], -row[3]))
        criterion = f"maximum validation coverage subject to selective risk <= {target_risk:.2f}"
    _, coverage, _, threshold, risk = selected
    return {
        "threshold": threshold,
        "target_selective_risk": target_risk,
        "selection_criterion": criterion,
        "validation_coverage": coverage,
        "validation_selective_risk": risk,
        "validation_abstained_percent": 100.0 * (1.0 - coverage),
    }


def abstention_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = probabilities.argmax(axis=1)
    y_ids = pd.Categorical(y_true, categories=LABELS).codes
    accepted = probabilities.max(axis=1) >= threshold
    if not accepted.any():
        raise ValueError("Abstention threshold accepts no examples")
    accepted_labels = [LABELS[i] for i in predicted[accepted]]
    accepted_true = y_true.iloc[np.flatnonzero(accepted)]
    return {
        "threshold": float(threshold),
        "accepted": int(accepted.sum()),
        "total": int(len(y_true)),
        "coverage": float(accepted.mean()),
        "abstained_percent": float(100.0 * (1.0 - accepted.mean())),
        "accuracy_accepted": float(accuracy_score(accepted_true, accepted_labels)),
        "macro_f1_accepted": float(
            f1_score(accepted_true, accepted_labels, labels=LABELS, average="macro", zero_division=0)
        ),
        "selective_risk": float(np.mean(predicted[accepted] != y_ids[accepted])),
    }


def save_reliability_plot(bins: pd.DataFrame, path: Path) -> None:
    nonempty = bins[bins["count"] > 0]
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axis.plot(nonempty["mean_confidence"], nonempty["accuracy"], "o-", label="Calibrated model")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run() -> dict:
    started = time.perf_counter()
    partitions = load_partitions()
    vectorizer = make_vectorizer("word_bigram", max_features=300000)
    x_train = vectorizer.fit_transform(text(partitions["train"]))
    x_validation = vectorizer.transform(text(partitions["validation"]))
    x_test = vectorizer.transform(text(partitions["test"]))
    from sklearn.svm import LinearSVC

    base_classifier = LinearSVC(C=1.0)
    base_classifier.fit(x_train, partitions["train"]["category"])
    validation_scores = base_classifier.decision_function(x_validation)
    test_scores = base_classifier.decision_function(x_test)
    calibrator = LogisticRegression(max_iter=500, solver="lbfgs", random_state=SEED)
    calibrator.fit(validation_scores, partitions["validation"]["category"])
    validation_probabilities = probabilities_from_scores(validation_scores, calibrator)
    test_probabilities = probabilities_from_scores(test_scores, calibrator)
    validation_calibration, validation_bins = calibration_metrics(
        partitions["validation"]["category"], validation_probabilities
    )
    test_calibration, test_bins = calibration_metrics(
        partitions["test"]["category"], test_probabilities
    )
    policy = select_abstention_threshold(partitions["validation"]["category"], validation_probabilities)
    test_abstention = abstention_metrics(
        partitions["test"]["category"], test_probabilities, policy["threshold"]
    )

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    validation_bins.to_csv(TABLES / "calibration_validation_reliability.csv", index=False)
    test_bins.to_csv(TABLES / "calibration_test_reliability.csv", index=False)
    save_reliability_plot(test_bins, FIGURES / "calibration_reliability.png")
    metadata = {
        "status": "COMPLETED",
        "model": "TF-IDF word unigram/bigram + LinearSVC(C=1.0)",
        "seed": SEED,
        "protocol": "base classifier fit on train; calibrator fit on validation decision scores; test evaluated once",
        "calibration_method": "multiclass logistic regression (softmax calibration) on validation scores",
        "validation_calibration": validation_calibration,
        "test_calibration": test_calibration,
        "abstention_policy": policy,
        "test_abstention": test_abstention,
        "runtime_seconds": time.perf_counter() - started,
        "caveat": "Calibration on known sources does not guarantee calibration on unseen publishers or future data.",
    }
    (TABLES / "calibration_abstention_metrics.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": base_classifier,
            "calibrator": calibrator,
            "labels": list(LABELS),
            "threshold": policy["threshold"],
            "seed": SEED,
        },
        MODELS / "reference_calibrated_bundle.joblib",
    )
    return metadata


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
