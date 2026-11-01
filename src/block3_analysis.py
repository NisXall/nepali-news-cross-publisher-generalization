"""Block 3 diagnostics from saved A/B/C predictions and manifests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
MANIFESTS = TABLES / "matched_50k_manifests"
C_MANIFESTS = TABLES / "matched_50k_c_manifests"


def metrics(frame: pd.DataFrame, labels: list[str] | None = None) -> dict[str, float]:
    labels = labels or sorted(set(frame["category"]) | set(frame["predicted_category"]))
    return {
        "rows": int(len(frame)),
        "accuracy": float(accuracy_score(frame["category"], frame["predicted_category"])),
        "macro_f1": float(
            f1_score(frame["category"], frame["predicted_category"], labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(frame["category"], frame["predicted_category"], labels=labels, average="weighted", zero_division=0)
        ),
    }


def prediction_sets() -> list[tuple[str, str, Path, Path]]:
    entries = [
        ("A", "TF-IDF LinearSVC", TABLES / "matched50k_tfidf_A_test_predictions.csv", MANIFESTS / "A_train.parquet"),
        ("B", "TF-IDF LinearSVC", TABLES / "matched50k_tfidf_B_test_predictions.csv", MANIFESTS / "B_train.parquet"),
        ("A", "MiniLM LinearSVC", TABLES / "matched50k_minilm_A_test_predictions.csv", MANIFESTS / "A_train.parquet"),
        ("B", "MiniLM LinearSVC", TABLES / "matched50k_minilm_B_test_predictions.csv", MANIFESTS / "B_train.parquet"),
        ("A", "XLM-RoBERTa-base", TABLES / "matched50k_xlmr_A_test_predictions.csv", MANIFESTS / "A_train.parquet"),
        ("B", "XLM-RoBERTa-base", TABLES / "matched50k_xlmr_B_test_predictions.csv", MANIFESTS / "B_train.parquet"),
    ]
    for path in sorted(TABLES.glob("matched50k_xlmr_C_*_test_predictions.csv")):
        source = path.name.removeprefix("matched50k_xlmr_C_").removesuffix("_test_predictions.csv")
        entries.append((f"C_{source}", "XLM-RoBERTa-base", path, C_MANIFESTS / f"{source}_train.parquet"))
    return entries


def reweighted_metrics(frame: pd.DataFrame, train: pd.DataFrame) -> dict[str, object]:
    train_prior = train["category"].value_counts(normalize=True)
    test_counts = frame["category"].value_counts()
    usable = sorted(set(train_prior.index) & set(test_counts.index))
    target = train_prior.reindex(usable).fillna(0.0).to_numpy()
    target = target / target.sum()
    observed = test_counts.reindex(usable).to_numpy(dtype=float)
    weights = target / (observed / len(frame))
    index = {label: i for i, label in enumerate(usable)}
    confusion = np.zeros((len(usable), len(usable)), dtype=float)
    for truth, predicted, weight in zip(
        frame["category"], frame["predicted_category"], frame["category"].map(lambda x: weights[index[x]] if x in index else 0.0)
    ):
        if truth in index and predicted in index:
            confusion[index[truth], index[predicted]] += weight
    tp = np.diag(confusion)
    precision = np.divide(tp, confusion.sum(axis=0), out=np.zeros(len(usable)), where=confusion.sum(axis=0) != 0)
    recall = np.divide(tp, confusion.sum(axis=1), out=np.zeros(len(usable)), where=confusion.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(usable)), where=(precision + recall) != 0)
    return {
        "rows": int(len(frame)),
        "target_prior": {label: float(target[i]) for i, label in enumerate(usable)},
        "excluded_test_only_categories": sorted(set(test_counts.index) - set(usable)),
        "weighted_accuracy": float(tp.sum() / confusion.sum()),
        "weighted_macro_f1": float(f1.mean()),
        "weighted_category_support": {label: float(confusion[i].sum()) for i, label in enumerate(usable)},
    }


def main() -> None:
    degradation = pd.read_csv(TABLES / "matched_50k_abc_consolidated.csv")
    degradation["test_macro_f1"] = pd.to_numeric(degradation["test_macro_f1"])
    a_b = degradation[degradation["condition"].isin(["A", "B"])].pivot_table(
        index="model", columns="condition", values=["test_accuracy", "test_macro_f1", "test_weighted_f1"], aggfunc="first"
    )
    a_b.columns = [f"{metric}_{condition}" for metric, condition in a_b.columns]
    a_b["delta_B_minus_A_macro_f1"] = a_b["test_macro_f1_B"] - a_b["test_macro_f1_A"]
    a_b["delta_B_minus_A_accuracy"] = a_b["test_accuracy_B"] - a_b["test_accuracy_A"]
    a_b.reset_index().to_csv(TABLES / "matched_50k_ab_degradation.csv", index=False)

    source_rows = []
    error_samples = []
    reweighted = []
    for condition, model, path, train_path in prediction_sets():
        frame = pd.read_csv(path, usecols=["original_index", "source", "category", "predicted_category"])
        frame["correct"] = frame["category"] == frame["predicted_category"]
        source_group = frame.groupby("source", sort=True)
        for source, group in source_group:
            row = {"condition": condition, "model": model, "source": source, **metrics(group)}
            row["error_rate"] = float(1 - row["accuracy"])
            source_rows.append(row)
        worst = (
            pd.DataFrame(source_rows)
            .query("condition == @condition and model == @model")
            .sort_values(["macro_f1", "rows"])
            .iloc[0]["source"]
        )
        errors = frame[(frame["source"] == worst) & ~frame["correct"]].head(50)
        for record in errors.to_dict("records"):
            error_samples.append(
                {
                    "condition": condition,
                    "model": model,
                    "source": worst,
                    "original_index": int(record["original_index"]),
                    "true_category": record["category"],
                    "predicted_category": record["predicted_category"],
                }
            )
        train = pd.read_parquet(train_path, columns=["category"])
        reweighted.append(
            {
                "condition": condition,
                "model": model,
                "diagnostic": "test-side prior matching; no retraining",
                "raw_metrics": metrics(frame),
                "reweighted_metrics": reweighted_metrics(frame, train),
            }
        )

    pd.DataFrame(source_rows).to_csv(TABLES / "matched_50k_source_degradation.csv", index=False)
    pd.DataFrame(error_samples).to_csv(TABLES / "matched_50k_error_samples.csv", index=False)
    (TABLES / "matched_50k_test_prior_reweighting.json").write_text(
        json.dumps(
            {
                "method": "test-side reweighting to the corresponding training category prior",
                "warning": "Diagnostic only; original test metrics remain the primary evaluation.",
                "results": reweighted,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
