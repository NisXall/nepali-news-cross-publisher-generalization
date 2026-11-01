"""Per-class metrics and independent bootstrap intervals for saved predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def _metrics(frame: pd.DataFrame, labels: list[str]) -> dict[str, float]:
    return {
        "rows": int(len(frame)),
        "accuracy": float(accuracy_score(frame["category"], frame["predicted_category"])),
        "macro_f1": float(
            f1_score(
                frame["category"],
                frame["predicted_category"],
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                frame["category"],
                frame["predicted_category"],
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def _per_class(frame: pd.DataFrame, condition: str, model: str) -> pd.DataFrame:
    labels = sorted(set(frame["category"]) | set(frame["predicted_category"]))
    precision, recall, f1, support = precision_recall_fscore_support(
        frame["category"],
        frame["predicted_category"],
        labels=labels,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "condition": condition,
            "model": model,
            "category": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def _bootstrap(
    frame: pd.DataFrame,
    condition: str,
    model: str,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    labels = sorted(set(frame["category"]) | set(frame["predicted_category"]))
    truth = frame["category"].to_numpy()
    predicted = frame["predicted_category"].to_numpy()
    label_to_index = {label: i for i, label in enumerate(labels)}
    truth_index = np.array([label_to_index[label] for label in truth])
    predicted_index = np.array([label_to_index[label] for label in predicted])
    correct = truth_index == predicted_index
    pair_index = truth_index * len(labels) + predicted_index
    rng = np.random.default_rng(seed)
    accuracies = np.empty(resamples)
    macro_f1s = np.empty(resamples)
    for i in range(resamples):
        indexes = rng.integers(0, len(frame), len(frame))
        accuracies[i] = float(correct[indexes].mean())
        confusion = np.bincount(
            pair_index[indexes], minlength=len(labels) * len(labels)
        ).reshape(len(labels), len(labels))
        true_support = confusion.sum(axis=1)
        predicted_support = confusion.sum(axis=0)
        denominators = true_support + predicted_support
        per_class_f1 = np.divide(
            2 * np.diag(confusion),
            denominators,
            out=np.zeros(len(labels), dtype=float),
            where=denominators != 0,
        )
        macro_f1s[i] = float(per_class_f1.mean())
    return {
        "condition": condition,
        "model": model,
        "rows": int(len(frame)),
        "resamples": resamples,
        "seed": seed,
        "point_estimate": _metrics(frame, labels),
        "accuracy_ci95": [float(np.quantile(accuracies, 0.025)), float(np.quantile(accuracies, 0.975))],
        "macro_f1_ci95": [float(np.quantile(macro_f1s, 0.025)), float(np.quantile(macro_f1s, 0.975))],
    }


def collect_predictions(table_dir: Path) -> list[tuple[str, str, Path]]:
    entries = [
        ("A", "TF-IDF LinearSVC", table_dir / "matched50k_tfidf_A_test_predictions.csv"),
        ("B", "TF-IDF LinearSVC", table_dir / "matched50k_tfidf_B_test_predictions.csv"),
        ("A", "MiniLM LinearSVC", table_dir / "matched50k_minilm_A_test_predictions.csv"),
        ("B", "MiniLM LinearSVC", table_dir / "matched50k_minilm_B_test_predictions.csv"),
        ("A", "XLM-RoBERTa-base", table_dir / "matched50k_xlmr_A_test_predictions.csv"),
        ("B", "XLM-RoBERTa-base", table_dir / "matched50k_xlmr_B_test_predictions.csv"),
    ]
    for path in sorted(table_dir.glob("matched50k_xlmr_C_*_test_predictions.csv")):
        source = path.name.removeprefix("matched50k_xlmr_C_").removesuffix("_test_predictions.csv")
        entries.append((f"C_{source}", "XLM-RoBERTa-base", path))
    return [entry for entry in entries if entry[2].exists()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.resamples < 100:
        raise ValueError("--resamples must be at least 100")

    per_class_frames = []
    bootstrap = []
    coverage = []
    for condition, model, path in collect_predictions(args.table_dir):
        frame = pd.read_csv(path, usecols=["category", "predicted_category"])
        if frame.empty or frame[["category", "predicted_category"]].isna().any().any():
            raise ValueError(f"Invalid prediction file: {path}")
        per_class_frames.append(_per_class(frame, condition, model))
        bootstrap.append(_bootstrap(frame, condition, model, args.resamples, args.seed))
        coverage.append({"condition": condition, "model": model, "prediction_file": str(path), "rows": len(frame)})

    output_dir = args.table_dir
    pd.concat(per_class_frames, ignore_index=True).to_csv(
        output_dir / "matched_50k_per_class_metrics.csv", index=False
    )
    (output_dir / "matched_50k_independent_bootstrap.json").write_text(
        json.dumps(
            {
                "method": "independent document bootstrap; A and B have different test rows",
                "resamples": args.resamples,
                "seed": args.seed,
                "interval": "percentile 95% interval",
                "results": bootstrap,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "matched_50k_analysis_coverage.json").write_text(
        json.dumps(
            {
                "available_prediction_sets": coverage,
                "not_available": [
                    "Matched C TF-IDF per-document predictions",
                    "Matched C MiniLM per-document predictions",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
