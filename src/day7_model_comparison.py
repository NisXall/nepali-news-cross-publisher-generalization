"""Day 7 consolidation of comparable validation results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_comparison() -> pd.DataFrame:
    baseline = pd.read_csv(ROOT / "results" / "tables" / "baseline_comparison.csv")
    rows = baseline.copy()
    day5 = ROOT / "results" / "tables" / "day5_training_size.csv"
    if day5.exists():
        size_results = pd.read_csv(day5)
        best = size_results.loc[size_results["fraction"].idxmax()]
        rows = pd.concat(
            [
                rows,
                pd.DataFrame(
                    [
                        {
                            "model": "LinearSVC bigram TF-IDF (100% training)",
                            "accuracy": best["accuracy"],
                            "macro_f1": best["macro_f1"],
                            "weighted_f1": best["weighted_f1"],
                            "split": "fixed validation",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return rows


def write_comparison() -> pd.DataFrame:
    comparison = load_comparison()
    output = ROOT / "results" / "tables" / "day7_model_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output, index=False)
    return comparison


if __name__ == "__main__":
    print(write_comparison().to_string(index=False))
