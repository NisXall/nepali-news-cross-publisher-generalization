"""Day 5 nested training-size experiment."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
SEED = 42
FRACTIONS = (0.25, 0.50, 1.00)


def run_training_size_experiment(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    fractions: tuple[float, ...] = FRACTIONS,
) -> pd.DataFrame:
    ordered = train.sample(frac=1, random_state=SEED).reset_index(drop=True)
    results = []
    for fraction in fractions:
        subset = ordered.iloc[: int(len(ordered) * fraction)]
        started = time.perf_counter()
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=300000,
            sublinear_tf=True,
        )
        x_train = vectorizer.fit_transform(subset["content"].fillna(""))
        x_validation = vectorizer.transform(validation["content"].fillna(""))
        model = LinearSVC(C=1.0)
        model.fit(x_train, subset["category"])
        predicted = model.predict(x_validation)
        results.append(
            {
                "fraction": fraction,
                "train_rows": len(subset),
                "features": x_train.shape[1],
                "accuracy": accuracy_score(validation["category"], predicted),
                "macro_f1": f1_score(validation["category"], predicted, average="macro"),
                "weighted_f1": f1_score(
                    validation["category"], predicted, average="weighted"
                ),
                "seconds": time.perf_counter() - started,
            }
        )
    return pd.DataFrame(results)


def main() -> None:
    train = pd.read_parquet(ROOT / "data" / "processed" / "train.parquet")
    validation = pd.read_parquet(ROOT / "data" / "processed" / "validation.parquet")
    results = run_training_size_experiment(train, validation)
    output = ROOT / "results" / "tables" / "day5_training_size.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
