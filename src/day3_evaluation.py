"""Audit and document the Day 3 evaluation protocol."""

from pathlib import Path
import json
import hashlib

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
SEED = 42


def load_partitions() -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_parquet(DATA / f"{name}.parquet")
        for name in ("train", "validation", "test")
    }


def duplicate_audit(partitions: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    rows = []
    for name, frame in partitions.items():
        rows.append(frame[["original_index", "normalized_content"]].assign(partition=name))
    all_rows = pd.concat(rows, ignore_index=True)
    nonempty = all_rows["normalized_content"].fillna("").ne("")
    group_partitions = (
        all_rows.loc[nonempty]
        .groupby("normalized_content")["partition"]
        .nunique()
    )
    crossing = group_partitions[group_partitions > 1]
    summary = {
        "normalized_duplicate_groups": int(
            all_rows.loc[nonempty, "normalized_content"].duplicated(keep=False).sum()
        ),
        "groups_crossing_partitions": int(crossing.size),
        "rows_in_groups_crossing_partitions": int(
            all_rows["normalized_content"].isin(crossing.index).sum()
        ),
    }
    group_table = (
        all_rows.loc[nonempty]
        .groupby("normalized_content")
        .agg(
            rows=("original_index", "size"),
            partitions=("partition", lambda values: "|".join(sorted(set(values)))),
        )
        .reset_index()
    )
    group_table = group_table[group_table["rows"] > 1].copy()
    group_table["group_id"] = group_table["normalized_content"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    group_table = group_table.drop(columns="normalized_content")
    group_table = group_table[["group_id", "rows", "partitions"]]
    return group_table, summary


def random_split_duplicate_audit(clean: pd.DataFrame) -> dict:
    train, remainder = train_test_split(
        clean, test_size=0.4, random_state=SEED, stratify=clean["category"]
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.5,
        random_state=SEED + 1,
        stratify=remainder["category"],
    )
    assigned = pd.concat(
        [
            train[["normalized_content"]].assign(partition="train"),
            validation[["normalized_content"]].assign(partition="validation"),
            test[["normalized_content"]].assign(partition="test"),
        ],
        ignore_index=True,
    )
    groups = (
        assigned[assigned["normalized_content"].ne("")]
        .groupby("normalized_content")["partition"]
        .nunique()
    )
    return {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "groups_crossing_partitions": int((groups > 1).sum()),
    }


def distribution_tables(partitions: dict[str, pd.DataFrame]) -> None:
    category = pd.concat(
        [
            frame["category"].value_counts().rename(name)
            for name, frame in partitions.items()
        ],
        axis=1,
    ).fillna(0).astype(int)
    source = pd.concat(
        [
            frame["source"].value_counts().rename(name)
            for name, frame in partitions.items()
        ],
        axis=1,
    ).fillna(0).astype(int)
    category.index.name = "category"
    source.index.name = "source"
    category.to_csv(TABLES / "day3_category_distribution.csv")
    source.to_csv(TABLES / "day3_source_distribution.csv")

    all_rows = pd.concat(partitions.values(), ignore_index=True)
    source_category = pd.crosstab(
        all_rows["source"],
        all_rows["category"],
        normalize="index",
    )
    source_category.to_csv(TABLES / "day3_source_category_proportions.csv")

    source_majority = (
        pd.concat([frame[["source", "category"]] for frame in partitions.values()])
        .groupby("source")["category"]
        .agg(lambda values: values.mode().iat[0])
    )
    diagnostic_accuracy = float(
        (all_rows["category"] == all_rows["source"].map(source_majority)).mean()
    )
    source_counts = all_rows.groupby("source")["category"].nunique().sort_values()
    eligible = source_counts[
        (all_rows.groupby("source").size() >= 1000) & (source_counts >= 5)
    ].index.tolist()
    metadata = {
        "source_only_majority_accuracy": diagnostic_accuracy,
        "source_category_counts": source_counts.to_dict(),
        "eligible_held_out_sources": eligible,
        "candidate_rule": "at least 1,000 documents and at least 5 observed categories",
    }
    (TABLES / "day3_source_diagnostic.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    ax = category.sort_values("train").plot.bar(figsize=(12, 5))
    ax.set_ylabel("Documents")
    ax.set_title("Category counts by evaluation partition")
    plt.tight_layout()
    plt.savefig(FIGURES / "day3_category_distribution.png", dpi=150)
    plt.close()


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    partitions = load_partitions()
    clean = pd.read_parquet(DATA / "np20ng_clean.parquet")
    group_table, grouped_summary = duplicate_audit(partitions)
    group_table.to_csv(TABLES / "day3_duplicate_group_audit.csv", index=False)
    random_summary = random_split_duplicate_audit(
        clean[["category", "normalized_content"]]
    )
    distribution_tables(partitions)
    metadata = {
        "seed": SEED,
        "partition_sizes": {name: len(frame) for name, frame in partitions.items()},
        "grouped_split_duplicate_audit": grouped_summary,
        "ordinary_random_split_duplicate_audit": random_summary,
    }
    (TABLES / "day3_split_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    if grouped_summary["groups_crossing_partitions"] != 0:
        raise AssertionError("The grouped split has duplicate groups crossing partitions.")


if __name__ == "__main__":
    main()
