import pandas as pd

from src.classical_benchmark import (
    LABELS,
    eligible_sources,
    strip_source_text,
    validate_partitions,
)


def test_partition_validation_matches_fixed_artifacts():
    partitions = {
        name: pd.read_parquet(f"data/processed/{name}.parquet")
        for name in ("train", "validation", "test")
    }
    summary = validate_partitions(partitions)
    assert summary["partition_rows"] == {
        "train": 138711,
        "validation": 46237,
        "test": 46238,
    }
    assert summary["duplicate_groups_crossing_partitions"] == 0
    assert summary["all_expected_categories_present"]


def test_strip_source_text_is_limited_and_deterministic():
    value = "लेखक: राम https://example.com नेपाल समाचार"
    assert strip_source_text(value) == "नेपाल समाचार"


def test_label_universe_is_stable():
    assert len(LABELS) == 20
    assert LABELS == tuple(sorted(LABELS))


def test_source_eligibility_uses_document_and_category_thresholds():
    rows = []
    for source, count, categories in (
        ("eligible", 1000, ["Arts", "Sports", "Politics", "Health", "World"]),
        ("small", 999, ["Arts", "Sports", "Politics", "Health", "World"]),
        ("narrow", 1000, ["Arts"]),
    ):
        for index in range(count):
            rows.append({"source": source, "category": categories[index % len(categories)]})
    frame = pd.DataFrame(rows)
    assert eligible_sources(frame) == ["eligible"]
