import pandas as pd

from src.matched_budget_benchmark import _shingles, create_manifests


def test_word_shingles_are_deterministic():
    assert _shingles("a b c d e f") == {"a b c d e", "b c d e f"}
    assert _shingles("") == set()


def test_matched_manifests_have_fixed_sizes_and_all_labels():
    metadata = create_manifests()
    assert metadata["fixed_partitions_preserved"]
    for condition in ("A", "B"):
        assert metadata["conditions"][condition]["rows"] == [50000, 10000, 10000]
        assert metadata["conditions"][condition]["categories"] == [20, 20, 20]
