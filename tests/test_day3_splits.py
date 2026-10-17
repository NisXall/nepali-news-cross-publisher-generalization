import pandas as pd

from src.day3_evaluation import duplicate_audit, load_partitions


def test_grouped_partitions_have_no_cross_partition_duplicate_groups():
    summary = duplicate_audit(load_partitions())[1]
    assert summary["groups_crossing_partitions"] == 0


def test_partition_labels_are_disjoint():
    partitions = load_partitions()
    indices = [set(frame["original_index"]) for frame in partitions.values()]
    assert indices[0].isdisjoint(indices[1])
    assert indices[0].isdisjoint(indices[2])
    assert indices[1].isdisjoint(indices[2])
