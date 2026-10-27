"""Fast MinHash/LSH near-duplicate audit with exact Jaccard verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

SHINGLE_SIZE = 5
PERMUTATIONS = 64
BANDS = 16
ROWS_PER_BAND = 4
JACCARD_THRESHOLD = 0.8
CANDIDATE_THRESHOLD = 0.5
P = np.uint64(18446744073709551557)


def shingles(value: str, size: int = SHINGLE_SIZE) -> set[str]:
    words = value.split()
    if len(words) < size:
        return {value} if value else set()
    return {" ".join(words[i:i + size]) for i in range(len(words) - size + 1)}


def _hash_shingle(value: str) -> np.uint64:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return np.uint64(int.from_bytes(digest, "little"))


def _signature(shingle_hashes: np.ndarray, multipliers: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    values = (multipliers[:, None] * shingle_hashes[None, :] + offsets[:, None]) % P
    return values.min(axis=1)


def _union_find_groups(pairs: list[tuple[int, int]], size: int) -> list[list[int]]:
    parent = list(range(size))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for left, right in pairs:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
    groups: dict[int, list[int]] = {}
    for index in {item for pair in pairs for item in pair}:
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def audit(frame: pd.DataFrame, text_col: str, id_col: str, label_col: str, source_col: str) -> dict:
    started = time.perf_counter()
    values = frame[text_col].fillna("").astype(str).tolist()
    document_shingles = [shingles(value) for value in values]
    shingle_hashes = [
        np.asarray([_hash_shingle(item) for item in items], dtype=np.uint64)
        for items in document_shingles
    ]
    rng = np.random.default_rng(42)
    multipliers = rng.integers(1, np.iinfo(np.uint64).max, PERMUTATIONS, dtype=np.uint64)
    offsets = rng.integers(0, np.iinfo(np.uint64).max, PERMUTATIONS, dtype=np.uint64)
    buckets: dict[tuple[int, bytes], list[int]] = {}
    for index, hashes in enumerate(shingle_hashes):
        if not len(hashes):
            continue
        signature = _signature(hashes, multipliers, offsets)
        for band in range(BANDS):
            start = band * ROWS_PER_BAND
            key = (band, signature[start:start + ROWS_PER_BAND].tobytes())
            buckets.setdefault(key, []).append(index)
    candidates: set[tuple[int, int]] = set()
    for members in buckets.values():
        for left_offset, left in enumerate(members[:-1]):
            for right in members[left_offset + 1:]:
                candidates.add((min(left, right), max(left, right)))

    qualifying: list[tuple[int, int, float]] = []
    for left, right in candidates:
        union = document_shingles[left] | document_shingles[right]
        if not union:
            continue
        score = len(document_shingles[left] & document_shingles[right]) / len(union)
        if score >= JACCARD_THRESHOLD:
            qualifying.append((left, right, score))
    groups = _union_find_groups([(left, right) for left, right, _ in qualifying], len(frame))
    partition_columns = ["partition"] if "partition" in frame.columns else []
    group_rows = []
    for group in groups:
        members = frame.iloc[group]
        group_rows.append({
            "group_size": len(group),
            "indices": [str(value) for value in members[id_col]],
            "sources": sorted(set(str(value) for value in members[source_col])),
            "labels": sorted(set(str(value) for value in members[label_col])),
            "partitions": {
                column: sorted(set(str(value) for value in members[column]))
                for column in partition_columns
            },
        })
    return {
        "status": "COMPLETED",
        "method": "MinHash/LSH candidates at 0.5 followed by exact Jaccard verification at 0.8",
        "shingle_size_words": SHINGLE_SIZE,
        "minhash_permutations": PERMUTATIONS,
        "lsh_bands": BANDS,
        "lsh_rows_per_band": ROWS_PER_BAND,
        "candidate_threshold": CANDIDATE_THRESHOLD,
        "jaccard_threshold": JACCARD_THRESHOLD,
        "rows": len(frame),
        "candidate_pairs": len(candidates),
        "qualifying_pairs": len(qualifying),
        "max_group_size": max((len(group) for group in groups), default=0),
        "groups": sorted(group_rows, key=lambda row: (-row["group_size"], row["indices"][0])),
        "runtime_seconds": time.perf_counter() - started,
    }


def _self_test() -> None:
    base = " ".join(f"w{i}" for i in range(30))
    near = " ".join(f"x{i}" if i == 29 else f"w{i}" for i in range(30))
    far = " ".join(f"z{i}" for i in range(30))
    frame = pd.DataFrame({
        "id": ["base", "near", "far"],
        "text": [base, near, far],
        "label": ["A", "A", "B"],
        "source": ["one", "two", "three"],
    })
    result = audit(frame, "text", "id", "label", "source")
    assert result["qualifying_pairs"] == 1, result
    assert result["groups"][0]["indices"] == ["base", "near"], result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", type=Path)
    parser.add_argument("--text-col", default="normalized_content")
    parser.add_argument("--id-col", default="original_index")
    parser.add_argument("--label-col", default="category")
    parser.add_argument("--source-col", default="source")
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print("self-test passed")
        return
    if not args.manifest or not args.output:
        parser.error("--manifest and --output are required unless --self-test is used")
    frames = []
    for path in args.manifest:
        frame = pd.read_parquet(path)
        frame["partition"] = path.stem
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    result = audit(combined, args.text_col, args.id_col, args.label_col, args.source_col)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "rows", "candidate_pairs", "qualifying_pairs", "max_group_size", "runtime_seconds")}, indent=2))


if __name__ == "__main__":
    main()
