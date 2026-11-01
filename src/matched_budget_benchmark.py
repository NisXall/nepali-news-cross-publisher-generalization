"""Matched-budget audit and model comparison for the Nepali benchmark."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from src.classical_benchmark import LABELS, make_vectorizer, text
from src.day8_neural_experiments import EMBEDDING_MODEL, encode_full_cached, load_partitions, texts
from src.classical_benchmark import eligible_sources

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
MANIFESTS = TABLES / "matched_50k_manifests"
C_MANIFESTS = TABLES / "matched_50k_c_manifests"
SEED = 42
TRAIN_ROWS = 50_000
EVALUATION_ROWS = 10_000
AUDIT_THRESHOLD = 0.8
SHINGLE_SIZE = 5
MINHASH_PERMUTATIONS = 64
LSH_BANDS = 8
LSH_ROWS = 8


def _hash64(value: str, seed: int) -> int:
    return int.from_bytes(
        hashlib.blake2b(value.encode("utf-8"), digest_size=8, person=seed.to_bytes(8, "little")).digest(),
        "little",
    )


def _shingles(value: str) -> set[str]:
    words = value.split()
    if len(words) < SHINGLE_SIZE:
        return {value} if value else set()
    return {" ".join(words[index:index + SHINGLE_SIZE]) for index in range(len(words) - SHINGLE_SIZE + 1)}


def run_near_duplicate_audit() -> dict:
    """Audit Jaccard-0.8 candidates with fixed word-shingle MinHash/LSH settings."""
    started = time.perf_counter()
    clean = pd.read_parquet(DATA / "np20ng_clean.parquet", columns=["original_index", "source", "normalized_content"])
    signatures = np.full((len(clean), MINHASH_PERMUTATIONS), np.iinfo(np.uint64).max, dtype=np.uint64)
    buckets: dict[tuple[int, int], list[int]] = {}
    shingle_cache: dict[int, set[str]] = {}
    for row_number, value in enumerate(clean["normalized_content"].fillna("").astype(str)):
        shingles = _shingles(value)
        if not shingles:
            continue
        shingle_cache[row_number] = shingles
        signature = np.array(
            [min(_hash64(shingle, seed) for shingle in shingles) for seed in range(MINHASH_PERMUTATIONS)],
            dtype=np.uint64,
        )
        signatures[row_number] = signature
        for band in range(LSH_BANDS):
            start = band * LSH_ROWS
            key = (band, hash(signature[start:start + LSH_ROWS].tobytes()))
            buckets.setdefault(key, []).append(row_number)

    candidates: set[tuple[int, int]] = set()
    for rows in buckets.values():
        if len(rows) < 2:
            continue
        for left_index, left in enumerate(rows[:-1]):
            for right in rows[left_index + 1:]:
                candidates.add((min(left, right), max(left, right)))

    groups: dict[int, set[int]] = {}
    qualifying_pairs = []
    for left, right in candidates:
        left_shingles = shingle_cache[left]
        right_shingles = shingle_cache[right]
        jaccard = len(left_shingles & right_shingles) / len(left_shingles | right_shingles)
        if jaccard >= AUDIT_THRESHOLD:
            qualifying_pairs.append((left, right, jaccard))
            groups.setdefault(left, set()).add(right)
            groups.setdefault(right, set()).add(left)

    fixed_partitions = {
        name: set(frame["original_index"])
        for name, frame in load_partitions().items()
    }
    index_to_partition = {
        index: name for name, indices in fixed_partitions.items() for index in indices
    }
    cross_partition_pairs = [
        {
            "left_original_index": int(clean.iloc[left]["original_index"]),
            "right_original_index": int(clean.iloc[right]["original_index"]),
            "left_partition": index_to_partition.get(clean.iloc[left]["original_index"], "clean-only"),
            "right_partition": index_to_partition.get(clean.iloc[right]["original_index"], "clean-only"),
            "left_source": clean.iloc[left]["source"],
            "right_source": clean.iloc[right]["source"],
            "jaccard": jaccard,
        }
        for left, right, jaccard in qualifying_pairs
        if index_to_partition.get(clean.iloc[left]["original_index"])
        != index_to_partition.get(clean.iloc[right]["original_index"])
    ]
    cross_source_pairs = [
        {
            "left_original_index": int(clean.iloc[left]["original_index"]),
            "right_original_index": int(clean.iloc[right]["original_index"]),
            "left_source": clean.iloc[left]["source"],
            "right_source": clean.iloc[right]["source"],
            "left_partition": index_to_partition.get(clean.iloc[left]["original_index"], "clean-only"),
            "right_partition": index_to_partition.get(clean.iloc[right]["original_index"], "clean-only"),
            "jaccard": jaccard,
        }
        for left, right, jaccard in qualifying_pairs
        if clean.iloc[left]["source"] != clean.iloc[right]["source"]
    ]
    result = {
        "status": "COMPLETED",
        "method": "word-shingle MinHash/LSH candidate audit followed by exact Jaccard verification",
        "threshold": AUDIT_THRESHOLD,
        "shingle_size_words": SHINGLE_SIZE,
        "minhash_permutations": MINHASH_PERMUTATIONS,
        "lsh_bands": LSH_BANDS,
        "lsh_rows_per_band": LSH_ROWS,
        "clean_rows": len(clean),
        "candidate_pairs": len(candidates),
        "qualifying_pairs": len(qualifying_pairs),
        "cross_fixed_partition_pairs": len(cross_partition_pairs),
        "cross_source_pairs": len(cross_source_pairs),
        "runtime_seconds": time.perf_counter() - started,
        "protocol_note": "The repository had no prior MinHash/LSH implementation or shingle parameters; these fixed parameters are a documented implementation of the requested audit, not a post-hoc threshold selection.",
    }
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "near_duplicate_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(cross_partition_pairs).to_csv(TABLES / "near_duplicate_cross_partition_pairs.csv", index=False)
    pd.DataFrame(cross_source_pairs).to_csv(TABLES / "near_duplicate_cross_source_pairs.csv", index=False)
    return result


def _sample(frame: pd.DataFrame, rows: int) -> pd.DataFrame:
    sampled, _ = train_test_split(frame, train_size=rows, random_state=SEED, stratify=frame["category"])
    return sampled.reset_index(drop=True)


def create_manifests() -> dict:
    clean = pd.read_parquet(DATA / "np20ng_clean.parquet")
    fixed = load_partitions()
    ordinary_train, ordinary_remainder = train_test_split(
        clean, test_size=0.4, random_state=SEED, stratify=clean["category"]
    )
    ordinary_validation, ordinary_test = train_test_split(
        ordinary_remainder, test_size=0.5, random_state=SEED + 1, stratify=ordinary_remainder["category"]
    )
    conditions = {
        "A": (_sample(ordinary_train, TRAIN_ROWS), _sample(ordinary_validation, EVALUATION_ROWS), _sample(ordinary_test, EVALUATION_ROWS)),
        "B": (_sample(fixed["train"], TRAIN_ROWS), _sample(fixed["validation"], EVALUATION_ROWS), _sample(fixed["test"], EVALUATION_ROWS)),
    }
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    result = {}
    for condition, frames in conditions.items():
        paths = []
        for name, frame in zip(("train", "validation", "test"), frames):
            path = MANIFESTS / f"{condition}_{name}.parquet"
            frame.to_parquet(path, index=False)
            paths.append(str(path))
        result[condition] = {
            "paths": paths,
            "rows": [len(frame) for frame in frames],
            "categories": [int(frame["category"].nunique()) for frame in frames],
            "sources": [sorted(frame["source"].unique().tolist()) for frame in frames],
        }
    metadata = {
        "status": "COMPLETED",
        "seed": SEED,
        "train_rows": TRAIN_ROWS,
        "evaluation_rows": EVALUATION_ROWS,
        "conditions": result,
        "fixed_partitions_preserved": True,
        "selection_note": "Manifests are created once and are the only rows used for the matched A/B comparison.",
    }
    (TABLES / "matched_50k_manifest_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _metrics(frame: pd.DataFrame, predicted: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(frame["category"], predicted)),
        "macro_f1": float(f1_score(frame["category"], predicted, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(frame["category"], predicted, labels=LABELS, average="weighted", zero_division=0)),
    }


def run_classical(
    condition: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    max_features: int = 300000,
) -> list[dict]:
    rows = []
    for classifier_name, classifier in (
        ("LogisticRegression", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED)),
        ("LinearSVC", LinearSVC(C=1.0)),
    ):
        vectorizer = make_vectorizer("word_bigram", max_features=max_features)
        x_train = vectorizer.fit_transform(text(train))
        x_validation = vectorizer.transform(text(validation))
        x_test = vectorizer.transform(text(test))
        classifier.fit(x_train, train["category"])
        validation_predicted = classifier.predict(x_validation)
        test_predicted = classifier.predict(x_test)
        rows.append({
            "condition": condition,
            "model": f"TF-IDF word bigram + {classifier_name}",
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "validation_metrics": _metrics(validation, validation_predicted),
            "test_metrics": _metrics(test, test_predicted),
            "n_features": int(x_train.shape[1]),
            "max_features": max_features,
        })
    return rows


def run_minilm(condition: str, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    embeddings = {}
    cache = {}
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        embeddings[name], cache[name] = encode_full_cached(frame, f"matched_{condition}_{name}", model, batch_size=128)
    rows = []
    for classifier_name, classifier in (
        ("LogisticRegression", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED)),
        ("LinearSVC", LinearSVC(C=1.0)),
    ):
        classifier.fit(embeddings["train"], train["category"])
        validation_predicted = classifier.predict(embeddings["validation"])
        test_predicted = classifier.predict(embeddings["test"])
        rows.append({
            "condition": condition,
            "model": f"{EMBEDDING_MODEL} + {classifier_name}",
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "validation_metrics": _metrics(validation, validation_predicted),
            "test_metrics": _metrics(test, test_predicted),
            "embedding_dimension": int(embeddings["train"].shape[1]),
            "cache": cache,
        })
    return rows


def run_matched_baselines() -> dict:
    metadata = json.loads((TABLES / "matched_50k_manifest_metadata.json").read_text(encoding="utf-8"))
    all_rows = []
    for condition in ("A", "B"):
        frames = {
            name: pd.read_parquet(MANIFESTS / f"{condition}_{name}.parquet")
            for name in ("train", "validation", "test")
        }
        all_rows.extend(run_classical(condition, frames["train"], frames["validation"], frames["test"]))
        all_rows.extend(run_minilm(condition, frames["train"], frames["validation"], frames["test"]))
    output = TABLES / "matched_50k_baseline_results.json"
    output.write_text(json.dumps({"metadata": metadata, "results": all_rows}, indent=2), encoding="utf-8")
    return {"metadata": metadata, "results": all_rows}


def save_matched_ab_predictions() -> dict:
    """Regenerate per-document predictions for the strongest A/B baselines."""
    import torch

    output = {}
    for condition in ("A", "B"):
        frames = {
            name: pd.read_parquet(MANIFESTS / f"{condition}_{name}.parquet")
            for name in ("train", "validation", "test")
        }
        vectorizer = make_vectorizer("word_bigram", max_features=300000)
        x_train = vectorizer.fit_transform(text(frames["train"]))
        classifier = LinearSVC(C=1.0).fit(x_train, frames["train"]["category"])
        x_test = vectorizer.transform(text(frames["test"]))
        tfidf_predicted = classifier.predict(x_test)
        tfidf_path = TABLES / f"matched50k_tfidf_{condition}_test_predictions.csv"
        pd.DataFrame({
            "original_index": frames["test"]["original_index"],
            "source": frames["test"]["source"],
            "category": frames["test"]["category"],
            "predicted_category": tfidf_predicted,
        }).to_csv(tfidf_path, index=False)

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            EMBEDDING_MODEL,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        embeddings = {}
        caches = {}
        for name, frame in frames.items():
            embeddings[name], caches[name] = encode_full_cached(
                frame, f"matched_{condition}_{name}", model, batch_size=128
            )
        minilm = LinearSVC(C=1.0).fit(embeddings["train"], frames["train"]["category"])
        minilm_predicted = minilm.predict(embeddings["test"])
        minilm_path = TABLES / f"matched50k_minilm_{condition}_test_predictions.csv"
        pd.DataFrame({
            "original_index": frames["test"]["original_index"],
            "source": frames["test"]["source"],
            "category": frames["test"]["category"],
            "predicted_category": minilm_predicted,
        }).to_csv(minilm_path, index=False)
        output[condition] = {
            "tfidf_prediction_path": str(tfidf_path),
            "minilm_prediction_path": str(minilm_path),
            "embedding_cache": caches,
            "test_rows": len(frames["test"]),
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result_path = TABLES / "matched_50k_ab_prediction_metadata.json"
    result_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def create_c_manifests() -> dict:
    clean = pd.read_parquet(DATA / "np20ng_clean.parquet")
    metadata = {
        "status": "COMPLETED",
        "condition": "C",
        "seed": SEED,
        "train_rows": TRAIN_ROWS,
        "validation_rows": EVALUATION_ROWS,
        "test_policy": "all eligible held-out publisher rows",
        "eligibility_rule": "at least 1,000 documents and at least 5 observed categories",
        "sources": {},
    }
    C_MANIFESTS.mkdir(parents=True, exist_ok=True)
    for source in eligible_sources(clean):
        test = clean.loc[clean["source"].eq(source)].copy()
        source_excluded = clean.loc[clean["source"].ne(source)].copy()
        train, validation = train_test_split(
            source_excluded,
            test_size=0.2,
            random_state=SEED,
            stratify=source_excluded["category"],
        )
        train = _sample(train, TRAIN_ROWS)
        validation = _sample(validation, min(EVALUATION_ROWS, len(validation)))
        frames = {"train": train, "validation": validation, "test": test.reset_index(drop=True)}
        paths = {}
        for name, frame in frames.items():
            path = C_MANIFESTS / f"{source}_{name}.parquet"
            frame.to_parquet(path, index=False)
            paths[name] = str(path)
        metadata["sources"][source] = {
            "paths": paths,
            "rows": {name: len(frame) for name, frame in frames.items()},
            "categories": {name: int(frame["category"].nunique()) for name, frame in frames.items()},
            "held_out_source": source,
        }
    (TABLES / "matched_50k_c_manifest_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def run_matched_c_baselines() -> dict:
    metadata = json.loads((TABLES / "matched_50k_c_manifest_metadata.json").read_text(encoding="utf-8"))
    rows = []
    for source in metadata["sources"]:
        frames = {
            name: pd.read_parquet(C_MANIFESTS / f"{source}_{name}.parquet")
            for name in ("train", "validation", "test")
        }
        rows.extend(
            run_classical(
                f"C_{source}",
                frames["train"],
                frames["validation"],
                frames["test"],
                max_features=10000,
            )
        )
        rows.extend(run_minilm(f"C_{source}", frames["train"], frames["validation"], frames["test"]))
    output = TABLES / "matched_50k_c_baseline_results.json"
    output.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "results": rows,
                "status": "BOUNDED_CLASSICAL_CONFIGURATION",
                "protocol_note": "C classical models use the existing 10,000-feature bounded configuration because the 300,000-feature matched path did not complete within the measured runtime window.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"metadata": metadata, "results": rows}


def run_matched_c_xlmr(source: str) -> dict:
    from src.day8_neural_experiments import run_xlm_condition

    frames = {
        name: pd.read_parquet(C_MANIFESTS / f"{source}_{name}.parquet")
        for name in ("train", "validation", "test")
    }
    return run_xlm_condition(
        "C",
        frames["train"],
        frames["validation"],
        frames["test"],
        {
            "condition": "C",
            "held_out_source": source,
            "description": "matched 50k leave-one-source-out manifest; held-out publisher excluded from train and validation",
            "matched_budget": True,
            "near_duplicate_audit_status": "BLOCKED",
        },
        f"matched50k_xlmr_C_{source}",
        train_limit=None,
        evaluation_limit=None,
        epochs=1,
    )


def run_matched_xlmr(condition: str) -> dict:
    from src.day8_neural_experiments import run_xlm_condition

    frames = {
        name: pd.read_parquet(MANIFESTS / f"{condition}_{name}.parquet")
        for name in ("train", "validation", "test")
    }
    return run_xlm_condition(
        condition,
        frames["train"],
        frames["validation"],
        frames["test"],
        {
            "condition": condition,
            "description": (
                "ordinary random matched 50k manifest; duplicate leakage permitted"
                if condition == "A"
                else "fixed duplicate-aware matched 50k manifest"
            ),
            "matched_budget": True,
            "near_duplicate_audit_status": "BLOCKED",
        },
        f"matched50k_xlmr_{condition}",
        train_limit=None,
        evaluation_limit=None,
        epochs=1,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--manifests", action="store_true")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--ab-predictions", action="store_true")
    parser.add_argument("--c-manifests", action="store_true")
    parser.add_argument("--c-baselines", action="store_true")
    parser.add_argument("--c-xlmr", choices=tuple(eligible_sources(pd.read_parquet(DATA / "np20ng_clean.parquet", columns=["source", "category"]))))
    parser.add_argument("--xlmr", choices=("A", "B"))
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(run_near_duplicate_audit(), indent=2))
    if args.manifests:
        print(json.dumps(create_manifests(), indent=2))
    if args.baselines:
        print(json.dumps(run_matched_baselines(), indent=2))
    if args.ab_predictions:
        print(json.dumps(save_matched_ab_predictions(), indent=2))
    if args.c_manifests:
        print(json.dumps(create_c_manifests(), indent=2))
    if args.c_baselines:
        print(json.dumps(run_matched_c_baselines(), indent=2))
    if args.xlmr:
        print(json.dumps(run_matched_xlmr(args.xlmr), indent=2))
    if args.c_xlmr:
        print(json.dumps(run_matched_c_xlmr(args.c_xlmr), indent=2))
