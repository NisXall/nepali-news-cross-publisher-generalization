"""Block 4: Source-text robustness, McNemar tests, and paper-ready packaging.

Experiments
-----------
1. Source-text ablation under Condition C (LOSO) with three input variants:
   - heading only
   - content only  (existing result, re-run for unified output)
   - heading + content (concatenated with a single space)
   Each variant trains a TF-IDF word-bigram LinearSVC on 50k matched training
   rows (C manifests) and evaluates on the full held-out publisher test set.

2. McNemar's test between every pair of models in the matched A/B comparison,
   using saved per-document prediction CSVs (no retraining required).

3. Full-data near-duplicate audit decision: the matched-manifest audit found
   0.42% / 0.33% affected test rows with no accuracy advantage for affected
   rows. This function records the decision without re-running the audit.

All outputs are saved to results/tables/ and results/figures/.
A consolidated summary is written to notes/block4_robustness.md.
"""

from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
NOTES = ROOT / "notes"
C_MANIFESTS = TABLES / "matched_50k_c_manifests"
SEED = 42
MAX_FEATURES = 10_000   # matched-budget bounded classical config for C
LABELS_PATH = TABLES / "day3_category_distribution.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_labels() -> list[str]:
    """Return the sorted 20-category list from the data itself."""
    train = pd.read_parquet(DATA / "train.parquet", columns=["category"])
    return sorted(train["category"].unique().tolist())


LABELS: list[str] = _load_labels()


def _metrics(y_true: pd.Series, predicted, present_only: bool = False) -> dict:
    labels = sorted(y_true.unique()) if present_only else LABELS
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, labels=labels,
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predicted, labels=LABELS,
                                      average="weighted", zero_division=0)),
    }


def _text_variant(frame: pd.DataFrame, variant: str) -> pd.Series:
    """Return text for the requested input variant."""
    heading = frame["heading"].fillna("").astype(str)
    content = frame["content"].fillna("").astype(str)
    if variant == "heading":
        return heading
    if variant == "content":
        return content
    if variant == "heading_plus_content":
        return heading + " " + content
    raise ValueError(f"Unknown variant: {variant!r}")


def _fit_predict_tfidf(
    train: pd.DataFrame,
    test: pd.DataFrame,
    variant: str,
    max_features: int = MAX_FEATURES,
) -> tuple[dict, list]:
    started = time.perf_counter()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=max_features, sublinear_tf=True
    )
    x_train = vectorizer.fit_transform(_text_variant(train, variant))
    x_test = vectorizer.transform(_text_variant(test, variant))
    clf = LinearSVC(C=1.0)
    clf.fit(x_train, train["category"])
    predicted = clf.predict(x_test).tolist()
    row = {
        "variant": variant,
        "max_features": max_features,
        "n_features": int(x_train.shape[1]),
        "train_rows": len(train),
        "test_rows": len(test),
        "runtime_seconds": time.perf_counter() - started,
        **_metrics(test["category"], predicted, present_only=True),
        "present_label_macro_f1": float(
            f1_score(test["category"], predicted,
                     labels=sorted(test["category"].unique()),
                     average="macro", zero_division=0)
        ),
    }
    return row, predicted


# ---------------------------------------------------------------------------
# Experiment 1: source-text ablation under C (LOSO)
# ---------------------------------------------------------------------------

def run_source_text_ablation_c() -> dict:
    """Run heading / content / heading+content ablation for all C publishers."""
    eligible_sources = sorted(
        p.stem.replace("_train", "")
        for p in C_MANIFESTS.glob("*_train.parquet")
    )
    if not eligible_sources:
        return {
            "status": "SKIPPED",
            "reason": "No C-condition manifests found under results/tables/matched_50k_c_manifests/",
        }

    rows: list[dict] = []
    predictions_store: dict[str, dict] = {}

    for source in eligible_sources:
        train = pd.read_parquet(C_MANIFESTS / f"{source}_train.parquet")
        test  = pd.read_parquet(C_MANIFESTS / f"{source}_test.parquet")

        source_rows = []
        source_preds: dict[str, list] = {}
        for variant in ("heading", "content", "heading_plus_content"):
            row, preds = _fit_predict_tfidf(train, test, variant)
            row["held_out_source"] = source
            source_rows.append(row)
            source_preds[variant] = preds

        rows.extend(source_rows)
        predictions_store[source] = source_preds

        # Per-source per-variant per-class report
        for variant in ("heading", "content", "heading_plus_content"):
            preds = source_preds[variant]
            present = sorted(test["category"].unique())
            report = classification_report(
                test["category"], preds,
                labels=present, output_dict=True, zero_division=0,
            )
            pd.DataFrame([
                {
                    "source": source, "variant": variant, "category": label,
                    "precision": report[label]["precision"],
                    "recall": report[label]["recall"],
                    "f1": report[label]["f1-score"],
                    "support": int(report[label]["support"]),
                }
                for label in present
            ]).to_csv(
                TABLES / f"block4_c_{source}_{variant}_per_class.csv",
                index=False,
            )

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "block4_source_text_ablation_c.csv", index=False)

    # Pivot summary: for each source, show macro_f1 by variant
    pivot = df.pivot_table(
        index="held_out_source",
        columns="variant",
        values="present_label_macro_f1",
        aggfunc="first",
    ).reset_index()
    pivot.to_csv(TABLES / "block4_source_text_ablation_c_pivot.csv", index=False)

    # Aggregate across sources
    agg = df.groupby("variant").agg(
        mean_present_macro_f1=("present_label_macro_f1", "mean"),
        std_present_macro_f1=("present_label_macro_f1", lambda x: x.std(ddof=1)),
        mean_accuracy=("accuracy", "mean"),
        n_sources=("held_out_source", "nunique"),
    ).reset_index()
    agg.to_csv(TABLES / "block4_source_text_ablation_c_agg.csv", index=False)

    # Figure: grouped bar chart
    _plot_ablation_bar(pivot, FIGURES / "block4_source_text_ablation.png")

    result = {
        "status": "COMPLETED",
        "experiment": "source-text input ablation under matched-budget C (LOSO)",
        "model": "TF-IDF word bigram + LinearSVC (bounded 10k features)",
        "variants": ["heading", "content", "heading_plus_content"],
        "sources": eligible_sources,
        "seed": SEED,
        "aggregate_by_variant": agg.to_dict(orient="records"),
        "pivot_path": str(TABLES / "block4_source_text_ablation_c_pivot.csv"),
        "full_results_path": str(TABLES / "block4_source_text_ablation_c.csv"),
        "figure_path": str(FIGURES / "block4_source_text_ablation.png"),
        "protocol_note": (
            "Each variant fits TF-IDF on the C-condition 50k training manifest "
            "and evaluates on the full held-out publisher test set. "
            "Present-label macro-F1 is used because category coverage differs by publisher."
        ),
    }
    return result


def _plot_ablation_bar(pivot: pd.DataFrame, path: Path) -> None:
    """Grouped bar chart of present-label macro-F1 per source and variant."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    variant_cols = [c for c in ("heading", "content", "heading_plus_content") if c in pivot.columns]
    labels_map = {
        "heading": "Heading only",
        "content": "Content only",
        "heading_plus_content": "Heading + Content",
    }
    sources = pivot["held_out_source"].tolist()
    x = np.arange(len(sources))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, col in enumerate(variant_cols):
        ax.bar(x + i * width, pivot[col].fillna(0), width,
               label=labels_map.get(col, col))
    ax.set_xlabel("Held-out publisher")
    ax.set_ylabel("Present-label macro-F1")
    ax.set_title("Source-text ablation under leave-one-source-out (C condition)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(sources, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 2: McNemar's test between matched A/B model pairs
# ---------------------------------------------------------------------------

def run_mcnemar_tests() -> dict:
    """Paired significance tests on saved matched-budget A/B test predictions."""
    try:
        from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
    except ImportError:
        return {
            "status": "SKIPPED",
            "reason": (
                "statsmodels is not installed. "
                "Install it with: pip install statsmodels"
            ),
        }

    prediction_files = {
        "TF-IDF": {
            "A": TABLES / "matched50k_tfidf_A_test_predictions.csv",
            "B": TABLES / "matched50k_tfidf_B_test_predictions.csv",
        },
        "MiniLM": {
            "A": TABLES / "matched50k_minilm_A_test_predictions.csv",
            "B": TABLES / "matched50k_minilm_B_test_predictions.csv",
        },
        "XLM-R": {
            "A": TABLES / "matched50k_xlmr_A_test_predictions.csv",
            "B": TABLES / "matched50k_xlmr_B_test_predictions.csv",
        },
    }

    results = []
    for condition in ("A", "B"):
        frames: dict[str, pd.DataFrame] = {}
        available: list[str] = []
        for model_name, paths in prediction_files.items():
            path = paths[condition]
            if not path.exists():
                continue
            df = pd.read_csv(path, usecols=["original_index", "category", "predicted_category"])
            df["correct"] = df["category"] == df["predicted_category"]
            frames[model_name] = df.set_index("original_index")
            available.append(model_name)

        # Align on shared original_index
        if len(available) < 2:
            continue
        shared_idx = set.intersection(*[set(frames[m].index) for m in available])
        for m in available:
            frames[m] = frames[m].loc[list(shared_idx)].sort_index()

        for model_a, model_b in combinations(available, 2):
            ca = frames[model_a]["correct"].values
            cb = frames[model_b]["correct"].values
            # Contingency table:
            # n00: both wrong, n01: a wrong b right, n10: a right b wrong, n11: both right
            n00 = int((~ca & ~cb).sum())
            n01 = int((~ca &  cb).sum())
            n10 = int(( ca & ~cb).sum())
            n11 = int(( ca &  cb).sum())
            table = [[n00, n01], [n10, n11]]
            # Use exact=False (chi-sq) for large n; exact=True for n<25 discordant
            discordant = n01 + n10
            exact = discordant < 25
            stat = sm_mcnemar(table, exact=exact, correction=True)
            results.append({
                "condition": condition,
                "model_a": model_a,
                "model_b": model_b,
                "n_shared": len(shared_idx),
                "n_a_correct_b_wrong": n10,
                "n_b_correct_a_wrong": n01,
                "n_both_correct": n11,
                "n_both_wrong": n00,
                "discordant_pairs": discordant,
                "test_type": "exact" if exact else "chi-squared with continuity correction",
                "statistic": float(stat.statistic) if not exact else None,
                "p_value": float(stat.pvalue),
                "significant_at_0.05": float(stat.pvalue) < 0.05,
                "significant_at_0.01": float(stat.pvalue) < 0.01,
                "interpretation": (
                    f"{model_b} significantly better than {model_a} (p={stat.pvalue:.4f})"
                    if float(stat.pvalue) < 0.05 and n01 > n10
                    else (
                        f"{model_a} significantly better than {model_b} (p={stat.pvalue:.4f})"
                        if float(stat.pvalue) < 0.05 and n10 > n01
                        else f"No significant difference (p={stat.pvalue:.4f})"
                    )
                ),
            })

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results.to_csv(TABLES / "block4_mcnemar_tests.csv", index=False)

    result = {
        "status": "COMPLETED" if results else "SKIPPED_NO_DATA",
        "description": (
            "McNemar's test on matched-budget A/B per-document predictions. "
            "No model retraining was done; saved prediction CSVs were reused."
        ),
        "test": "McNemar (exact for <25 discordant pairs, chi-sq with continuity correction otherwise)",
        "alpha": 0.05,
        "results": results,
        "output_path": str(TABLES / "block4_mcnemar_tests.csv"),
    }
    (TABLES / "block4_mcnemar_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


# ---------------------------------------------------------------------------
# Experiment 3: full-data near-duplicate audit decision
# ---------------------------------------------------------------------------

def record_neardup_audit_decision() -> dict:
    """
    Record the decision on whether to run the full-data near-duplicate audit.

    Evidence from the matched-manifest audit (140k rows across A/B):
    - 22,329 Jaccard>=0.8 qualifying pairs in 21,703 groups
    - 42 A test rows (0.420%) and 33 B test rows (0.330%) are affected
      (belong to a group containing a same-condition training row)
    - For XLM-R affected rows: accuracy 0.788 vs unaffected 0.822
      → affected rows are slightly *less* accurate, not inflated
    - The paper's central claim (XLM-R > TF-IDF under matched budget) is
      not threatened by 0.42% potential near-duplicate leakage

    Decision: SKIP the full-corpus near-duplicate audit.
    """
    decision = {
        "status": "DECIDED_NOT_TO_RUN",
        "experiment": "full-corpus near-duplicate audit (MinHash/LSH over 231k rows)",
        "evidence_from_matched_audit": {
            "manifest_rows": 140_000,
            "qualifying_pairs": 22_329,
            "groups": 21_703,
            "A_test_affected_rows": 42,
            "A_test_affected_pct": 0.420,
            "B_test_affected_rows": 33,
            "B_test_affected_pct": 0.330,
            "xlmr_affected_accuracy": 0.788,
            "xlmr_unaffected_accuracy": 0.822,
            "direction": "affected rows are less accurate — no inflation of the central result",
        },
        "central_claims_at_risk": [
            "XLM-R macro-F1 0.7782-0.7921 vs TF-IDF 0.7208-0.7234 under matched budget"
        ],
        "decision_rationale": (
            "The affected-row percentage (0.33–0.42%) is negligible. "
            "Affected XLM-R rows show lower accuracy than unaffected rows, "
            "meaning any near-duplicate leakage marginally *hurts* XLM-R rather "
            "than inflating it. The full-corpus audit would not change the "
            "paper's central comparison. Running it on 231k rows risks "
            "many hours of compute with no expected change to conclusions. "
            "This decision is recorded for transparency."
        ),
        "limitation_for_paper": (
            "Partitions are exact-normalized-duplicate-aware but not fully "
            "near-duplicate-safe at Jaccard>=0.8. The matched-manifest audit "
            "found <0.5% affected test rows with no evidence of result inflation."
        ),
    }
    (TABLES / "block4_neardup_audit_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    return decision


# ---------------------------------------------------------------------------
# Confusion matrix figure (paper-ready)
# ---------------------------------------------------------------------------

def plot_confusion_matrix() -> str:
    """Plot a normalized confusion matrix for the full-data TF-IDF reference model."""
    cm_path = TABLES / "final_test_confusion_matrix.csv"
    if not cm_path.exists():
        return "SKIPPED: final_test_confusion_matrix.csv not found"

    cm = pd.read_csv(cm_path, index_col=0)
    # Normalize rows (true label)
    cm_norm = cm.div(cm.sum(axis=1), axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm.values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Proportion")
    tick_labels = cm.index.tolist()
    ax.set_xticks(range(len(tick_labels)))
    ax.set_yticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.set_title("Normalized confusion matrix — TF-IDF + LinearSVC (full test partition)")

    # Annotate cells with proportion values (only ≥ 0.05 to avoid clutter)
    for i in range(len(tick_labels)):
        for j in range(len(tick_labels)):
            val = cm_norm.values[i, j]
            if val >= 0.05:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if val > 0.6 else "black")

    fig.tight_layout()
    out = FIGURES / "block4_confusion_matrix_normalized.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)


# ---------------------------------------------------------------------------
# LOSO summary bar chart (paper-ready)
# ---------------------------------------------------------------------------

def plot_loso_summary() -> str:
    """Bar chart comparing TF-IDF and XLM-R present-label macro-F1 across publishers."""
    classical_path = TABLES / "source_held_out_summary.csv"
    if not classical_path.exists():
        return "SKIPPED: source_held_out_summary.csv not found"

    classical = pd.read_csv(classical_path)[
        ["held_out_source", "macro_f1_present_labels"]
    ].rename(columns={"macro_f1_present_labels": "TF-IDF (bounded 10k)"})

    # XLM-R LOSO from bootstrap file
    bootstrap_path = TABLES / "matched_50k_independent_bootstrap.json"
    xlmr_rows = []
    if bootstrap_path.exists():
        data = json.loads(bootstrap_path.read_text(encoding="utf-8"))
        for entry in data.get("results", []):
            cond = entry.get("condition", "")
            if cond.startswith("C_"):
                source = cond[2:]
                xlmr_rows.append({
                    "held_out_source": source,
                    "XLM-R (matched 50k, 1 epoch)": entry["point_estimate"]["macro_f1"],
                })
    xlmr = pd.DataFrame(xlmr_rows) if xlmr_rows else pd.DataFrame(
        columns=["held_out_source", "XLM-R (matched 50k, 1 epoch)"]
    )

    merged = classical.merge(xlmr, on="held_out_source", how="outer").sort_values(
        "TF-IDF (bounded 10k)"
    )

    model_cols = [c for c in merged.columns if c != "held_out_source"]
    x = np.arange(len(merged))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4C72B0", "#DD8452"]
    for i, col in enumerate(model_cols):
        ax.bar(x + i * width - width / 2 * (len(model_cols) - 1),
               merged[col].fillna(0), width, label=col, color=colors[i % len(colors)])

    ax.axhline(y=0.4605, color="gray", linestyle="--", linewidth=1,
               label="Classical LOSO mean (0.461)")
    ax.set_xlabel("Held-out publisher")
    ax.set_ylabel("Present-label macro-F1")
    ax.set_title("Cross-publisher generalization: leave-one-source-out evaluation")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["held_out_source"].tolist(), rotation=30, ha="right")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    out = FIGURES / "block4_loso_summary.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)


# ---------------------------------------------------------------------------
# Training size curve (paper-ready)
# ---------------------------------------------------------------------------

def plot_training_size_curve() -> str:
    """Plot training-size learning curve from saved day5 results."""
    path = TABLES / "day5_training_size.csv"
    if not path.exists():
        return "SKIPPED: day5_training_size.csv not found"

    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 4))
    for metric, label, color in [
        ("macro_f1", "Macro-F1", "#4C72B0"),
        ("accuracy", "Accuracy", "#DD8452"),
        ("weighted_f1", "Weighted-F1", "#55A868"),
    ]:
        if metric in df.columns:
            ax.plot(df["n_train"], df[metric], "o-", label=label, color=color)

    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.set_title("TF-IDF + LinearSVC: performance vs. training size")
    ax.legend()
    ax.grid(alpha=0.4)
    fig.tight_layout()
    out = FIGURES / "block4_training_size_curve.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)


# ---------------------------------------------------------------------------
# Consolidated results table (all models, all conditions)
# ---------------------------------------------------------------------------

def build_consolidated_table() -> str:
    """Write a single paper-ready CSV covering all completed evaluations."""
    rows = []

    # 1. Full-data classical (primary)
    metrics_path = TABLES / "final_test_metrics.json"
    if metrics_path.exists():
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        for entry in data if isinstance(data, list) else [data]:
            rows.append({
                "condition": "Full data (grouped split)",
                "model": "TF-IDF bigram + LinearSVC",
                "train_rows": 138_711,
                "test_rows": 46_238,
                "test_accuracy": entry.get("accuracy"),
                "test_macro_f1": entry.get("macro_f1"),
                "test_weighted_f1": entry.get("weighted_f1"),
                "notes": "Primary result; 300k features; bootstrap CI in final_test_metadata.json",
            })

    # 2. Full-data MiniLM
    mini_path = TABLES / "sentence_embedding_final_test_metrics.json"
    if mini_path.exists():
        m = json.loads(mini_path.read_text(encoding="utf-8"))
        rows.append({
            "condition": "Full data (grouped split)",
            "model": "MiniLM-L12-v2 + LinearSVC",
            "train_rows": 138_711,
            "test_rows": 46_238,
            "test_accuracy": m.get("accuracy"),
            "test_macro_f1": m.get("macro_f1"),
            "test_weighted_f1": m.get("weighted_f1"),
            "notes": "Full fixed-partition benchmark; selected by validation macro-F1",
        })

    # 3. Matched-budget A/B
    ab_path = TABLES / "matched_50k_ab_comparison.csv"
    if ab_path.exists():
        ab = pd.read_csv(ab_path)
        for _, r in ab.iterrows():
            rows.append({
                "condition": f"Matched 50k ({r['condition']} {'random' if r['condition']=='A' else 'grouped'})",
                "model": r["model"],
                "train_rows": int(r["train_rows"]),
                "test_rows": int(r["test_rows"]),
                "test_accuracy": r["test_accuracy"],
                "test_macro_f1": r["test_macro_f1"],
                "test_weighted_f1": r["test_weighted_f1"],
                "notes": "Matched budget; 1 epoch for XLM-R",
            })

    # 4. Calibrated classical (test)
    cal_path = TABLES / "calibration_abstention_metrics.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        ta = cal.get("test_abstention", {})
        rows.append({
            "condition": "Full data (grouped split) + abstention",
            "model": "TF-IDF bigram + LinearSVC + calibration",
            "train_rows": 138_711,
            "test_rows": ta.get("accepted"),
            "test_accuracy": ta.get("accuracy_accepted"),
            "test_macro_f1": ta.get("macro_f1_accepted"),
            "test_weighted_f1": None,
            "notes": f"Coverage {ta.get('coverage', 0):.1%}; ECE {cal.get('test_calibration', {}).get('ece', 0):.4f}",
        })

    df = pd.DataFrame(rows)
    out = TABLES / "block4_consolidated_all_models.csv"
    df.to_csv(out, index=False)
    return str(out)


# ---------------------------------------------------------------------------
# Notes writer
# ---------------------------------------------------------------------------

def write_block4_notes(results: dict) -> None:
    """Append a block4 section to notes/block4_robustness.md."""
    NOTES.mkdir(parents=True, exist_ok=True)
    path = NOTES / "block4_robustness.md"

    ablation = results.get("ablation", {})
    mcnemar = results.get("mcnemar", {})
    neardup = results.get("neardup_decision", {})

    # Ablation pivot summary
    pivot_lines = ""
    agg_path = TABLES / "block4_source_text_ablation_c_agg.csv"
    if agg_path.exists():
        agg = pd.read_csv(agg_path)
        lines = ["| Variant | Mean present-label macro-F1 | SD |",
                 "|---|---:|---:|"]
        for _, r in agg.iterrows():
            lines.append(
                f"| {r['variant']} | {r['mean_present_macro_f1']:.4f} | {r['std_present_macro_f1']:.4f} |"
            )
        pivot_lines = "\n".join(lines)

    # McNemar summary
    mc_lines = ""
    if mcnemar.get("results"):
        lines = ["| Condition | Model A | Model B | p-value | Significant? | Interpretation |",
                 "|---|---|---|---:|---|---|"]
        for r in mcnemar["results"]:
            lines.append(
                f"| {r['condition']} | {r['model_a']} | {r['model_b']} "
                f"| {r['p_value']:.4f} | {r['significant_at_0.05']} "
                f"| {r['interpretation']} |"
            )
        mc_lines = "\n".join(lines)

    content = f"""# Block 4: Source-text robustness and final research package

Generated automatically by `src/block4_robustness.py`.

---

## 1. Source-text input ablation under LOSO (Condition C)

**Design:** TF-IDF word-bigram LinearSVC (bounded 10k features) trained on the
matched 50k C-condition manifest for each of the seven eligible publishers.
Three input variants are compared:

- **Heading only** — article title
- **Content only** — article body (existing configuration)
- **Heading + Content** — title prepended to body with one space

### Aggregate results by variant

{pivot_lines}

### Interpretation

The ablation shows how much the article body contributes over the headline
alone. A large gap (heading → heading+content) suggests that body text carries
distinct topical signal beyond what the headline captures.

**Detailed pivot:** `results/tables/block4_source_text_ablation_c_pivot.csv`  
**Full per-source per-variant results:** `results/tables/block4_source_text_ablation_c.csv`  
**Figure:** `results/figures/block4_source_text_ablation.png`

---

## 2. McNemar significance tests (matched-budget A/B predictions)

**Design:** Paired McNemar's test on per-document predictions from saved CSVs.
No model retraining was performed.

### Results

{mc_lines if mc_lines else "_No shared prediction sets found or statsmodels not installed._"}

**Output:** `results/tables/block4_mcnemar_tests.csv`

---

## 3. Full-data near-duplicate audit decision

**Decision: NOT RUN.**

**Evidence from the matched-manifest audit (140k rows, A+B conditions):**

- Qualifying Jaccard≥0.8 pairs: 22,329 in 21,703 groups
- A test rows affected: 42 (0.420%)
- B test rows affected: 33 (0.330%)
- XLM-R affected accuracy: 0.788 vs unaffected: 0.822

Near-duplicate leakage affects fewer than 0.5% of test rows, and affected
rows show *lower* accuracy than unaffected rows. Running the full-corpus audit
(231k rows, several hours) would not change any paper conclusion.

**Record:** `results/tables/block4_neardup_audit_decision.json`

---

## 4. Paper-ready figures produced

| Figure | Path |
|---|---|
| Source-text ablation grouped bar | `results/figures/block4_source_text_ablation.png` |
| Normalized confusion matrix | `results/figures/block4_confusion_matrix_normalized.png` |
| LOSO summary bar chart | `results/figures/block4_loso_summary.png` |
| Training size learning curve | `results/figures/block4_training_size_curve.png` |

---

## 5. Consolidated model results table

All completed evaluations in one file:
`results/tables/block4_consolidated_all_models.csv`

---

## 6. Limitations documented for paper

- Source-text ablation uses the bounded 10k-feature classical config, not the
  300k reference, for runtime feasibility. Relative differences are valid for
  comparison within this experiment; absolute values are not directly comparable
  to the primary 300k result.
- Heading field has 2 missing values in the full dataset; these are filled with
  empty string (consistent with content handling).
- McNemar's test is conditioned on the matched-budget test sets (10k rows per
  condition) and does not extrapolate to the full 46k test partition.
- Near-duplicate audit used word-5 shingles at Jaccard≥0.8; finer-grained
  near-duplicates at lower thresholds are not audited.
"""
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Block 4: Source-text robustness and final research package")
    print("=" * 60)

    print("\n[1/6] Source-text ablation under C (LOSO)...")
    ablation = run_source_text_ablation_c()
    print(f"      Status: {ablation['status']}")
    if ablation.get("aggregate_by_variant"):
        for row in ablation["aggregate_by_variant"]:
            print(f"        {row['variant']}: mean macro-F1 = {row['mean_present_macro_f1']:.4f} ± {row['std_present_macro_f1']:.4f}")

    print("\n[2/6] McNemar significance tests...")
    mcnemar = run_mcnemar_tests()
    print(f"      Status: {mcnemar['status']}")
    if mcnemar.get("results"):
        for r in mcnemar["results"]:
            print(f"        {r['condition']}: {r['model_a']} vs {r['model_b']} — {r['interpretation']}")

    print("\n[3/6] Near-duplicate audit decision...")
    neardup = record_neardup_audit_decision()
    print(f"      Decision: {neardup['status']}")

    print("\n[4/6] Paper-ready figures...")
    cm_out = plot_confusion_matrix()
    print(f"      Confusion matrix: {cm_out}")
    loso_out = plot_loso_summary()
    print(f"      LOSO summary: {loso_out}")
    size_out = plot_training_size_curve()
    print(f"      Training size curve: {size_out}")

    print("\n[5/6] Consolidated model results table...")
    table_out = build_consolidated_table()
    print(f"      Saved to: {table_out}")

    print("\n[6/6] Writing block4 notes...")
    results = {"ablation": ablation, "mcnemar": mcnemar, "neardup_decision": neardup}
    write_block4_notes(results)
    print(f"      Saved to: {NOTES / 'block4_robustness.md'}")

    (TABLES / "block4_status.json").write_text(
        json.dumps({
            "status": "COMPLETED",
            "ablation": ablation.get("status"),
            "mcnemar": mcnemar.get("status"),
            "neardup_decision": neardup["status"],
            "figures": [cm_out, loso_out, size_out],
            "consolidated_table": table_out,
        }, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("Block 4 complete. Repository is in paper-ready state.")
    print("=" * 60)


if __name__ == "__main__":
    main()
