# Classical benchmark: cross-source generalization

## Scope

This phase answers how the verified classical text classifier behaves when
publisher/source changes. It reuses the fixed duplicate-aware partitions and
does not regenerate them. The final test partition was evaluated only after
the reference configuration was frozen.

## Integrity audit

Command:

```powershell
.\.venv\Scripts\python.exe src\classical_benchmark.py --experiment final-test
```

The audit recorded:

- train/validation/test rows: 138,711 / 46,237 / 46,238;
- original indices disjoint across all partitions;
- zero normalized duplicate groups crossing partitions;
- all 20 categories present in every partition.

The machine-readable audit is
`results/tables/classical_benchmark_integrity.json`.

## Final reference test

The existing best validation configuration was frozen:

- word TF-IDF unigrams and bigrams;
- `min_df=2`;
- `max_features=300000`;
- `sublinear_tf=True`;
- `LinearSVC(C=1.0)`;
- no class weighting.

The vectorizer and classifier were fitted on the fixed training partition and
evaluated once on the untouched test partition. Results:

| Metric | Test result |
|---|---:|
| Accuracy | 0.7850 |
| Macro-F1 | 0.7504 |
| Weighted-F1 | 0.7816 |

The document-level bootstrap used 1,000 resamples with seed 42:

- accuracy 95% interval: [0.7813, 0.7885];
- macro-F1 95% interval: [0.7453, 0.7553].

These intervals describe resampling variation among the fixed test documents.
They do not estimate publisher-level or split-level uncertainty.

## Source-held-out evaluation

Eligibility was fixed before looking at scores: at least 1,000 documents and
at least 5 observed categories. Seven sources met the rule:

`Aarthiknews`, `Ekantipur`, `Gorkhapatra`, `Nagarik`, `Nepalkhabar`,
`Onlinekhabar`, and `Setopati`.

Each source was held out completely; the model trained on all other clean rows.
The exploratory word bigram configuration used a 10,000-feature cap to keep
the seven-model run feasible on the current machine. It is therefore not a
direct replacement for the 300,000-feature reference score.

Macro-F1 is reported over labels present in each held-out source. Missing
categories are listed explicitly and are not silently assigned ordinary
zero-support scores.

| Held-out source | Rows | Categories | Accuracy | Present-label macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| Aarthiknews | 25,965 | 7 | 0.2018 | 0.2851 | 0.2457 |
| Ekantipur | 29,875 | 10 | 0.4684 | 0.4203 | 0.5528 |
| Gorkhapatra | 5,059 | 8 | 0.6539 | 0.6864 | 0.7171 |
| Nagarik | 11,106 | 6 | 0.4814 | 0.5811 | 0.5293 |
| Nepalkhabar | 69,504 | 10 | 0.3613 | 0.3138 | 0.4375 |
| Onlinekhabar | 66,716 | 15 | 0.6758 | 0.5591 | 0.7417 |
| Setopati | 15,541 | 8 | 0.2666 | 0.3774 | 0.2982 |

Descriptive mean across sources:

- accuracy: 0.4442;
- present-label macro-F1: 0.4605;
- present-label macro-F1 standard deviation: 0.1506;
- weighted-F1: 0.5032.

The large spread is consistent with source/category association and differing
category coverage. It is not evidence that publisher writing style alone
causes the gap.

## Random versus duplicate-aware comparison

Using the same 10,000-feature exploratory configuration:

| Split | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Fixed duplicate-aware grouped | 0.7622 | 0.7259 | 0.7591 |
| Ordinary stratified random | 0.7589 | 0.7239 | 0.7552 |

The grouped split has zero duplicate crossings; the ordinary random audit has
156. These scores cannot isolate duplicate leakage causally because the split
also changes source and category composition.

## Source-related text diagnostic

A word bigram LinearSVC trained to predict publisher from article text achieved:

- validation accuracy: 0.7385;
- validation macro-F1: 0.6677;
- 10,000 word features.

This demonstrates that publisher identity is recoverable from article text and
can therefore act as a shortcut in random or grouped category evaluation.

A limited, transparent stripping ablation removed URLs, email-like tokens,
simple byline prefixes, and simple courtesy/source prefixes. Category results
changed only slightly:

- original text: macro-F1 0.7259;
- stripped text: macro-F1 0.7255.

This is not a complete boilerplate-removal study. The rules are intentionally
conservative and may miss publisher-specific templates.

## Error and conflict artifacts

- Per-source confusion matrices:
  `results/tables/source_held_out_<source>_confusion.csv`
- Per-source per-category metrics:
  `results/tables/source_held_out_per_class.csv`
- Held-out summary:
  `results/tables/source_held_out_summary.csv`
- Held-out metadata and aggregation:
  `results/tables/source_held_out_metadata.json`
- Text-source diagnostic:
  `results/tables/source_text_diagnostic.json`
- Source stripping ablation:
  `results/tables/source_text_ablation.csv`
- Random/grouped comparison:
  `results/tables/random_vs_grouped_comparison.csv`
- Conflicting duplicate review table:
  `results/tables/duplicate_conflict_review.csv`

The 14 conflicting duplicate groups remain unchanged. The review table is for
human adjudication; no labels were silently relabeled.

## Blocked or not run

- New bounded multi-model validation sweep: blocked by resource cost. The
  100k- and 30k-feature character runs exceeded roughly 10 minutes and
  1.2-1.5 GB resident memory without producing a table. A 10k-feature
  word-only exploratory run was also stopped after several minutes before
  completion because the planned four-configuration process remained too
  expensive.
- Character TF-IDF and word+character fusion: blocked for the same reason.
- New multiclass LogisticRegression rerun: not run after an attempt exceeded
  15 minutes; equivalent verified unigram and bigram LogisticRegression
  results already exist in `results/tables/baseline_comparison.csv`.
- Embedding and transformer metrics: still unavailable. Their dependencies and
  model artifacts are not installed or cached in the selected environment.

The blocked status is recorded in
`results/tables/classical_blocked_experiments.json`.

## Reproducible commands

```powershell
.\.venv\Scripts\python.exe src\classical_benchmark.py --experiment final-test
.\.venv\Scripts\python.exe src\classical_benchmark.py --experiment source-held-out
.\.venv\Scripts\python.exe src\classical_benchmark.py --experiment random-grouped
.\.venv\Scripts\python.exe src\classical_benchmark.py --experiment source-text
.\.venv\Scripts\python.exe src\classical_benchmark.py --experiment conflicts
.\.venv\Scripts\python.exe -m pytest tests\test_day3_splits.py tests\test_classical_benchmark.py -q
```

## Next neural phase

1. Keep the classical test artifacts immutable and do not tune against them.
2. Install and pin `transformers`, PyTorch, and tokenizer dependencies only in
   a separately recorded environment.
3. Run an XLM-R tokenizer-only length audit on the fixed train and validation
   partitions before downloading or training.
4. Run a local-cache model/tokenizer smoke test and a single forward/backward
   training-step test.
5. Use deterministic category IDs, fixed seeds, train-only fitting, validation
   macro-F1 checkpoint selection, early stopping, and reload verification.
6. Compare XLM-R against the exact fixed partitions and metrics used here.
7. Evaluate the selected neural checkpoint on the untouched test partition
   once, then report per-category and source-held-out results separately.
