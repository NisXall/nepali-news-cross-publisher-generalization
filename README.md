# Nepali NLP Study

Reproducible experiments for Nepali news-category classification using the
`Suyogyart/np20ng` dataset.

This repository is organized as a staged research study. It begins with data
auditing and classical TF-IDF baselines, then evaluates duplicate leakage,
training-data size, pretrained sentence embeddings, and transformer
feasibility. Results are only reported when the corresponding experiment was
actually executed.

> **Current headline result:** the strongest verified model is a word-level
> TF-IDF representation with unigram and bigram features followed by
> `LinearSVC(C=1.0)`. On the fixed validation partition it achieved
> **0.7868 accuracy, 0.7516 macro-F1, and 0.7838 weighted-F1**.

## Research questions

The study investigates:

1. What does the Nepali news dataset contain, and what data-quality risks does
   it have?
2. How strong are classical word-based baselines?
3. Do duplicate-aware partitions change the trustworthiness of evaluation?
4. How much does performance improve as the amount of labeled training data
   increases?
5. Can multilingual sentence embeddings or a multilingual transformer improve
   over TF-IDF?
6. How much of the task is influenced by publisher/source identity rather than
   article text?

## Dataset

### Source

- Dataset: [`Suyogyart/np20ng`](https://huggingface.co/datasets/Suyogyart/np20ng)
- Original rows: **231,216**
- Usable rows after removing missing/empty article text: **231,186**
- Categories: **20**
- Publishers/sources: **10**
- Text column: `content`
- Target column: `category`
- Source column: `source`
- Additional column: `heading`

The original local copy is stored at
`data/raw/np20ng_raw.parquet`. Processed data and fixed partitions are stored
under `data/processed/`. Raw data is preserved and is not overwritten by
cleaning.

### Data quality

The dataset has no missing category or source labels. It contains:

- 25 null content values;
- 30 empty or whitespace-only content rows;
- 2 missing or empty headings;
- substantial category imbalance;
- substantial source imbalance;
- duplicate normalized article content;
- 14 duplicate groups containing conflicting categories.

The largest categories include Politics, Society, Sports, Economy, and
Entertainment. Employment is the smallest category. The largest-to-smallest
class ratio is approximately **16.06**, so accuracy alone is not sufficient.
Macro-F1 is reported because it gives each category equal importance.

### Text normalization

The normalization used for duplicate auditing:

1. converts missing values to an empty string;
2. converts text to a string;
3. lowercases the text;
4. replaces runs of whitespace with one space;
5. strips leading and trailing whitespace.

It does not remove Nepali characters or punctuation. Normalized text is used
for grouping and duplicate detection, not as an aggressive linguistic rewrite.

## Evaluation protocol

The primary evaluation uses three fixed partitions:

| Partition | Rows | Purpose |
|---|---:|---|
| Train | 138,711 | Fit vectorizers, classifiers, and other learned components |
| Validation | 46,237 | Select models and compare settings |
| Test | 46,238 | Final evaluation only, after model selection |

The documented protocol is in
[notes/evaluation_protocol.md](notes/evaluation_protocol.md).

### Leakage prevention

- TF-IDF is fitted on training text only.
- Classifiers are fitted on training representations only.
- Validation is used for model selection and comparison.
- The final test partition is not used for tuning.
- Identical normalized documents are kept in one partition for the grouped
  evaluation.
- Source-held-out candidates are selected by predeclared size and category
  coverage rules, not by favorable model scores.

### Grouped versus random splits

The existing partitions are duplicate-aware. The Day 3 audit found:

- grouped split duplicate groups crossing partitions: **0**;
- ordinary stratified random split duplicate groups crossing partitions: **156**.

These splits answer different questions. A random split estimates performance
when examples come from the same overall distribution. A grouped split is
stricter because identical normalized documents cannot appear on both sides.
Differences between the two cannot be attributed to duplicates alone because
their source and category distributions can also differ.

### Metrics

- **Accuracy:** fraction of all examples classified correctly.
- **Macro-F1:** average F1 across categories, giving each category equal weight.
- **Weighted-F1:** F1 averaged using category frequency as the weight.
- **Per-category precision/recall/F1:** identifies minority or overlapping
  categories that aggregate metrics can hide.
- **SVM margin:** highest decision score minus the second-highest decision
  score. This is a rough separation measure, not a calibrated probability.

The primary metric is **macro-F1**. Accuracy and weighted-F1 are secondary
metrics.

## Results

All reported baseline scores below use the same fixed validation examples.

### Day 2: classical baselines

| Experiment | Accuracy | Macro-F1 | Weighted-F1 | Features |
|---|---:|---:|---:|---:|
| Unigram TF-IDF + Logistic Regression | 0.7429 | 0.7028 | 0.7396 | 32,113 |
| Unigram/bigram TF-IDF + Logistic Regression | 0.7678 | 0.7243 | 0.7639 | 300,000 |
| Unigram/bigram TF-IDF + LinearSVC | **0.7868** | **0.7516** | **0.7838** | 300,000 |

The strongest classical configuration uses:

```text
TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=300000,
    sublinear_tf=True,
)
LinearSVC(C=1.0)
```

The strongest model made 9,860 incorrect predictions out of 46,237
validation examples. Common confusions include:

- Business ↔ Economy;
- Arts ↔ Entertainment;
- Economy → Politics;
- Society ↔ Politics/Crime;
- Technology ↔ Automobile.

These errors are consistent with semantically overlapping categories, class
imbalance, source concentration, and possible label ambiguity.

### Day 5: training-size experiment

Day 5 uses the strongest verified classical model and keeps validation fixed.
The training rows are shuffled once with seed 42; nested prefixes form the
25%, 50%, and 100% subsets.

| Training fraction | Training rows | Accuracy | Macro-F1 | Weighted-F1 |
|---:|---:|---:|---:|---:|
| 25% | 34,677 | 0.7542 | 0.7129 | 0.7488 |
| 50% | 69,355 | 0.7719 | 0.7336 | 0.7677 |
| 100% | 138,711 | 0.7868 | 0.7516 | 0.7838 |

Performance improves as more labeled data is used. This is descriptive
evidence that the model benefits from more training data. It does not isolate
sample count from the fact that each subset also learns its own TF-IDF
vocabulary.

### Source diagnostic

A source-only majority-category diagnostic achieved **0.3153 accuracy**. This
is not a text model and is not a competitive baseline. It shows that publisher
identity contains category information and should be considered when
interpreting random-split results.

Candidate source-held-out publishers are selected using at least 1,000
documents and at least 5 observed categories. No source-held-out model score
has been used for the original headline baseline. The completed classical
cross-source benchmark is documented in
[notes/classical_benchmark.md](notes/classical_benchmark.md).

The frozen reference model was evaluated once on the untouched test partition:
**0.7850 accuracy, 0.7504 macro-F1, and 0.7816 weighted-F1**. Its
document-level bootstrap 95% intervals are **[0.7813, 0.7885]** for accuracy
and **[0.7453, 0.7553]** for macro-F1. These intervals are conditional on the
fixed test documents and do not estimate publisher-level uncertainty.

The source-held-out exploratory evaluation used a bounded 10,000-feature word
bigram configuration. Across seven eligible publishers, mean
present-label macro-F1 was **0.4605** with standard deviation **0.1506**.
Source-held-out results are descriptive because category coverage differs by
publisher.

### Calibration and abstention

The frozen LinearSVC reference model now has a leakage-safe experimental
confidence layer. The base classifier is fit on train only; a multiclass
logistic calibrator is fit on validation decision scores, and a threshold is
selected on validation to maximize coverage subject to selective risk at most
0.20. On the untouched test partition, calibrated probabilities had Brier
score **0.0152**, log loss **0.6443**, and 10-bin ECE **0.0209**. The frozen
abstention policy accepted **98.36%** of test examples with **0.7942**
accuracy and **0.7614** macro-F1 among accepted predictions; selective risk
was **0.2058**. This does not establish calibration on unseen publishers or
future data.

The saved bundle is
`results/models/reference_calibrated_bundle.joblib`; metrics, reliability
tables, and the plot are under `results/tables/` and `results/figures/`.

### Day 4: sentence embeddings

Day 4 proposes
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` as a multilingual
sentence-embedding model. The reusable implementation supports:

- dependency and smoke-test checking;
- one vector per article;
- cache keys based on model name, preprocessing version, and document IDs;
- train-only Logistic Regression evaluation on fixed partitions.

The full fixed-partition benchmark is now complete. The model was run with
normalized 384-dimensional embeddings, Logistic Regression and LinearSVC were
compared on validation macro-F1, and only the selected LinearSVC was evaluated
on test. The final artifacts are
`results/tables/sentence_embedding_benchmark_status.json`,
`results/tables/sentence_embedding_validation_metrics.csv`,
`results/tables/sentence_embedding_final_test_metrics.json`, and
`results/tables/sentence_embedding_final_test_predictions.csv`.

Validation selected LinearSVC with **0.6253 macro-F1**. On the untouched test
partition it achieved **0.7114 accuracy, 0.6288 macro-F1, and 0.6979
weighted-F1**. Earlier 1,000/250 and 50,000/12,500 runs remain pilots and are
not substituted for this full result.

### Day 6: transformer preparation

Day 6 prepares an XLM-RoBERTa pipeline. A complete run would need:

- deterministic category-to-integer mapping;
- tokenizer-based sequence-length and truncation audit;
- train/validation tokenization;
- forward-pass and training-step smoke test;
- checkpoint selection using validation macro-F1;
- checkpoint save/reload verification.

The initial Day 6 check found the required packages and model weights missing.
The environment was subsequently prepared and the neural experiments were
advanced. PyTorch, Transformers, and Sentence Transformers are now installed,
XLM-R weights are cached, and a later check found CUDA enabled on an NVIDIA
GeForce RTX 4060 Laptop GPU. Sentence-embedding smoke/pilot and XLM-R smoke,
token audit, and bounded training pilot results are documented in
[notes/neural_experiments.md](notes/neural_experiments.md). Full neural
XLM-R fine-tuning remains incomplete; its pilot scores are not full benchmark
scores. The sentence-embedding benchmark now has one final test evaluation.

The completed sentence-embedding benchmark encoded all 138,711 training,
46,237 validation, and 46,238 test rows. It used
`sentence-transformers==6.2.0`, `torch==2.13.0+cu130`, and
`scikit-learn==1.9.1` on an NVIDIA GeForce RTX 4060 Laptop GPU. Larger CUDA
XLM-R pilots used 8,192 training/2,048 validation examples and remain pilots.

The XLM-R condition runner and protocol manifest are
[src/day8_neural_experiments.py](src/day8_neural_experiments.py),
`results/tables/xlmr_benchmark_protocol.json`, and
`results/tables/xlmr_condition_summary.csv`. A one-epoch CUDA sweep was
bounded to 8,192 training and 2,048 evaluation rows per condition because the
existing 8,192-row pilot took 443 seconds before multi-fold evaluation. A and
B completed as bounded runs; only the Aarthiknews C fold reached test
prediction before the sweep was stopped. These are not full-data or complete
LOSO results.

### Day 7: model comparison

Day 7 consolidates available comparable results. Missing experiments are
marked as unavailable rather than assigned zero scores. The current
defensible conclusion is:

> Bigram TF-IDF with LinearSVC is the strongest verified model in this
> repository so far. The full MiniLM embedding benchmark is now complete, but
> it does not exceed the classical reference; XLM-R remains pilot-only.

### Classical cross-source benchmark

The reusable implementation is
`src/classical_benchmark.py`. It verifies partition integrity, writes final
test predictions and decision scores, runs leave-one-source-out evaluation,
compares the random and duplicate-aware splits, predicts source from text,
and creates a review table for conflicting duplicate labels.

Character n-gram and word-plus-character experiments were blocked by resource
limits in the current environment. Existing verified LogisticRegression
results remain in `results/tables/baseline_comparison.csv`; no replacement
score was fabricated.

### Matched 50k A/B comparison

The matched-budget experiment uses immutable manifests under
`results/tables/matched_50k_manifests/`: 50,000 training rows and 10,000 rows
each for validation and test, with all 20 labels present. The same manifests
were used for TF-IDF, MiniLM, and XLM-R within each condition. Consolidated
results are in `results/tables/matched_50k_ab_comparison.csv`.

| Condition | Model | Test accuracy | Test macro-F1 |
|---|---|---:|---:|
| A | TF-IDF + LinearSVC | 0.7640 | 0.7208 |
| A | MiniLM + LinearSVC | 0.6985 | 0.6046 |
| A | XLM-R | **0.8364** | **0.7921** |
| B | TF-IDF + LinearSVC | 0.7590 | 0.7234 |
| B | MiniLM + LinearSVC | 0.6982 | 0.6120 |
| B | XLM-R | **0.8220** | **0.7782** |

These are complete one-epoch matched-budget results, not full-data results.
The corrected matched-manifest audit completed on 140,000 A/B manifest
rows. It found 22,329 exact-verified Jaccard-0.8 pairs in 21,703 groups,
including 95 groups crossing A boundaries and 82 crossing B boundaries.
Only 42 A test rows (0.420%) and 33 B test rows (0.330%) belonged to a group
containing a same-condition training row. The affected XLM-R rows were not
more accurate than unaffected rows. TF-IDF and MiniLM diagnostics are recorded
alongside the XLM-R results in
`results/tables/matched_50k_near_duplicate_model_impact.json`.
The partitions therefore remain exact-normalized-duplicate-aware, not fully
near-duplicate-safe.

## Reproducibility

### Requirements

The project was developed and tested with:

- Windows;
- Python 3.13;
- pandas;
- NumPy;
- scikit-learn;
- SciPy;
- PyArrow;
- Matplotlib;
- Jupyter;
- pytest.

Optional dependencies for uncompleted experiments:

- `sentence-transformers` for Day 4;
- `transformers` and PyTorch for Days 6–7 transformer work.

The repository does not currently contain a locked `requirements.txt` or
`pyproject.toml`. Use the project `.venv` or another explicitly selected
environment and record package versions before comparing new results.

### Run the Day 3 audit

From the repository root:

```powershell
.\.venv\Scripts\python.exe src\day3_evaluation.py
```

This writes split metadata, duplicate audits, source/category tables, source
diagnostics, and a partition-distribution figure under `results/`.

### Run the reusable Day 4–7 modules

```powershell
.\.venv\Scripts\python.exe src\day4_embeddings.py
.\.venv\Scripts\python.exe src\day5_training_size.py
.\.venv\Scripts\python.exe src\day6_transformer_setup.py
.\.venv\Scripts\python.exe src\day7_model_comparison.py
.\.venv\Scripts\python.exe src\classical_benchmark.py --experiment final-test
.\.venv\Scripts\python.exe src\classical_benchmark.py --experiment source-held-out
.\.venv\Scripts\python.exe -m src.calibration_abstention
```

Day 4 and Day 6 report explicit blocked statuses when optional dependencies or
model files are unavailable. They do not silently download or fabricate
results.

### Run the tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_day3_splits.py tests\test_classical_benchmark.py tests\test_day8_neural.py tests\test_calibration_abstention.py -q
```

The tests verify:

- no normalized duplicate group crosses grouped partitions;
- original document indices are disjoint across train, validation, and test.
- final test and source-held-out metric artifacts use explicit category
  handling and reproducible configurations.

### Execute notebooks

The notebooks are intended as readable demonstrations of the reusable modules:

```powershell
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace notebooks\04_day4_embeddings.ipynb
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace notebooks\05_day5_training_size.ipynb
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace notebooks\06_day6_transformer_setup.ipynb
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace notebooks\07_day7_model_comparison.ipynb
```

For consistent notebook execution, select the project `.venv` as the Jupyter
kernel. Avoid using a different global Python environment because optional
packages may appear available there while remaining absent from the project
environment.

## Repository structure

```text
.
├── data/
│   ├── raw/                 # immutable source data
│   └── processed/           # cleaned data and fixed partitions
├── notebooks/
│   ├── 01_data_explore.ipynb
│   ├── 02_data_audit_and_baselines.ipynb
│   ├── 03_.ipynb
│   ├── 04_day4_embeddings.ipynb
│   ├── 05_day5_training_size.ipynb
│   ├── 06_day6_transformer_setup.ipynb
│   └── 07_day7_model_comparison.ipynb
├── notes/                  # concise daily records and protocol
├── results/
│   ├── figures/             # generated plots
│   └── tables/              # generated metrics and audit tables
├── src/
│   ├── day3_evaluation.py
│   ├── day4_embeddings.py
│   ├── day5_training_size.py
│   ├── day6_transformer_setup.py
│   └── day7_model_comparison.py
├── tests/
│   └── test_day3_splits.py
└── explanations/            # local ignored learning guides
```

The `explanations/` directory is intentionally ignored by Git. It contains
personal study notes that explain the research questions, methods, results,
limitations, and likely questions for each day.

## Reproducibility and reporting rules

When extending this project:

1. Inspect existing artifacts before rerunning expensive experiments.
2. Preserve raw data.
3. Use repository-relative paths or a project-root helper.
4. Fix and record random seeds.
5. Fit learned preprocessing on training data only.
6. Keep validation and test roles separate.
7. Report actual metrics only after execution.
8. Record package versions, model identifiers, hardware, batch sizes, and
   truncation settings for neural experiments.
9. Save predictions and per-category metrics when an experiment is run.
10. Explain whether a split is random, duplicate-aware/grouped, or
    source-held-out.
11. Do not interpret a source-held-out score drop as publisher writing style
    alone because source and category are strongly associated.
12. Treat missing results as unavailable, not as zero.

## Limitations

This study currently has several important limitations:

- Classical headline scores are supplemented by a single frozen final test
  evaluation; neural pilots remain validation-only.
- Source-held-out classical evaluation is complete for seven eligible
  publishers, while neural source-held-out evaluation was not run.
- XLM-R has only been run in bounded pilots; the full MiniLM embedding result
  is documented separately and does not exceed the classical reference.
- Category/source association may create shortcuts in random or grouped
  evaluations.
- Duplicate groups with conflicting labels remain unresolved.
- The Day 5 training-size experiment changes the learned TF-IDF vocabulary at
  each size.
- Per-category and source/category subgroup claims require adequate sample
  sizes; tiny groups should not be overinterpreted.
- The repository does not yet provide a fully pinned dependency lockfile.

## Next recommended experiments

In order:

1. Decide whether to install and record the Day 4 embedding dependencies.
2. Run the embedding smoke test before full encoding.
3. Add a source-held-out evaluation using the predeclared candidate rule.
4. Decide whether transformer downloads and CPU/GPU runtime are feasible.
5. Run the Day 6 transformer smoke test before full fine-tuning.
6. Evaluate the selected model once on the untouched test partition.
7. Add a complete per-category and source-held-out comparison report.
8. Investigate conflicting duplicate labels without silently relabeling data.

## Citation and data-use notes

This repository uses the `Suyogyart/np20ng` dataset. Consult the dataset
card and upstream project for its licensing, attribution, and data-use
requirements before redistribution or publication.

This README documents the current local research workflow and measured
artifacts. It is not a claim that all proposed model families have been
completed.
