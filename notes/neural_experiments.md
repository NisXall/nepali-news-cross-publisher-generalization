# Neural experiments

## Environment

The neural environment was installed from
`requirements-neural.txt`:

- Python 3.13.7
- PyTorch 2.14.1+cpu
- Transformers 5.20.2
- Sentence Transformers 6.2.0
- Accelerate 1.15.0
- CUDA unavailable
- CPU-only execution

The selected models were downloaded to the local Hugging Face cache. Model
weights and embedding caches are not repository artifacts.

## Sentence embeddings

Model:

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

The smoke test returned finite 384-dimensional embeddings for Nepali examples.
Embedding caches include model name, preprocessing version, split identity,
document IDs, and batch size in their key. Full-run cache reuse additionally
requires the exact partition row count, 384 dimensions, and finite values.

A bounded pilot used a deterministic stratified sample of 1,000 training rows
and 250 validation rows from the existing fixed partitions. It is not a
full-dataset benchmark.

| Classifier | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| LogisticRegression | 0.5200 | 0.3427 | 0.4881 |
| LinearSVC | 0.6280 | 0.5376 | 0.6093 |

The pilot took about 43 seconds for embedding generation and less than two
seconds for classifier fitting. Results are stored in
`results/tables/day8_embedding_pilot.csv`.

### Full fixed-partition benchmark

The full benchmark used the existing fixed partitions without regenerating
splits:

| Partition | Rows |
|---|---:|
| Train | 138,711 |
| Validation | 46,237 |
| Test | 46,238 |

Configuration:

- Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Revision: `main` from the local Hugging Face cache
- Input: `content`
- Preprocessing: missing content becomes `""`, then values are cast to strings;
  no additional normalization
- Embedding dimension: 384
- Normalized embeddings: yes
- Batch size: 128
- Seed: 42
- Device: NVIDIA GeForce RTX 4060 Laptop GPU
- Selection: validation macro-F1

| Classifier | Validation accuracy | Validation macro-F1 | Validation weighted-F1 |
|---|---:|---:|---:|
| LinearSVC | 0.7090 | **0.6253** | 0.6956 |
| LogisticRegression | 0.6955 | 0.6102 | 0.6859 |

LinearSVC was selected without inspecting test labels. Its single final test
evaluation achieved **0.7114 accuracy, 0.6288 macro-F1, and 0.6979
weighted-F1**. Test predictions contain 46,238 unique fixed-partition
documents. The final test metrics and predictions are stored in
`results/tables/sentence_embedding_final_test_metrics.json` and
`results/tables/sentence_embedding_final_test_predictions.csv`; the complete
run manifest is `results/tables/sentence_embedding_benchmark_status.json`.
The classifier bundle is stored under the ignored `results/models/` directory.

## XLM-RoBERTa

Model: `xlm-roberta-base`.

The local-cache smoke test completed on CPU:

- input shape: 8 x 256;
- logits shape: 8 x 20;
- finite loss: true;
- hidden size: 768.

The token audit sampled 500 rows from each fixed partition. With maximum
sequence length 256:

| Partition | P50 | P90 | P95 | P99 | Truncation rate |
|---|---:|---:|---:|---:|---:|
| Train | 282 | 888 | 1,294 | 2,402 | 56.8% |
| Validation | 315 | 940 | 1,225 | 2,743 | 60.6% |
| Test | 313 | 978 | 1,244 | 2,745 | 62.2% |

These rates show that max length 256 would discard substantial article text.
The audit is a sample-based feasibility signal, not a full partition
distribution.

A one-epoch CPU pilot used a deterministic stratified sample of 256 training
rows, 64 validation rows, batch size 4, learning rate 2e-5, and max length
128:

- mean training loss: 2.8866;
- validation accuracy: 0.1094;
- validation macro-F1: 0.0104;
- runtime: approximately 243 seconds.

This pilot confirms that tokenization, forward loss, backpropagation, and
classification outputs work. It is not a meaningful model comparison and is
not used as a final result.

## Current status

Completed:

- Dependencies installed and versions recorded.
- Sentence-embedding smoke and bounded classifier pilot.
- Embedding cache implementation.
- XLM-R tokenizer/model/loss smoke test.
- XLM-R token-length/truncation audit.
- Bounded XLM-R CPU training pilot.
- Larger CUDA pilots after detecting an available NVIDIA GeForce RTX 4060
  Laptop GPU with approximately 8 GB VRAM.

## GPU follow-up pilots

The current environment was rechecked on 2026-10-24 and reported:

- NVIDIA GeForce RTX 4060 Laptop GPU;
- 8,188 MiB total VRAM, approximately 6,924 MiB free at inspection;
- CUDA-enabled PyTorch 2.13.0+cu130;
- `torch.cuda.is_available() == True`.

The sentence-embedding pilot was expanded to 50,000 training and 12,500
validation articles. It ran on CUDA and returned:

| Classifier | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| MiniLM + LogisticRegression | 0.6762 | 0.5843 | 0.6645 |
| MiniLM + LinearSVC | 0.6942 | 0.6066 | 0.6797 |

This remains a pilot using subsets of the fixed partitions. The explicit
artifacts are `results/tables/day8_embedding_pilot_gpu_50000.csv` and
`results/tables/day8_embedding_status_gpu_50000.json`.

XLM-R was expanded to 8,192 training and 2,048 validation articles, one epoch,
batch size 4, maximum sequence length 256, and learning rate 2e-5. It ran on
the RTX 4060, used approximately 5,357 MiB peak allocated VRAM, and took
227.1 seconds:

| Metric | Result |
|---|---:|
| Mean training loss | 1.3122 |
| Validation accuracy | 0.7598 |
| Validation macro-F1 | 0.6624 |

This is a substantially larger and more informative pilot than the earlier
CPU run, but it is still not a full benchmark or final test result. The
explicit artifact is `results/tables/day8_xlmr_training_pilot_gpu_8192.json`.

Not completed:

- Full-data XLM-R fine-tuning and final test evaluation.
- Complete XLM-R A/B/C comparison. The reproducible bounded runner completed
  A and B at 8,192 training/2,048 evaluation rows and reached the
  Aarthiknews C test fold before the multi-fold sweep was stopped for compute.
- The remaining C sources were not run and have no reported scores.

## XLM-R bounded A/B/C comparison

The new runner uses one seed (42), `xlm-roberta-base`, maximum length 256,
learning rate `2e-5`, one epoch, batch size 4, gradient accumulation 4,
effective batch size 16, and CUDA bf16. Checkpoints are saved under
`results/checkpoints/xlmr_*`; per-condition predictions, per-class tables, and
confusion matrices are under `results/tables/`.

| Condition | Validation macro-F1 | Test macro-F1 | Test accuracy | Status |
|---|---:|---:|---:|---|
| A ordinary random | 0.5927 | 0.5696 | 0.7275 | bounded |
| B fixed duplicate-aware | 0.5957 | 0.5806 | 0.7300 | bounded |
| C Aarthiknews | not finalized | 0.1301 | 0.4082 | partial bounded |

Condition A permits exact normalized duplicate leakage, while B uses the
existing fixed duplicate-aware partitions. The repository has no MinHash/LSH
implementation for the documented Jaccard-0.8 near-duplicate safeguard; the
runner records this discrepancy and does not claim that safeguard.
- Full XLM-R fine-tuning on 138,711 training rows.
- Validation checkpoint selection for a full model.
- Test evaluation for a neural model.
- Source-held-out neural evaluation.
- Neural confidence calibration.

Full neural training is still not claimed because the complete fixed training
partition has 138,711 articles, and XLM-R sequence length and checkpoint
selection require a longer controlled run. The classical model remains the
only frozen final benchmark model.

## Matched 50k A/B comparison

The matched A/B manifests use 50,000 training rows and 10,000 validation/test
rows per condition, with all 20 labels present. TF-IDF, MiniLM, and XLM-R use
the same rows within each condition. XLM-R uses one epoch, maximum length 256,
learning rate `2e-5`, batch size 4, gradient accumulation 4, effective batch
size 16, CUDA bf16, and seed 42.

| Condition | Model | Test accuracy | Test macro-F1 |
|---|---|---:|---:|
| A | TF-IDF + LinearSVC | 0.7640 | 0.7208 |
| A | MiniLM + LinearSVC | 0.6985 | 0.6046 |
| A | XLM-R | **0.8364** | **0.7921** |
| B | TF-IDF + LinearSVC | 0.7590 | 0.7234 |
| B | MiniLM + LinearSVC | 0.6982 | 0.6120 |
| B | XLM-R | **0.8220** | **0.7782** |

The matched XLM-R A and B runs each took about 36.5 minutes on the RTX 4060,
with observed temperature up to 58°C and peak memory around 6.7 GB of 8.2 GB.
They are complete for the matched manifests but remain bounded-budget results
relative to the full 138,711-row protocol.

### Matched near-duplicate impact diagnostic

The corrected word-5 MinHash/LSH audit used a 0.5 candidate threshold followed
by exact Jaccard verification at 0.8. Across the six A/B manifests (140,000
rows), it found 22,329 qualifying pairs in 21,703 connected groups. There were
95 groups crossing A internal boundaries and 82 crossing B internal
boundaries, so B is not described as fully near-duplicate-safe.

The test-impact diagnostic found 42 affected A test rows (0.420%) and 33
affected B test rows (0.330%), defined as test rows in a verified group with a
same-condition training row. Three affected rows in each condition had
conflicting training labels. For XLM-R, affected rows were less accurate than
unaffected rows; the corresponding TF-IDF and MiniLM diagnostics are saved in
`results/tables/matched_50k_near_duplicate_model_impact.json`.

Per-document predictions regenerated for the strongest A/B baselines use
TF-IDF LinearSVC and multilingual
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, with existing
embedding caches reused. The consolidated A/B/C table is
`results/tables/matched_50k_abc_consolidated.csv`.

## Reproducible commands

```powershell
.\.venv\Scripts\python.exe src\day6_transformer_setup.py
.\.venv\Scripts\python.exe -m pytest tests\test_day8_neural.py -q
```

The reusable functions are in `src/day8_neural_experiments.py`.
