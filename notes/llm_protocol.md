# LLM Protocol: Instruction-Following Baseline under Publisher Shift

Status: **protocol planning only; no model download, inference, or LLM result
exists.**

## 1. Experiment identity

**Experiment name:** Instruction-Following LLM Baseline for Nepali News Topic
Classification under Publisher Shift

**Research question:** How does a multilingual instruction-following LLM
classify Nepali news from held-out publishers under zero-shot and fixed
class-balanced few-shot prompting, and how does its in-source versus
cross-source gap compare with the existing trained models?

This is a new experiment family. Existing classical, MiniLM, and XLM-R
artifacts are immutable. New artifacts must use `results/llm/`,
`results/tables/llm_*`, `results/figures/llm_*`, and `notes/llm_*`.

## 2. Existing data facts verified before planning

### 2.1 Fixed partitions

The original fixed partitions are:

| Partition | Rows | Use |
|---|---:|---|
| `data/processed/train.parquet` | 138,711 | Demonstration pool and any approved training-side use |
| `data/processed/validation.parquet` | 46,237 | Prompt/condition selection only |
| `data/processed/test.parquet` | 46,238 | One final evaluation after freezing decisions |

The repository protocol uses seed 42 for the primary fixed partitions. Text
normalization for duplicate auditing converts missing values to empty strings,
casts to string, lowercases, collapses whitespace, and strips leading/trailing
whitespace. Nepali characters and punctuation are retained.

The model input is **`content` only**. `heading`, `source`, and publisher
identity are not supplied to the LLM. This is implemented by
`src/classical_benchmark.py:text()`, which reads `frame["content"]`; the
default path does not use `heading` or `source`.

### 2.2 Eligible C publishers and complete test sets

The C eligibility rule is at least 1,000 documents and at least 5 observed
categories. The complete eligible C test sets are the following parquet
manifests. Counts were read directly from
`results/tables/matched_50k_c_manifests/*_test.parquet`.

| Held-out publisher | Test rows | Categories present | Per-category counts |
|---|---:|---:|---|
| Aarthiknews | 25,965 | 7 | Automobile 900; Bank 7,638; Business 4,706; Economy 5,247; Employment 429; Health 4,442; Market 2,603 |
| Ekantipur | 29,875 | 10 | Business 3,390; Entertainment 3,592; Health 1,433; Lifestyle 71; Literature 2,627; Market 1,537; Opinion 2,405; Sports 6,655; Technology 1,129; World 7,036 |
| Gorkhapatra | 5,059 | 8 | Economy 1,037; Education 806; Entertainment 504; Health 825; Opinion 409; Society 306; Sports 523; Tourism 649 |
| Nagarik | 11,106 | 6 | Arts 2,559; Economy 2,207; Education 1,355; Politics 2,136; Society 2,112; Technology 737 |
| Nepalkhabar | 69,504 | 10 | Automobile 739; Crime 225; Economy 14,010; Entertainment 2,926; Lifestyle 53; Literature 31; Politics 22,847; Society 25,382; Technology 726; World 2,565 |
| Onlinekhabar | 66,716 | 15 | Automobile 1,534; Bank 2,888; Crime 4,027; Employment 846; Entertainment 15,642; Health 2,227; Lifestyle 2,327; Literature 1,491; Market 233; Opinion 1,131; Politics 6,038; Sports 17,214; Technology 1,477; Tourism 1,811; World 7,830 |
| Setopati | 15,541 | 8 | Arts 5,153; Automobile 1,979; Business 3,549; Economy 1,152; Employment 656; Health 542; Technology 2,356; Tourism 154 |

The C manifest construction is in
`src/matched_budget_benchmark.py:create_c_manifests()`. For each held-out
publisher, it sets `test` to all rows from that publisher, removes that
publisher from the cleaned corpus, stratifies the remaining rows into
train/validation, samples 50,000 train rows and 10,000 validation rows, and
writes the three manifests. Thus each C training set excludes the held-out
publisher before sampling.

### 2.3 Existing per-document prediction coverage

Verified files with article IDs and `predicted_category`:

- XLM-R C: all seven folds:
  `matched50k_xlmr_C_{Aarthiknews,Ekantipur,Gorkhapatra,Nagarik,Nepalkhabar,Onlinekhabar,Setopati}_test_predictions.csv`
- XLM-R A/B:
  `matched50k_xlmr_A_test_predictions.csv`,
  `matched50k_xlmr_B_test_predictions.csv`
- TF-IDF A/B:
  `matched50k_tfidf_A_test_predictions.csv`,
  `matched50k_tfidf_B_test_predictions.csv`
- MiniLM A/B:
  `matched50k_minilm_A_test_predictions.csv`,
  `matched50k_minilm_B_test_predictions.csv`

Each verified prediction file contains `original_index`, `source`, `category`,
and `predicted_category`.

Matched C TF-IDF and matched C MiniLM per-document prediction files **do not
exist**. Only their aggregate C result tables exist, so no C per-document
comparison for those two model families may be claimed.

### 2.4 Validation sampling feasibility

The three publishers not included in the seven eligible C folds are
`Crimenews`, `Nepalipatra`, and `Educationpati`. In the original fixed
validation partition they contribute:

| Publisher | Validation rows | Categories |
|---|---:|---|
| Crimenews | 790 | Crime |
| Nepalipatra | 380 | Lifestyle, Tourism |
| Educationpati | 306 | Education |
| **Total** | **1,476** | **4 categories total** |

Their pooled category counts are Crime 790, Education 306, Lifestyle 275, and
Tourism 105. A 400-row sample can be physically drawn, but it cannot be a
representative 20-category stratified validation sample because these three
publishers contain only four categories. Therefore the primary fallback is a
stratified sample from **all 46,237 validation rows**, with the publisher
composition and limitation recorded. Any three-publisher-only smoke result
must be labeled coverage-limited and must not select the final prompt.

## 3. Frozen LLM evaluation protocol

### 3.1 Conditions

Run exactly these two prompt conditions:

1. **Zero-shot:** use the validation-selected instruction with the 20 category
   strings and require one category output. The instruction may include the
   short frozen category descriptions if that is the validation-selected
   wording.
2. **One-example-per-class few-shot:** add exactly one fixed example per
   class (20 demonstrations), selected from the relevant training partition
   only, to the same validation-selected instruction.

The 20 category strings are always offered:

`Arts`, `Automobile`, `Bank`, `Business`, `Crime`, `Economy`, `Education`,
`Employment`, `Entertainment`, `Health`, `Lifestyle`, `Literature`, `Market`,
`Opinion`, `Politics`, `Society`, `Sports`, `Technology`, `Tourism`, `World`.

Metrics are reported over labels present in each held-out publisher, matching
the existing C protocol. Also report pooled C results separately from the
mean and standard deviation across publisher-level results. Do not substitute
pooled results for the publisher mean/std.

### 3.2 Inputs and samples

- Input: `content` only.
- No `heading`, `source`, publisher name, or source metadata.
- Fixed truncation must be chosen before validation scoring and recorded in
  tokens/characters and tokenizer name.
- Seed: 42.
- Draw the approximately 400-row validation sample first from all validation
  rows, stratified by category with seed 42. Save its IDs before any test
  sampling or inference.
- For each C publisher, sample approximately 200 test rows from the complete
  eligible C test set using the natural publisher distribution; use all rows
  if a publisher has fewer than 200.
- Exclude the validation IDs and every row in their verified MinHash
  near-duplicate groups from every C test sampling pool and demonstration
  pool. Verify and log zero ID overlap and zero excluded near-duplicate
  overlap.
- Save test IDs and sampling manifests before any inference.
- Optionally evaluate an in-source A sample of 1,000 rows, with IDs saved
  before inference. This is a secondary comparison, not a replacement for C.

### 3.3 Demonstrations

For each C fold, demonstrations come only from that fold's training
partition, never from its held-out publisher or test set. The one-example
per-class set is fixed and seeded before inference.

Check demonstrations against the corresponding test IDs and near-duplicate
groups using the repository near-duplicate method in `src/near_dup_audit.py`:
word-5 shingles, 64 MinHash permutations, 16 bands of 4 rows, candidate
generation followed by exact Jaccard verification at 0.8. The older
`matched_budget_benchmark.py` audit helper has different internal LSH
parameters; use `src/near_dup_audit.py` for this new check and record the
implementation path.

Before building few-shot prompts, measure the token length of the 20
demonstrations plus each test article. If this exceeds the selected model's
practical context on the 8 GB RTX 4060 Laptop GPU, shorten demonstrations to
a fixed recorded limit without dropping classes. Demonstration and article
limits must be recorded separately.

### 3.4 Model, decoding, and invalid outputs

- Exact LLM checkpoint/provider: **unknown; must be selected and frozen before
  implementation**.
- License, language coverage, quantization, context length, and pretraining
  exposure: **unknown until the model is selected**.
- Decoding must be deterministic: temperature 0 where supported; otherwise
  the provider's lowest-determinism setting, recorded exactly.
- Output parser must accept only an exact category string after documented
  whitespace/quote normalization.
- Invalid, ambiguous, refusal, empty, or multi-label outputs are not silently
  mapped to a category. Record them as invalid and report the invalid-output
  rate.
- Retry policy: retry only infrastructure failures such as out-of-memory or
  process crash, with each retry logged. Constrained decoding means there are
  no parse retries. Invalid outputs remain invalid.

### 3.5 Validation and test freeze

Use the approximately 400 validation rows to select the instruction and
allowed wording. Because the three non-C publishers do not cover all 20
categories, use the all-validation fallback described above. Keep the
validation sample IDs fixed and stratified with seed 42. The validation sample
is drawn before test sampling, and its IDs and near-duplicate groups are
excluded from C test and demonstration pools.

After prompt/configuration selection, freeze the prompt, label descriptions,
model/checkpoint, tokenizer, truncation, decoder settings, parser, retry
policy, and sampling manifests. Touch the test sample only once per condition.

For each publisher, define "present labels" from the sampled test set's true
labels. Report accuracy, weighted-F1, and present-label macro-F1 using the
existing metric convention (`labels` equal to labels present in `y_true`).
Rescore XLM-R, and any re-run TF-IDF or MiniLM, on exactly the same sampled
IDs when making matched comparisons. Never compare these sampled metrics
directly with the existing full-fold C metrics.

### 3.6 Budget and go/no-go rule

The exact inference budget, provider/API cost, local runtime limit, and
selected model are **unknown until a model and execution route are chosen**.
Before the full run, execute a validation-only smoke test and record runtime,
tokens, failures, invalid outputs, and estimated total cost.

If the smoke test indicates that the full run will not finish by **2026-11-03**,
reduce the per-publisher test sample size equally across both conditions
before any test inference. Do not reduce only a selected condition or
publisher. Record the revised sample size as a protocol amendment.

## 4. Comparison rules and threats

- Compare LLM conditions on identical saved test IDs within each publisher.
- Do not claim matched training budgets: the LLM uses zero or 20 examples,
  XLM-R used a matched 50k one-epoch budget, and the frozen classical model
  used 138,711 training rows.
- A-versus-C differences are descriptive only because sampled C macro-F1 uses
  labels present in each publisher and C has different publisher composition.
- Report accuracy, present-label macro-F1, weighted-F1, per-category support,
  invalid-output rate, runtime/token/cost budget, and independent intervals.
- Preserve every prompt, label description, model version, tokenizer,
  decoding setting, sampling manifest, raw response or privacy-safe hash,
  retry/failure log, and parser version under the new LLM artifact paths.

Threats to validity:

1. The selected LLM may have pretrained exposure to these news sites or
   overlapping web text.
2. Results use one prompt family, one model/checkpoint, and possibly one
   quantization.
3. Approximately 200 examples per publisher produce wide publisher-level
   intervals, especially for minority categories.
4. The C category distributions differ strongly across publishers.
5. Demonstration selection and category descriptions can influence results.
6. External API behavior, model updates, rate limits, and hidden system
   prompts may reduce reproducibility.
7. Test-side near-duplicate exposure is audited but not assumed to be zero.
8. Prompt selection uses validation rows from publishers that also occur in
   the C test pools. Articles and verified near-duplicate groups are excluded,
   but publisher-level distribution overlap remains.

## 5. Required pre-inference checklist

- [ ] Select and record the exact model/checkpoint/provider, license, and
  quantization.
- [ ] Freeze category descriptions and both prompt templates.
- [ ] Freeze tokenizer/truncation, decoding, parser, retry, and budget rules.
- [ ] Draw and hash validation/test IDs before inference.
- [ ] Select demonstrations from each C training partition only.
- [ ] Run the near-duplicate check against demonstrations and test samples.
- [ ] Run the validation-only smoke test and apply the date-based go/no-go rule.
- [ ] Do not modify any existing classical, MiniLM, or XLM-R artifact.

## Amendments

**2026-11-01 protocol amendment:** The validation sample is drawn first from
all validation rows and is excluded, together with its verified MinHash
near-duplicate groups, from C test sampling and demonstrations. The LLM
experiment now has two conditions only: validation-selected zero-shot and
one-example-per-class few-shot. Constrained decoding removes parse retries;
only infrastructure failures may be retried. LLM metrics use labels present in
the sampled test set, and matched model comparisons must rescore on those same
IDs. Few-shot context length is measured before prompt construction, with a
fixed demonstration-shortening rule if required by the 8 GB GPU. The
publisher overlap between validation prompt selection and C test pools is an
explicit threat.

**2026-11-01 truncation amendment:** Every candidate model uses the first 400
tokens of each article for zero-shot and few-shot evaluation. Each
few-shot demonstration uses the first 100 tokens. The limits are measured
with each candidate's own tokenizer and recorded separately; they are not
replaced by the XLM-R proxy measurement.

**2026-11-01 Qwen validation amendment:** The validation-only bake-off selected
`Qwen/Qwen2.5-3B-Instruct` for continued planning based on the frozen
100-row validation subset. The exact revision is
`aa8e72537993ba99e69dfaafa59ed015b17504d1`, loaded locally with
bitsandbytes NF4 and float16 compute through Transformers. On the same 50
training articles, its tokenizer measured median 1,852 tokens, 95th
percentile 3,240, and maximum 6,967. The zero-shot labels-only constrained
scoring result on the 100 validation rows was accuracy 0.1900, present-label
macro-F1 0.1141, and weighted-F1 0.1823, versus majority accuracy 0.1300.
This result is validation-only and does not authorize or include test
inference. The full per-row bake-off output is
`results/tables/llm_qwen_bakeoff_100.csv`, and model settings are recorded in
`results/llm/qwen_bakeoff_model_config.json`.

**2026-11-01 Gemma final-attempt amendment:** The final permitted model attempt
is `unsloth/gemma-3-4b-it-bnb-4bit`, evaluated only on the frozen 100-row
validation bake-off subset with the same deterministic labels-only scoring
protocol and tokenizer-specific 400-token article/100-token demonstration
limits. Proceed beyond validation only if accuracy is at least 0.35. If that
threshold is not met, record “LLM stage stopped: no tested model clearly above
baseline” and do not run any further LLM data. Test articles remain
forbidden.
