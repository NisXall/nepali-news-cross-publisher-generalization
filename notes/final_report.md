# Nepali NLP Benchmark: Cross-Source Generalization

## 1. Research question and contributions

This study measures how Nepali news topic classifiers generalize across fixed
duplicate-aware partitions and across publishers. It also tests how much
duplicate leakage, source/category association, and label imbalance affect
interpretation. Contributions are a reproducible classical benchmark,
duplicate-aware and source-held-out analyses, bounded multilingual neural
feasibility experiments, and leakage-safe confidence calibration with
abstention for the frozen reference classifier.

## 2. Dataset and preprocessing

The study uses `Suyogyart/np20ng`: 231,216 original rows, 231,186 usable
rows, 20 categories, 10 publishers, and `content`, `heading`, `source`, and
`category` fields. Missing content is represented as empty text. Duplicate
normalization lowercases text, collapses whitespace, and trims boundaries; it
does not remove Nepali characters or punctuation. The benchmark text strategy
is content-only and is applied consistently.

## 3. Duplicate auditing and conflicting labels

The fixed grouped partitions contain no normalized duplicate group crossing
partitions. The audit found 14 duplicate groups with conflicting categories;
these labels remain unchanged. A review table is in
`results/tables/duplicate_conflict_review.csv`.

## 4. Imbalance and source/category association

The dataset has substantial category and publisher imbalance; the
largest-to-smallest category ratio is approximately 16.06. A source-only
majority-category diagnostic reached 0.3153 accuracy, demonstrating that
publisher identity is informative and can inflate random-distribution
interpretations. Macro-F1 is therefore the primary aggregate metric.

## 5. Split protocol and leakage safeguards

The fixed partitions are train 138,711, validation 46,237, and test 46,238,
with seed 42 and disjoint original indices. Vectorizers and classifiers are
fit on train only. Validation is used for selection and calibration; test is
evaluated once after freezing the policy. Source-held-out candidates use the
predeclared rule of at least 1,000 documents and 5 observed categories.

## 6. Classical baselines and training size

On the fixed validation partition:

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Unigram TF-IDF + LogisticRegression | 0.7429 | 0.7028 | 0.7396 |
| Unigram/bigram TF-IDF + LogisticRegression | 0.7678 | 0.7243 | 0.7639 |
| Unigram/bigram TF-IDF + LinearSVC | 0.7868 | 0.7516 | 0.7838 |

For the strongest classical configuration, training fractions 25%, 50%, and
100% produced validation macro-F1 0.7129, 0.7336, and 0.7516 respectively.
Character and word-plus-character runs are BLOCKED by resource cost, not
assigned zero scores.

## 7. Random versus duplicate-aware comparison

The ordinary stratified random split had 156 duplicate groups crossing
partitions. The bounded 10,000-feature word-bigram LinearSVC comparison gave
duplicate-aware macro-F1 0.7259 versus random macro-F1 0.7239. These splits
also differ in source and category distributions, so the difference is not
attributed to duplicates alone.

## 8. Source-held-out generalization

Seven eligible publishers were evaluated with the bounded 10,000-feature
classical configuration. Mean present-label macro-F1 was 0.4605 with standard
deviation 0.1506. Per-source and per-class results are in
`results/tables/source_held_out_summary.csv` and
`results/tables/source_held_out_per_class.csv`. These are descriptive
out-of-source results; category coverage differs by publisher.

## 9. Source-related-text ablation

A limited deterministic stripping function removes URLs, email-like tokens,
and a small set of byline/courtesy prefixes. Its results and diagnostic are
in `results/tables/source_text_ablation.csv` and
`results/tables/source_text_diagnostic.json`. This is not a claim of complete
boilerplate removal, and source-held-out performance cannot be interpreted as
writing-style performance alone because source and category are associated.

## 10. Sentence-embedding results

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` loaded locally
and returned finite 384-dimensional embeddings. A deterministic pilot used
1,000 train and 250 validation rows:

| Classifier | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| LogisticRegression | 0.5200 | 0.3427 | 0.4881 |
| LinearSVC | 0.6280 | 0.5376 | 0.6093 |

These are PILOT results and are not comparable to the full classical
benchmark. Caches include model, preprocessing, split, row identity, and
batch-size information.

## 11. XLM-R results

`xlm-roberta-base` completed a local tokenizer/model forward and loss smoke
test with finite loss and logits of shape 8 x 20. A token audit at maximum
length 256 found sample truncation rates of 56.8% train, 60.6% validation,
and 62.2% test. A one-epoch CPU pilot on 256 train and 64 validation rows
completed, but full fine-tuning was NOT RUN because the environment is
CPU-only and the corpus is large. No XLM-R test score or superiority claim is
made.

## 12. Calibration and abstention

The final reference LinearSVC is fit on train only. A multiclass logistic
calibrator is fit on validation decision scores, and the threshold is selected
on validation to maximize coverage subject to selective risk <= 0.20. The
threshold was 0.2932 and validation coverage was 0.9831.

On untouched test: calibrated Brier score 0.0152, log loss 0.6443, and
10-equal-width-bin ECE 0.0209. The frozen abstention policy accepted 98.36%
of examples, with accepted accuracy 0.7942, accepted macro-F1 0.7614, and
selective risk 0.2058. Raw LinearSVC margins remain non-probabilistic ranking
signals. Calibration on known publishers does not guarantee calibration on
unseen publishers or future data.

## 13. Error analysis and robustness

The strongest classical errors include Business/Economy, Arts/Entertainment,
Economy/Politics, Society/Politics or Crime, and Technology/Automobile.
Conflicting duplicate groups were reviewed without relabeling. The inference
preprocessing maps null input to empty text and preserves Nepali Unicode;
punctuation, mixed Nepali/English, repeated content, short text, and long
text are covered by the content-only path and token/truncation audits.
The model is a topic classifier, not a fact-checking or misinformation
detection system.

## 14. Runtime, memory, environment, reproducibility

The environment is Python 3.13.7 on CPU-only PyTorch 2.14.1+cpu,
Transformers 5.20.2, Sentence Transformers 6.2.0, and Accelerate 1.15.0.
The full classical final test pass and calibration pass were executed on the
fixed partitions. The calibration pass took about 337 seconds. The neural
XLM-R pilot took approximately 243 seconds. Seeds, split sizes, models, and
artifacts are recorded in
`results/tables/neural_reproducibility_manifest.json`; generated model bundles
and caches remain outside Git.

## 15. Conclusions

The strongest verified full benchmark is TF-IDF word unigrams/bigrams with
LinearSVC: test accuracy 0.7850, macro-F1 0.7504, and weighted-F1 0.7816.
Performance is materially lower and more variable for publisher-held-out
evaluation. The sentence and transformer work establishes feasibility only;
it does not show neural superiority. The calibration layer provides a
validation-defined abstention policy, but its slight test risk increase over
the target and publisher shift caveat prevent production-confidence claims.

## 16. Future work

Future work may run full sentence embeddings, GPU-supported XLM-R training,
neural source-held-out evaluation, richer boilerplate removal, and
publisher/time-aware uncertainty analysis. These experiments are NOT RUN and
are not findings of this report.
