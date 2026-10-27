# Model card: Nepali news category benchmark

## Intended use

This project classifies Nepali news articles into the 20 labels represented in
the `Suyogyart/np20ng` dataset. It is intended for research, reproducibility,
error analysis, and educational experimentation.

## Out of scope

The classifier is not a fact checker, misinformation detector, moderation
system, legal/medical authority, or measure of objective article truth.

## Best frozen model

The strongest frozen benchmark is word-level unigram/bigram TF-IDF with
`LinearSVC(C=1.0)`. Its untouched-test metrics are 0.7850 accuracy, 0.7504
macro-F1, and 0.7816 weighted-F1.

## Data and bias

The data contains 20 categories from 10 publishers with substantial
source/category association and class imbalance. A source-held-out evaluation
showed much lower and variable performance than the fixed grouped test:
present-label macro-F1 averaged 0.4605 across seven eligible publishers.
Performance should not be assumed to transfer to new publishers, time periods,
or writing styles.

## Confidence

Raw LinearSVC decision scores are ranking signals, not probabilities. The
repository now includes an experimental calibrated bundle:
`results/models/reference_calibrated_bundle.joblib`. Its multiclass logistic
calibrator was fitted on validation decision scores only, and its abstention
threshold was selected on validation data for a target selective risk of 0.20.
On the untouched test set, the policy accepted 98.36% of examples with 0.7942
accuracy and 0.7614 macro-F1 among accepted examples. These are research
metrics, not a production guarantee. Calibration on known publishers does not
guarantee calibration on unseen publishers or future data.

Reliability tables and the final figure are in
`results/tables/calibration_test_reliability.csv` and
`results/figures/calibration_reliability.png`.

## Neural status

Sentence-embedding and XLM-R smoke tests and bounded pilots are complete, but
no neural model has been selected or evaluated on the final test partition.
The pilot metrics must not be compared as full benchmarks.

## Known limitations

- Conflicting duplicate labels remain in the data.
- Category coverage differs by publisher.
- Test metrics are conditional on the fixed test partition.
- Neural full training was not completed on the CPU-only environment.
- The dataset's labels and source composition may encode historical or
  publisher-specific bias.
