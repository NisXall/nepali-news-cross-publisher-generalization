# Evaluation protocol

The primary evaluation uses three fixed partitions already present under
`data/processed/`: train (138,711 rows), validation (46,237 rows), and test
(46,238 rows). The seed used when the partitions were generated is 42 for the
outer split and 43 for the inner split. Text vectorizers and classifiers must
be fitted on train only; validation is for model selection, and test remains
untouched until a final model is selected.

The partitions are duplicate-aware: identical normalized documents are assigned
to one partition only. Normalization lowercases text, replaces runs of
whitespace with one space, and strips leading/trailing whitespace. It does not
remove Nepali characters or punctuation. The Day 3 audit verifies this property
and compares it with a descriptive ordinary stratified random split, where
duplicate groups may cross partitions.

Random and grouped splits answer different questions. A random split estimates
performance when documents are sampled from the same overall distribution.
The grouped split is stricter because repeated documents cannot appear on both
sides. Neither is a source-held-out test: a source-held-out evaluation must
keep the selected publisher completely out of training.

For source-held-out experiments, candidates are selected before looking at
model scores: at least 1,000 documents and at least 5 observed categories.
This excludes tiny or single-category publishers while acknowledging that
category coverage still differs by source. Candidate selection does not claim
that every source has full category coverage. The classical benchmark now runs
leave-one-source-out evaluation for the seven eligible publishers and reports
macro-F1 over labels present in each held-out source, while listing absent
labels explicitly.

The final reference test is evaluated only after selecting the configuration
from training and validation. LinearSVC decision scores are retained for
calibration-ready analysis but are not probabilities. Document-level bootstrap
intervals are conditional on the fixed test partition and do not represent
publisher-level uncertainty.
