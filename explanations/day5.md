# Day 5: How much labeled data is needed?

## Question

Does the strongest simple model improve as it receives more labeled training
examples?

## Method

Day 2's bigram TF-IDF + LinearSVC is used. The training partition is shuffled
with seed 42 once, then prefixes create nested 25%, 50%, and 100% subsets.
Validation examples never change. Each subset fits its own TF-IDF vocabulary
and its own SVM; therefore the experiment measures both more examples and a
larger learned representation.

## Results

At 25% (34,677 rows), accuracy is 0.7542, macro-F1 0.7129, weighted-F1
0.7488. At 50% (69,355 rows), the values are 0.7719, 0.7336, and 0.7677.
At 100% (138,711 rows), they are 0.7868, 0.7516, and 0.7838.

## Interpretation

Performance improves consistently with more training data. The result supports
the practical conclusion that the model is data-hungry, but it does not prove
that every gain comes only from sample count because each subset also learns
its own TF-IDF feature space. The validation set is still the basis for
comparison; no test tuning occurred.

## What someone may ask

**Why use nested subsets?** The 25% examples are contained in 50%, which are
contained in 100%, reducing noise from different samples.

**Why macro-F1?** It prevents large categories from completely dominating the
answer.

**What is missing?** A full per-category error taxonomy and adequate
publisher/category subgroup analysis would be the next extension.
