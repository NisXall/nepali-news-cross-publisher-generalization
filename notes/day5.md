# Day 5 - Data-size experiments

`notebooks/05_day5_training_size.ipynb` uses the verified bigram TF-IDF +
LinearSVC baseline and a deterministic nested sampling order. It keeps the
existing validation partition fixed. A 25% smoke run completed first in about 74 seconds with accuracy 0.7542,
macro-F1 0.7129, and weighted-F1 0.7488. The full nested run then produced:

| Training rows | Accuracy | Macro-F1 | Weighted-F1 | Runtime |
|---:|---:|---:|---:|---:|
| 34,677 (25%) | 0.7542 | 0.7129 | 0.7488 | 76.7 s |
| 69,355 (50%) | 0.7719 | 0.7336 | 0.7677 | 120.1 s |
| 138,711 (100%) | 0.7868 | 0.7516 | 0.7838 | 205.0 s |

The complete table is saved at `results/tables/day5_training_size.csv`.
The increase is descriptive for this nested sampling experiment; it does not
separate training-size effects from the changing learned vocabulary.
