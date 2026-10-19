# Day 4 - Pretrained sentence-embedding baseline

Day 4 was not run as a full experiment. The project `.venv` does not have the
`sentence-transformers` package, and no verified local sentence-embedding model
cache was found. Generating embeddings would therefore require installing a
dependency and downloading a model.

`src/day4_embeddings.py` and `notebooks/04_day4_embeddings.ipynb` now provide
the shared feasibility check and train-only cached-baseline implementation.
No embedding metrics, predictions, or claims of improvement were created.
The existing TF-IDF comparison remains the only verified baseline.

The next safe step is to add a pinned embedding dependency, verify a
multilingual model with Nepali support (for example,
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`), run a small
smoke test, and only then encode the fixed training and validation partitions.
The classifier must be fit on training embeddings only; the untouched test
partition should be used only after model selection.
