# Day 6 - Transformer preparation

`src/day6_transformer_setup.py` and
`notebooks/06_day6_transformer_setup.ipynb` record the selected environment
and run a guarded, local-cache-only XLM-RoBERTa smoke test. In the current
project `.venv`, `transformers` and PyTorch are absent and CUDA is therefore
unavailable. No training run or transformer metric is claimed.
