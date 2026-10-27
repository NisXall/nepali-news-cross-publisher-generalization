# Final research status

## Complete

- Dataset audit, duplicate audit, and fixed split integrity checks.
- Classical TF-IDF baselines and training-size experiment.
- Final classical test evaluation.
- Random versus duplicate-aware comparison.
- Source-held-out classical evaluation for seven eligible publishers.
- Source-text diagnostic and limited stripping ablation.
- Conflicting duplicate review table.
- Sentence-transformer dependency setup, model download, finite-embedding
  smoke test, cached bounded pilot, and linear classifier comparison.
- XLM-R dependency setup, local-cache forward/loss smoke test, token audit,
  and bounded CPU/GPU training pilots.
- Leakage-safe calibration and abstention for the frozen classical model,
  including a saved bundle, reliability tables/figure, and test evaluation.
- Larger CUDA sentence-embedding and XLM-R pilots using the detected RTX 4060
  Laptop GPU.
- Full sentence-embedding benchmark on all fixed partitions, with validation
  selection between Logistic Regression and LinearSVC and one final selected
  model test evaluation.
- A resumable XLM-R condition runner with bounded A/B results and a partial
  Aarthiknews held-out-source result; full-data XLM-R and complete neural LOSO
  remain incomplete because of compute cost.

## Partial

- XLM-R remains a bounded pilot; the sentence-embedding benchmark is now
  complete.
- XLM-R model selection, full fine-tuning, checkpoint reload, and source-held
  out-of-source neural evaluation are not complete.

## Blocked or not run

- Full sentence-embedding encoding over all fixed partitions: complete.
  LinearSVC was selected on validation (macro-F1 0.6253) and achieved test
  accuracy 0.7114, macro-F1 0.6288, and weighted-F1 0.6979.
- Full XLM-R fine-tuning: still not run; a CUDA-enabled RTX 4060 was detected
  and used for a larger 8,192/2,048 pilot.
- Neural calibration and abstention: NOT RUN because no frozen neural model was
  selected. Classical calibration is complete and is not a neural result.
- Character and word-plus-character classical experiments: blocked by resource
  cost as documented in `results/tables/classical_blocked_experiments.json`.

## Important interpretation

The classical LinearSVC model remains the only final frozen benchmark model.
Neural pilot metrics are feasibility evidence, not comparable full-dataset
results. The test partition was not used for calibration fitting or threshold
selection. The calibrated test metrics are a single final evaluation of the
frozen classical bundle.

## Calibration and abstention result

The calibrator is a multiclass logistic regression fitted to validation
decision scores from a LinearSVC trained only on the fixed training partition.
The predeclared policy maximizes validation coverage subject to selective risk
at most 0.20. It selected threshold 0.2932, with validation coverage 0.9831.
On the untouched test partition, coverage was 0.9836, accepted accuracy
0.7942, accepted macro-F1 0.7614, and selective risk 0.2058. Test calibration
was Brier 0.0152, log loss 0.6443, and 10-bin ECE 0.0209. Calibration on
known publishers does not guarantee calibration on unseen publishers or future
data.
