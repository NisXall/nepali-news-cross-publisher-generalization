# Day 3 - Trustworthy evaluation

## What was verified

- The repository has fixed train/validation/test partitions of 138,711 /
  46,237 / 46,238 rows.
- The partitions contain the same `source`, `category`, `heading`, `content`,
  and `normalized_content` schema plus `original_index`.
- The existing partitions use the documented duplicate-aware grouped protocol.
  The audit found zero normalized-content groups crossing partitions.
- All 20 categories are represented in each partition.

## Audits and outputs

`src/day3_evaluation.py` writes:

- `results/tables/day3_split_metadata.json`
- `results/tables/day3_duplicate_group_audit.csv`
- `results/tables/day3_category_distribution.csv`
- `results/tables/day3_source_distribution.csv`
- `results/tables/day3_source_category_proportions.csv`
- `results/tables/day3_source_diagnostic.json`
- `results/figures/day3_category_distribution.png`

The script also creates a descriptive ordinary stratified random split with
seed 42 (then 43 for its second split) to show how many duplicate groups would
cross partitions. This is an audit comparison, not a replacement for the
grouped partitions used by the benchmark.

The source-only majority-category calculation is a diagnostic for source and
category association, not a text model. Source-held-out candidates are chosen
using document count and category-coverage rules before model scores are
considered. No unseen-source generalization score is claimed on Day 3.

## Validation

The Day 3 tests assert both duplicate-group isolation and disjoint
`original_index` values. Run:

```text
python src/day3_evaluation.py
python -m pytest tests/test_day3_splits.py
```

## Limitations

The grouped split and random split have different source and category
distributions, so any score difference cannot be attributed to duplicate
removal alone. Source and category are strongly associated in this dataset;
therefore a source-held-out drop would measure distribution shift and label
coverage together, not publisher writing style in isolation.
