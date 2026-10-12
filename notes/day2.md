# Day 2 - Dataset Audit and Classical Baselines

## Dataset audit

- Number of original documents: 231,216
- Number of usable documents: 231,186
- Number of categories: 20
- Number of sources: 10
- Missing/empty content: 30 empty or whitespace-only documents; 25 content values were null

Rows with missing or empty content were removed. There were no missing category
or source values. Two heading values were missing, but headings were not used
as the modeling text.

## Duplicate investigation

- Number of duplicate groups: 246 normalized-content groups containing 601 rows
- Number of conflicting-label groups: 14
- Duplicate handling strategy: normalize content, identify duplicate groups, and use normalized content as the grouping key for stratified group splits. Duplicate rows were not independently spread across train, validation, and test. Conflicting-label groups were retained for audit and not silently relabeled.

There were 232 duplicate groups with one category and 14 groups containing
multiple categories.

## Evaluation protocol

- Train/validation/test sizes: 138,711 / 46,237 / 46,238
- Split strategy: a 5-fold `StratifiedGroupKFold` outer split followed by a 4-fold `StratifiedGroupKFold` inner split
- Random seeds: 42 for the outer split and 43 for the inner split
- Duplicate overlap checks: passed; normalized-content groups were disjoint across all three partitions

All 20 categories were present in each partition.

## Baseline results

- Unigram Logistic Regression: accuracy 0.7429; macro-F1 0.7028; weighted-F1 0.7396
- Bigram Logistic Regression: accuracy 0.7678; macro-F1 0.7243; weighted-F1 0.7639
- Bigram Linear SVM: accuracy 0.7868; macro-F1 0.7516; weighted-F1 0.7838
- Best validation macro-F1: 0.7516 from the bigram Linear SVM

The Logistic Regression models are retained as comparison baselines. The
Linear SVM is the model used for the main error analysis.

The unigram model used 32,113 features. The bigram models used 300,000 TF-IDF
features with `min_df=2`, `sublinear_tf=True`, and `C=1.0`.

## Error analysis

- Lowest-F1 categories: Business (0.56), Tourism (0.62), Employment (0.63), Economy (0.63), and Arts (0.65)
- Most confused categories: Business -> Economy (510), Arts -> Entertainment (395), Economy -> Politics (326), Economy -> Business (306), and Society -> Politics (285)
- Common error patterns: semantically adjacent categories overlap, especially business/economy/bank, arts/entertainment, society/politics/crime, and technology/automobile. Source concentration and category imbalance are additional likely contributors.
- High-margin incorrect predictions: the highest-margin examples included Economy -> Politics, Economy -> Politics, Economy -> Politics, Automobile -> Technology, and Society -> Politics. The largest recorded incorrect margin was 3.6878.

The SVM made 9,860 incorrect predictions out of 46,237 validation examples,
for an error rate of 0.2132.

## Source generalization

- Candidate held-out sources: Nepalkhabar and Onlinekhabar are the largest sources; Ekantipur, Aarthiknews, Setopati, and Nagarik are additional candidates with substantial volume
- Category coverage: coverage is uneven by source; some categories are absent or nearly absent for particular sources
- Feasibility and limitations: a source-held-out evaluation is feasible for sources with enough examples and reasonable category coverage, but it will measure both domain shift and label-coverage differences. Sources such as Crimenews and Educationpati are highly concentrated in one category and are poor standalone held-out choices for a full 20-class evaluation.

## Next steps

- What needs improvement: improve performance on minority and overlapping categories, verify label quality in conflicting duplicate groups, and measure performance under source-held-out splits
- Experiments for Day 3: tune Linear SVM regularization and TF-IDF settings, compare class weighting, run source-held-out evaluation, inspect high-margin errors manually, and evaluate duplicate/conflict cleanup strategies

