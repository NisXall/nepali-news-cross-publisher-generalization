# Day 1 - Dataset Exploration

## Dataset overview

- Dataset: `Suyogyart/np20ng`
- Original documents: 231,216
- Columns: `source`, `category`, `heading`, `content`
- Categories: 20
- Sources: 10
- Dataset memory footprint during exploration: approximately 1,083.55 MB

## Category distribution

The largest categories were:

- Politics: 31,021 documents (13.42%)
- Society: 27,802 (12.02%)
- Sports: 24,392 (10.55%)
- Economy: 23,653 (10.23%)
- Entertainment: 22,664 (9.80%)

The smallest categories were:

- Employment: 1,931 documents (0.84%)
- Tourism: 3,177 (1.37%)
- Education: 3,709 (1.60%)
- Lifestyle: 3,865 (1.67%)
- Opinion: 3,945 (1.71%)

The largest-to-smallest class ratio is 16.06, indicating substantial class
imbalance.

## Source distribution

- Nepalkhabar: 69,506
- Onlinekhabar: 66,716
- Ekantipur: 29,875
- Aarthiknews: 25,965
- Setopati: 15,541
- Nagarik: 11,108
- Gorkhapatra: 5,059
- Crimenews: 3,921
- Nepalipatra: 1,977
- Educationpati: 1,548

Source and category are strongly associated. Some sources are concentrated in
specific categories, such as Crimenews in Crime, Educationpati in Education,
and Aarthiknews in several finance-related categories. This creates a
potential source-distribution shortcut for classifiers.

## Missing values and text length

- Missing category values: 0
- Missing source values: 0
- Missing content values: 25
- Empty or whitespace-only content: 30
- Missing heading values: 2
- Empty or whitespace-only headings: 2
- Median content length: 177 words
- 95th-percentile content length: 786 words
- Maximum content length: 8,433 words

## Exploratory artifacts

- `results/figures/class_distribution.png`
- `results/figures/document_lengths.png`
- `data/raw/np20ng_raw.parquet`

## Main observations

The dataset is large and multi-source, but it is not balanced across either
categories or sources. The strong source-category relationship should be
considered when designing splits and evaluating generalization. Empty content
must be removed before modeling, while missing headings do not affect the
content-based baseline.

