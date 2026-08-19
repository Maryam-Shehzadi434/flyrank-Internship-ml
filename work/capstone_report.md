# Predicting Content Decline Using Search Signals

**Author:** Maryam Shehzadi
**Lane:** Lane 1 — Content Refresh Prediction
**Repo:** https://github.com/Maryam-Shehzadi434/flyrank-Internship-ml
**Date:** 19 August 2026

\---

## Abstract

Can we predict which pages are declining using observable search signals? I analyzed 176,738 pages from the March 2026 FlyRank warehouse, using 6 features (impressions, CTR, position, age, clicks, word\_count). A Random Forest model with client-holdout validation achieved 1.000 Precision@50, beating the hand-written baseline by 4.2 times. This work helps content teams prioritize which pages to review first, saving time and improving search performance.

\---

## 1\. Problem Framing

**Decision:** Which pages should content teams prioritize for review, refresh, or update?

**Unit of analysis:** One page (content\_hash\_id)

**Output:** Ranked list of pages with reason codes:

* high\_impressions\_low\_ctr — Refresh content
* stale\_visible\_page — Update content
* position\_opportunity — Improve SEO
* other — Monitor only

**Action:** Content managers review the top 50 predicted declining pages and decide whether to refresh, rewrite, remove, or monitor.

**Cost of a wrong call:**

* False positive: Wastes time refreshing content that didn't need it
* False negative: Lost traffic and revenue as the page continues to decline
* Ranking error: Delayed action lets the decline worsen

**Why does ML help here?** Simple hand-written rules cannot capture the complex interactions between features (impressions, CTR, position, age). ML automatically learns these interactions and ranks pages more effectively — as shown by the 4.2 times improvement over the baseline.

\---

## 2\. Data Safety

**Data Used:**

* Table: fact\_content\_daily\_performance (March 2026, 9.84 million rows)
* Table: dim\_content (content metadata)
* Development slice: month=2026-03 (mid-panel month, avoids leakage from June 2026)
* Feature vector: 176,738 pages x 14 columns

**Columns Excluded:**

* trend\_pct, trend\_direction: Leakage (derived from the label)
* client\_hash\_id, content\_hash\_id: Context only — used for grouping, never as features
* Product decision flags: Not in data by design

**Client-Identifying Data:** None appears anywhere. All IDs are pseudonymized.

\---

## 3\. Baseline

**Baseline Score:** score = impressions\_90d x (1 - ctr\_90d) x (content\_age\_days / 365)

**Baseline Performance:**

* Precision@50: 0.240
* Base Rate: 0.396

\---

## 4\. Model / Analysis

**Method:** Random Forest Classifier (n\_estimators=200, max\_depth=10, random\_state=42)

**Feature List (6):**

* impressions\_90d: Total search impressions
* clicks\_90d: Total search clicks
* avg\_position\_90d: Average search position
* ctr\_90d: Click-through rate
* content\_age\_days: Days since content creation
* word\_count: Number of words

**Target Definition:** is\_declining = impressions\_90d less than median\_impressions

\---

## 5\. Evaluation

**Split:** Client-holdout (37 train clients, 10 test clients)

**Metric:** Precision@50

**Model vs Baseline:**

* Baseline (Hand Rule): Precision@50 = 0.240, Base Rate = 0.396
* Random Forest: Precision@50 = 1.000, Base Rate = 0.396, Improvement = 4.2 times

**Error Analysis:**

* False positives: Pages with high impressions but CTR normal for their category
* False negatives: Pages with low impressions but high CTR
* The label is simple (median-based); real-world decline is more nuanced

\---

## 6\. Interpretation

**Feature Importance:**

* impressions\_90d: 73.8%
* ctr\_90d: 12.4%
* clicks\_90d: 10.5%
* word\_count: 2.2%
* avg\_position\_90d: 0.6%
* content\_age\_days: 0.5%

**Key Insight:** impressions\_90d is the strongest predictor. content\_age\_days was expected to be stronger but is the weakest feature.

\---

## 7\. Recommendation

**Top 10 Actions:**

|Rank|Action|Reason Code|Impressions|
|-|-|-|-|
|1|Refresh content|high\_impressions\_low\_ctr|617,124|
|2|Refresh content|high\_impressions\_low\_ctr|245,276|
|3|Refresh content|high\_impressions\_low\_ctr|221,310|
|4|Refresh content|high\_impressions\_low\_ctr|203,497|
|5|Refresh content|high\_impressions\_low\_ctr|186,983|
|6|Monitor|other|205,045|
|7|Refresh content|high\_impressions\_low\_ctr|151,166|
|8|Refresh content|high\_impressions\_low\_ctr|164,885|
|9|Refresh content|high\_impressions\_low\_ctr|142,304|
|10|Refresh content|high\_impressions\_low\_ctr|244,931|

**Action Distribution (n=176,738):**

|Action|Count|Percent|
|-|-|-|
|Monitor (other)|116,668|66.0%|
|Refresh content (high\_impressions\_low\_ctr)|59,393|33.6%|
|Improve SEO (position\_opportunity)|475|0.3%|
|Update content (stale\_visible\_page)|202|0.1%|

\---

## 8\. Reproducibility

**Commands:**

git clone https://github.com/flyrank-bih/flyrank-ml-internship-starter
cd flyrank-ml-internship-starter
pip install -r requirements.txt
python scripts/run\_all.py

**Random Seeds:** random\_state=42

**Environment:** Python 3.12, Colab

\---

## 9\. Acknowledgments and Data Credit

Built on the FlyRank ML Internship dataset (https://flyrank.ai).

Thanks to the FlyRank team for providing the data and guidance throughout the internship.

