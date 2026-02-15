# MSS Module Reproduction Guide

This folder contains the **MSS (Minimal Sufficient Set)** core pipeline for medical card selection.  
The current main logic uses **topic-structured redundancy control** (specialty + cluster), rather than relying on embedding-cosine redundancy as the primary decision signal.

---

## 1) What This MSS Module Includes

Core scripts (recommended):
- `mss_topic_diversity.py`  
  Topic-aware global greedy selection with saturation penalty
- `mss_full_pool_run.py`  
  Stratified two-stage selection (quota stage + global stage)
- `mss_metric_discriminability.py`  
  Metric discriminability analysis (to justify metric choices)

Optional supporting scripts:
- `mss.py` (baseline method comparison)
- `mss_softprompt_projection.py` (projection-space comparison)

Typical input:
- `deepseek_cards.json`

Typical outputs:
- `mss_topic_cards_bestscore.json`
- `mss_topic_results_bestscore.json`
- `mss_stratified_cards_bestscore.json`
- `mss_stratified_results_bestscore.json`
- `mss_metric_discriminability_report_v2.json`

---

## 2) Main MSS Strategies

### A. Topic-aware line (`mss_topic_diversity.py`)
- Single-stage global selection
- Uses structured saturation penalty:
  - specialty saturation
  - cluster saturation

### B. Stratified line (`mss_full_pool_run.py`)
- Two-stage selection:
  1. Specialty quota stage (coverage guarantee)
  2. Global free-competition stage
- Same structured principle for redundancy control

---

## 3) Unified Selection Score (Primary)

The final configuration is selected by:

\[
\text{selection\_score}
=
w_{tv}(1-\text{TV\_to\_pool})
+
w_{eff}\,\text{EffectiveSpecRatio}
+
w_{base}\,\overline{B}
\]

Where:
- `TV_to_pool` = distribution gap between MSS specialty distribution and full-pool distribution (lower is better)
- `EffectiveSpecRatio` = effective specialty diversity ratio (higher is better)
- `avg_base_score` (\(\overline{B}\)) = average card utility score (higher is better)

Default weights:
- `w_tv = 0.45`
- `w_eff = 0.35`
- `w_base = 0.20`

> Embedding-based coverage can still be reported as an auxiliary diagnostic, but it is not the primary optimization target.

---

## 4) Environment Setup

Recommended: Python 3.10+
