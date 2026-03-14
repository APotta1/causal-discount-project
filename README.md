# Causal discount project

Observational A/B experiment to estimate the causal effect of discounts on customer purchase probability under confounding.

## Overview

- **Simulated** an observational A/B experiment (discount vs no discount) with confounded treatment assignment.
- **ATE estimation** via:
  - Naive difference-in-means
  - Regression adjustment (outcome regression)
  - Difference-in-Differences (with covariate adjustment and clustered SEs)
  - Propensity Score Matching (with caliper and bootstrap CIs)
- **Bootstrap confidence intervals** for regression-adjusted and PSM estimators; bias analysis vs known ground truth.
- **Uplift modeling** (T-learner) for heterogeneous treatment effects and **targeted discount allocation** (top-X% by predicted uplift).

## Structure

- `src/simulate_data.py` — DGP and panel data simulation
- `src/utils.py` — bootstrap CI, diff-in-means, helpers
- `src/regression_ate.py` — regression-adjusted ATE
- `src/did.py` — Difference-in-Differences
- `src/psm.py` — propensity score matching
- `src/uplift.py` — T-learner uplift and targeting simulation
- `notebooks/01_eda.ipynb` — exploratory analysis
- `notebooks/02_causal_methods.ipynb` — all estimators and bias comparison

## Run

From project root:

```bash
python -m src.simulate_data   # writes data_simulated.csv
```

Then open the notebooks (run from `notebooks/` or with path set to parent so `from src...` works).
