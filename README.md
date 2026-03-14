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

## File-by-file

Every file is documented in place:

| Location | What’s documented |
|----------|--------------------|
| **[src/README.md](src/README.md)** | Each Python module: `__init__.py`, `utils.py`, `simulate_data.py`, `regression_ate.py`, `did.py`, `psm.py`, `uplift.py` |
| **[notebooks/README.md](notebooks/README.md)** | Each notebook: `01_eda.ipynb`, `02_causal_methods.ipynb` |

**Root-level files:**

- **`README.md`** (this file) — Project overview, structure, and how to run.
- **`requirements.txt`** — Python dependencies (e.g. `numpy`, `pandas`, `scikit-learn`, `statsmodels`). Install with `pip install -r requirements.txt` if you use a fresh environment.

## Run

From project root:

```bash
python -m src.simulate_data   # writes data_simulated.csv
```

Then open the notebooks (run from `notebooks/` or with path set to parent so `from src...` works).
