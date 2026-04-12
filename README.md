# Causal discount project

Does giving a customer a discount actually cause them to buy more?

The tricky part is that discounts aren't given randomly — marketing tends to target high-intent customers who were probably going to buy anyway. So if you just compare "people who got a discount" vs "people who didn't", the discount group looks better, but that's because they were already more likely to buy — not because the discount worked.

So you build a simulation where you know the true answer, then show that:

- **Naive comparison** — gives the wrong (biased) answer
- **DiD, PSM, Uplift modeling** — give answers much closer to the truth

---

## What this project does

- Simulated an observational A/B experiment to estimate the causal effect of discounts on customer purchase probability under confounding conditions.
- Estimated Average Treatment Effect (ATE) using naive difference-in-means, regression adjustment, Difference-in-Differences, and Propensity Score Matching.
- Constructed bootstrap confidence intervals and analyzed estimator bias relative to known ground truth.
- Developed uplift modeling approach (T-learner) to estimate heterogeneous treatment effects and optimize targeted discount allocation strategy.

**Implementation notes:** DiD uses covariate adjustment and clustered standard errors by user; PSM uses caliper matching; regression adjustment and PSM use bootstrap CIs. See `src/` and `notebooks/02_causal_methods.ipynb` for code.

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
