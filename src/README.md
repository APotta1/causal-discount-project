# Source code (`src/`)

This package implements the causal pipeline: simulated observational A/B under confounding; ATE estimation (naive difference-in-means, regression adjustment, DiD, PSM); bootstrap CIs and bias vs known ground truth; T-learner uplift for heterogeneous effects and targeted discount allocation. Each file is described below.

---

## `__init__.py`

Makes `src` a Python package so you can run `from src.utils import ...` and `import src.did` from the project root. It is intentionally minimal (no exports).

---

## `utils.py`

Shared helpers used across the other modules.

- **`sigmoid(x)`** — Maps real numbers to (0, 1); used in the data-generating process for binary outcomes.
- **`set_seed(seed)`** — Sets `numpy`’s random seed for reproducibility.
- **`bootstrap_ci(df, stat_fn, n_boot, alpha, estimate=None)`** — Builds a nonparametric bootstrap confidence interval for any estimator defined by `stat_fn(df) -> float`. Resamples rows, recomputes the statistic, and returns `estimate`, `ci_low`, `ci_high`. The optional `estimate` avoids recomputing the point estimate when the caller already has it (e.g. PSM, regression ATE).
- **`diff_in_means_ate(df, outcome_col, treat_col)`** — Naive ATE: mean outcome in treated minus mean outcome in control. Used as a biased baseline under confounding.
- **`ensure_columns(df, cols)`** — Validates that the DataFrame has the required columns; raises `ValueError` with the list of missing names if not.

---

## `simulate_data.py`

Data-generating process (DGP) for the observational A/B experiment.

- **`SimConfig`** — Dataclass for simulation settings: `n_users`, `seed`, treatment effect `tau` (log-odds), confounding strength, baseline difficulty, and time lift from pre to post.
- **`simulate_discount_data(cfg)`** — Returns `(df, meta)`:
  - **`df`**: Panel data with two rows per user (pre and post). Columns include `user_id`, `post`, `T` (treatment), `Y` (purchase), and covariates (`age`, `income`, `prior_purchases`, `sessions`, `is_mobile`). Treatment assignment is **confounded** by these covariates (e.g. high-intent users more likely to get the discount).
  - **`meta`**: Dict with `ate_true_post`, `tau_log_odds`, `n_users` for ground-truth comparison.
- **`to_user_level_post(df)`** — Returns one row per user for the post period only; used as input for PSM, regression ATE, and uplift.

When run as a script (`python -m src.simulate_data`), it writes `data_simulated.csv` to the current directory.

---

## `regression_ate.py`

**Regression adjustment** (outcome regression) for ATE estimation.

- **`regression_ate(df, covariates, n_boot, alpha, seed)`** — Fits a single outcome model \(E[Y \mid X, T]\) (logistic regression with treatment and covariates). Predicts under \(T=1\) and \(T=0\) for every unit and sets ATE = mean of (predicted Y1 − predicted Y0). Returns the point estimate and a **bootstrap confidence interval**. Use this when you want to adjust for covariates via modeling the outcome.

---

## `did.py`

**Difference-in-Differences (DiD)** for the panel (pre/post) design.

- **`did_ate(df)`** — Fits OLS: `Y ~ T + post + T:post + covariates`. The coefficient on `T:post` is the DiD treatment effect. Standard errors are **clustered by `user_id`** because each user appears in both periods. Returns `did_effect`, `std_err`, `ci_low`, `ci_high`, and `n_obs`. Assumes parallel trends conditional on the included covariates.

---

## `psm.py`

**Propensity Score Matching** for ATE on post-period, cross-sectional data.

- **`estimate_propensity(df, covariates)`** — Fits a logistic regression of `T` on covariates (standardized) and returns the propensity score for each row.
- **`match_on_propensity(df, pscore, caliper)`** — 1-to-1 nearest-neighbor matching on the propensity score, dropping pairs whose distance exceeds the caliper. Returns a DataFrame with `Y_treated`, `Y_control`, and the two propensity scores per pair.
- **`psm_ate(df_post, covariates, caliper)`** — Runs propensity estimation and matching, then computes ATE as the mean difference in outcomes in the matched pairs. Uses **bootstrap** for the confidence interval. Raises if no matches are found (e.g. caliper too small).

---

## `uplift.py`

**Uplift modeling** (heterogeneous treatment effects) and **targeted discount** simulation.

- **`t_learner_uplift(df_post, covariates)`** — T-learner: fits separate outcome models for treated and control (logistic regression). A single scaler is fit on the full dataset so both models use the same feature space. For each row, predicts \(P(Y=1 \mid X, T=1)\) and \(P(Y=1 \mid X, T=0)\) and sets **uplift** = predicted treat − predicted control. Adds columns `p_treat`, `p_control`, `uplift` to the DataFrame.
- **`targeting_simulation(df_scored, top_frac)`** — Business simulation: treat the top `top_frac` (e.g. 30%) by predicted uplift, leave the rest untreated. Compares expected conversion under this **targeted** strategy vs **blanket** discount (everyone treated) vs **no discount**. Returns a dict with expected conversion and lift for each scenario.

---

## Summary

| File               | Role                                                                 |
|--------------------|----------------------------------------------------------------------|
| `__init__.py`      | Package marker                                                       |
| `utils.py`         | Bootstrap CI, diff-in-means, sigmoid, set_seed, ensure_columns       |
| `simulate_data.py` | DGP: panel data + metadata (true ATE)                                |
| `regression_ate.py`| Regression-adjusted ATE + bootstrap CI                               |
| `did.py`           | DiD on panel with covariates and clustered SEs                       |
| `psm.py`           | Propensity score estimation, matching, PSM ATE + bootstrap CI        |
| `uplift.py`        | T-learner uplift and targeted discount simulation                    |
