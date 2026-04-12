# Notebooks

These notebooks implement the project goals: simulated observational A/B under confounding; ATE via naive difference-in-means, regression adjustment, DiD, and PSM; bootstrap CIs and bias vs ground truth; T-learner uplift and targeted discount allocation.

Notebooks assume the project root is on the path (e.g. run from `notebooks/` with `sys.path.append(os.path.abspath(".."))` at the top) so that `from src....` imports work.

---

## `01_eda.ipynb` — Exploratory Data Analysis

Run this **before** causal methods to understand the simulated data and why naive ATE is biased.

**What it does:**

1. **Setup and load** — Imports `SimConfig`, `simulate_discount_data`, `to_user_level_post`, and generates the panel plus post-only DataFrame. Prints panel shape, post-period shape, true ATE, and tau.
2. **Dataset overview** — Data types, missing values, and summary statistics for the post period.
3. **Treatment group sizes** — Counts and bar chart of treated vs control; treatment rate.
4. **Covariate distributions by treatment** — Because treatment is confounded, treated and control differ on covariates (e.g. prior_purchases, sessions). Visualizes this so you see why unadjusted comparison is misleading.
5. **Naive (biased) ATE** — Computes diff-in-means and compares to true ATE to show bias.
6. **Pre/post outcome trends** — How outcomes change from pre to post by group.
7. **Propensity score overlap** — Uses `estimate_propensity` from `src.psm` and plots propensity distributions for treated vs control to check overlap before running PSM.

Use this notebook to build intuition and check that the DGP and overlap assumptions make sense.

---

## `02_causal_methods.ipynb` — ATE estimators and uplift

Runs **all** causal estimators and compares them to ground truth, then does uplift and targeting.

**What it does:**

1. **Setup and simulate** — Same as 01: generate panel and post-only data with `simulate_discount_data(cfg)`; unpack `(df, meta)` and use `meta["ate_true_post"]` as ground truth.
2. **ATE estimators** — Computes and prints:
   - **Naive** — `diff_in_means_ate` (biased baseline).
   - **Regression adjustment** — `regression_ate` with bootstrap CI.
   - **DiD** — `did_ate` on the full panel (effect and SE).
   - **PSM** — `psm_ate` on post-only data with caliper and bootstrap CI.
3. **Bias vs ground truth** — Builds a small table: for each estimator, **ATE**, **Bias** (estimate − true ATE), and **Bias %**. Lets you see which methods reduce bias under confounding.
4. **Uplift and targeting** — Runs `t_learner_uplift` on post-period data, then `targeting_simulation` (e.g. top 30% by uplift). Prints expected conversion under targeted vs blanket vs no discount.

Use this notebook to compare methods and to demonstrate heterogeneous effects and targeted discount allocation.
