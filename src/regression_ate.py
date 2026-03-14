"""
Regression adjustment (outcome regression) for ATE estimation.
Fits E[Y|X,T], then estimates ATE as mean( E[Y|X,T=1] - E[Y|X,T=0] ).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.utils import ensure_columns, bootstrap_ci


DEFAULT_COVS = ["age", "income", "prior_purchases", "sessions", "is_mobile"]


def regression_ate(
    df: pd.DataFrame,
    covariates: list = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Estimate ATE by regression adjustment (outcome regression).
    Fits a single model E[Y|X,T] with T and covariates, then
    ATE = mean over sample of (E[Y|X,T=1] - E[Y|X,T=0]).

    Returns point estimate and bootstrap confidence interval.
    """
    covariates = covariates or DEFAULT_COVS
    ensure_columns(df, ["Y", "T"] + list(covariates))

    X = df[covariates].copy()
    X = (X - X.mean()) / (X.std() + 1e-9)
    X_with_t = pd.DataFrame(X, index=df.index).assign(T=df["T"].values)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_with_t, df["Y"].values)

    # Predict under T=1 and T=0 for everyone
    X1 = X_with_t.copy()
    X1["T"] = 1
    X0 = X_with_t.copy()
    X0["T"] = 0
    mu1 = model.predict_proba(X1)[:, 1]
    mu0 = model.predict_proba(X0)[:, 1]
    ate = float(np.mean(mu1 - mu0))

    def stat_fn(sample: pd.DataFrame) -> float:
        X_s = sample[covariates].copy()
        X_s = (X_s - X_s.mean()) / (X_s.std() + 1e-9)
        X_s = pd.DataFrame(X_s, index=sample.index).assign(T=sample["T"].values)
        m = LogisticRegression(max_iter=2000)
        m.fit(X_s, sample["Y"].values)
        X1_s = X_s.copy()
        X1_s["T"] = 1
        X0_s = X_s.copy()
        X0_s["T"] = 0
        return float(np.mean(m.predict_proba(X1_s)[:, 1] - m.predict_proba(X0_s)[:, 1]))

    ci = bootstrap_ci(df, stat_fn, n_boot=n_boot, alpha=alpha, seed=seed, estimate=ate)

    return {
        "regression_ate": ate,
        "std_err": (ci["ci_high"] - ci["ci_low"]) / (2 * 1.96) if n_boot else None,
        "bootstrap_ci": ci,
    }
