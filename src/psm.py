import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from src.utils import ensure_columns, bootstrap_ci


DEFAULT_COVS = ["age", "income", "prior_purchases", "sessions", "is_mobile"]


def estimate_propensity(df: pd.DataFrame, covariates=DEFAULT_COVS) -> np.ndarray:
    X = df[covariates].copy()
    X = (X - X.mean()) / (X.std() + 1e-9)
    model = LogisticRegression(max_iter=2000)
    model.fit(X, df["T"].values)
    return model.predict_proba(X)[:, 1]


def match_on_propensity(
    df: pd.DataFrame,
    pscore: np.ndarray,
    caliper: float = 0.05,
) -> pd.DataFrame:
    """
    1-to-1 nearest neighbor matching on propensity score, with caliper.
    Returns a matched dataframe with columns: Y_treated, Y_control.
    """
    d = df.copy()
    d["pscore"] = pscore

    treated = d[d["T"] == 1].reset_index(drop=True)
    control = d[d["T"] == 0].reset_index(drop=True)

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(control[["pscore"]].values)

    dist, idx = nn.kneighbors(treated[["pscore"]].values, n_neighbors=1)
    dist = dist.flatten()
    idx = idx.flatten()

    keep = dist <= caliper
    treated_kept = treated.loc[keep].reset_index(drop=True)
    matched_controls = control.loc[idx[keep]].reset_index(drop=True)

    return pd.DataFrame(
        {
            "Y_treated": treated_kept["Y"].values,
            "Y_control": matched_controls["Y"].values,
            "pscore_treated": treated_kept["pscore"].values,
            "pscore_control": matched_controls["pscore"].values,
        }
    )


def psm_ate(df_post: pd.DataFrame, covariates=DEFAULT_COVS, caliper: float = 0.05) -> dict:
    """
    PSM ATE estimate on post-period user-level data.
    """
    ensure_columns(df_post, ["Y", "T"] + list(covariates))

    p = estimate_propensity(df_post, covariates=covariates)
    matched = match_on_propensity(df_post, pscore=p, caliper=caliper)

    if len(matched) == 0:
        raise ValueError("No matches found. Increase caliper or check overlap.")

    # Compute point estimate first
    ate = float(matched["Y_treated"].mean() - matched["Y_control"].mean())

    def stat_fn(sample: pd.DataFrame) -> float:
        p_s = estimate_propensity(sample, covariates=covariates)
        m_s = match_on_propensity(sample, pscore=p_s, caliper=caliper)
        if len(m_s) == 0:
            return np.nan
        return float(m_s["Y_treated"].mean() - m_s["Y_control"].mean())

    # FIX: pass ate as estimate to avoid redundant PSM recomputation
    ci = bootstrap_ci(df_post, stat_fn, n_boot=300, alpha=0.05, estimate=ate)

    return {
        "psm_ate": ate,
        "matched_pairs": int(len(matched)),
        "caliper": float(caliper),
        "bootstrap_ci": ci,
    }
