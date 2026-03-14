import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from utils import ensure_columns


DEFAULT_COVS = ["age", "income", "prior_purchases", "sessions", "is_mobile"]


def _fit_scaler(df: pd.DataFrame, covariates=DEFAULT_COVS):
    """
    FIX: Compute mean/std from the FULL dataset once.
    Returns (mean, std) to be reused for all subsets and predictions,
    ensuring all models operate in the same feature space.
    """
    X = df[covariates].copy()
    return X.mean(), X.std() + 1e-9


def _apply_scaler(df: pd.DataFrame, mean, std, covariates=DEFAULT_COVS) -> pd.DataFrame:
    return (df[covariates].copy() - mean) / std


def t_learner_uplift(df_post: pd.DataFrame, covariates=DEFAULT_COVS) -> pd.DataFrame:
    """
    Train two outcome models:
      model_t: P(Y=1 | X, T=1)
      model_c: P(Y=1 | X, T=0)
    Predict uplift = p_t - p_c for every row.

    FIX: Use a single scaler fit on the full dataset so that treated,
    control, and prediction all share the same feature space.
    Previously each subset was standardized with its own mean/std,
    making the two models incomparable.
    """
    ensure_columns(df_post, ["Y", "T"] + list(covariates))

    # FIX: fit scaler on full dataset, apply consistently everywhere
    mean, std = _fit_scaler(df_post, covariates)

    treated = df_post[df_post["T"] == 1].copy()
    control = df_post[df_post["T"] == 0].copy()

    Xt = _apply_scaler(treated, mean, std, covariates)
    Xc = _apply_scaler(control, mean, std, covariates)
    Xall = _apply_scaler(df_post, mean, std, covariates)

    mt = LogisticRegression(max_iter=2000).fit(Xt, treated["Y"].values)
    mc = LogisticRegression(max_iter=2000).fit(Xc, control["Y"].values)

    p_t = mt.predict_proba(Xall)[:, 1]
    p_c = mc.predict_proba(Xall)[:, 1]
    uplift = p_t - p_c

    out = df_post.copy()
    out["p_treat"] = p_t
    out["p_control"] = p_c
    out["uplift"] = uplift
    return out


def targeting_simulation(df_scored: pd.DataFrame, top_frac: float = 0.3) -> dict:
    """
    Simple business-style simulation:
    - Target top X% by predicted uplift with treatment.
    - Everyone else no treatment.
    Compare expected conversion vs blanket discount.
    """
    ensure_columns(df_scored, ["uplift", "p_treat", "p_control"])

    d = df_scored.sort_values("uplift", ascending=False).reset_index(drop=True)
    k = int(len(d) * top_frac)

    # Targeted: top k get treated, rest untreated
    conv_targeted = np.mean(
        np.concatenate([d.loc[: k - 1, "p_treat"].values, d.loc[k:, "p_control"].values])
    )

    # Blanket discount: everyone treated
    conv_blanket = float(d["p_treat"].mean())

    # No discount baseline
    conv_none = float(d["p_control"].mean())

    return {
        "top_frac": float(top_frac),
        "expected_conv_targeted": float(conv_targeted),
        "expected_conv_blanket": float(conv_blanket),
        "expected_conv_none": float(conv_none),
        "lift_targeted_vs_blanket": float(conv_targeted - conv_blanket),
        "lift_targeted_vs_none": float(conv_targeted - conv_none),
    }
