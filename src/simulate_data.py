import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from utils import sigmoid, set_seed  # FIX 1: was "from src.utils import ..."


@dataclass
class SimConfig:
    n_users: int = 8000
    seed: int = 42

    # Treatment strength (true causal effect on log-odds in post period)
    tau: float = 0.55

    # Confounding strength: how much X drives treatment assignment
    confounding: float = 1.0

    # Baseline purchase difficulty
    base: float = -2.0

    # Time effect from pre -> post (global lift)
    time_lift: float = 0.25


def simulate_discount_data(cfg: SimConfig) -> Tuple[pd.DataFrame, Dict]:
    """
    Returns a user-level panel dataset with two rows per user:
      - pre period (post=0)
      - post period (post=1)

    Columns:
      user_id, post, T, Y, plus covariates X...
      true_potential_y0_post, true_potential_y1_post (for ground truth ATE)

    FIX 2: Also returns a metadata dict instead of storing in df.attrs,
    which is silently dropped by pd.concat and most DataFrame operations.
    """
    set_seed(cfg.seed)
    n = cfg.n_users

    # Covariates (confounders)
    age = np.clip(np.random.normal(32, 10, n), 18, 70)
    income = np.clip(np.random.lognormal(mean=10.4, sigma=0.35, size=n), 20000, 250000)
    prior_purchases = np.random.poisson(lam=1.2, size=n)
    sessions = np.random.poisson(lam=6.0, size=n)
    is_mobile = np.random.binomial(1, 0.55, size=n)

    # Standardize some features for stable logits
    inc_z = (income - income.mean()) / income.std()
    age_z = (age - age.mean()) / age.std()
    prior_z = (prior_purchases - prior_purchases.mean()) / (prior_purchases.std() + 1e-9)
    sess_z = (sessions - sessions.mean()) / (sessions.std() + 1e-9)

    # Treatment assignment is CONFOUNDED: depends on covariates
    # (marketing targets high-intent users with discounts)
    treat_logit = (
        -0.2
        + cfg.confounding * (0.9 * prior_z + 0.7 * sess_z + 0.4 * inc_z - 0.25 * age_z)
        + 0.25 * is_mobile
    )
    p_treat = sigmoid(treat_logit)
    T = np.random.binomial(1, p_treat, n)

    # Baseline purchase propensity (shared across time, affected by X)
    base_logit = (
        cfg.base
        + 0.9 * prior_z
        + 0.7 * sess_z
        + 0.25 * inc_z
        - 0.1 * age_z
        + 0.15 * is_mobile
    )

    # Pre-period outcome (no discount effect in pre)
    p_pre = sigmoid(base_logit)
    y_pre = np.random.binomial(1, p_pre, n)

    # Post-period potential outcomes
    # y0_post: if no discount
    p_y0_post = sigmoid(base_logit + cfg.time_lift)
    # y1_post: if discount, add true causal effect tau
    p_y1_post = sigmoid(base_logit + cfg.time_lift + cfg.tau)

    # Observed post outcome given treatment assignment
    p_post_obs = np.where(T == 1, p_y1_post, p_y0_post)
    y_post = np.random.binomial(1, p_post_obs, n)

    shared_cols = {
        "user_id": np.arange(n),
        "T": T,
        "age": age,
        "income": income,
        "prior_purchases": prior_purchases,
        "sessions": sessions,
        "is_mobile": is_mobile,
        "p_y0_post": p_y0_post,
        "p_y1_post": p_y1_post,
    }

    # FIX 2: Build pre and post explicitly rather than copying and patching
    df_pre = pd.DataFrame({"post": 0, "Y": y_pre, **shared_cols})
    df_post = pd.DataFrame({"post": 1, "Y": y_post, **shared_cols})

    df = pd.concat([df_pre, df_post], ignore_index=True)

    # FIX 2: Return metadata separately — df.attrs is dropped by pd.concat
    meta = {
        "ate_true_post": float(np.mean(p_y1_post - p_y0_post)),
        "tau_log_odds": float(cfg.tau),
        "n_users": int(cfg.n_users),
    }

    return df, meta


def to_user_level_post(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per user for post period only (good for PSM/uplift)."""
    return df[df["post"] == 1].reset_index(drop=True)


if __name__ == "__main__":
    cfg = SimConfig()
    df, meta = simulate_discount_data(cfg)
    out_path = "data_simulated.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path}")
    print(f"True ATE (post) = {meta['ate_true_post']:.4f}")
