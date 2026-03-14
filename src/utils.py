import numpy as np
import pandas as pd
from typing import Callable, Dict, Optional


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def bootstrap_ci(
    df: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    estimate: Optional[float] = None,  # FIX: accept pre-computed estimate
) -> Dict[str, float]:
    """
    Nonparametric bootstrap CI for an estimator defined by stat_fn.
    Returns dict with estimate, ci_low, ci_high.

    FIX: Added optional `estimate` parameter. If the caller already computed
    the point estimate (e.g. psm_ate), pass it here to avoid a redundant
    and potentially expensive extra call to stat_fn on the full dataset.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    stats = []
    for _ in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        sample = df.iloc[sample_idx]
        stats.append(stat_fn(sample))
    stats = np.array(stats)

    # FIX: skip redundant stat_fn call if estimate was provided
    est = estimate if estimate is not None else stat_fn(df)

    low = np.quantile(stats, alpha / 2)
    high = np.quantile(stats, 1 - alpha / 2)
    return {"estimate": float(est), "ci_low": float(low), "ci_high": float(high)}


def diff_in_means_ate(df: pd.DataFrame, outcome_col: str, treat_col: str) -> float:
    treated = df[df[treat_col] == 1][outcome_col].mean()
    control = df[df[treat_col] == 0][outcome_col].mean()
    return float(treated - control)


def ensure_columns(df: pd.DataFrame, cols: list) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
