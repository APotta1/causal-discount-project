import pandas as pd
import statsmodels.formula.api as smf
from utils import ensure_columns


COVARIATES = ["age", "income", "prior_purchases", "sessions", "is_mobile"]


def did_ate(df: pd.DataFrame) -> dict:
    """
    Difference-in-Differences via OLS:
      Y ~ T + post + T:post + covariates

    FIX: Added covariate adjustment. The DGP has strong confounding
    (treatment assignment depends on prior_purchases, sessions, etc.)
    so the unconditional parallel trends assumption does not hold.
    Including covariates makes parallel trends conditional on X,
    which is valid here since the same X drives both treatment and outcome.

    Returns effect estimate for interaction term (T:post) + standard error.
    """
    ensure_columns(df, ["Y", "T", "post", "user_id"] + COVARIATES)

    # FIX: Added covariates to formula
    formula = "Y ~ T + post + T:post + " + " + ".join(COVARIATES)

    # Clustered SE by user since panel (pre/post per user)
    model = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["user_id"]}
    )

    coef = model.params.get("T:post", float("nan"))
    se = model.bse.get("T:post", float("nan"))
    ci_low, ci_high = model.conf_int().loc["T:post"].tolist()

    return {
        "did_effect": float(coef),
        "std_err": float(se),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_obs": int(model.nobs),
    }
