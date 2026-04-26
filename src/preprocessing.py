"""
preprocessing.py
================
Data preprocessing pipeline for the tract-level opportunity dataset.

Implements two substantive data-quality fixes (each evaluated for impact in
`notebooks/02_preprocessing_impact.ipynb`):

  1. Missing-value imputation via an iterative / KNN strategy (rather than
     mean-fill), chosen because the missingness in Opportunity Atlas columns
     like `gsmn_math_pcst` (school quality) is correlated with poverty and
     urbanicity -- mean imputation would introduce bias.

  2. Outlier winsorization at the 1st/99th percentile for heavy-tailed
     features (income, density, rent). Chetty et al. (2020) document the
     long right tail of these features; unwinsorized values produce unstable
     leverage in linear models.

It also constructs engineered features (interaction terms, composite indices,
log-transforms) and applies feature standardization.

AI USAGE: Initial scaffolding drafted with Claude (Anthropic), then adapted.
See ATTRIBUTION.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column groupings
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "poor_share", "share_black", "share_hisp", "share_asian", "share_white",
    "hhinc_mean", "mean_commutetime", "frac_coll_plus", "foreign_share",
    "med_hhinc", "popdensity", "rent_twobed", "singleparent_share",
    "traveltime15_share", "emp_rate", "job_density", "gsmn_math_pcst",
    "nonwhite_share2010",
]

HEAVY_TAILED_FEATURES = [
    "hhinc_mean", "med_hhinc", "popdensity", "rent_twobed", "job_density",
]

OUTCOME_COLUMNS = ["kfr_pooled_pooled_p25", "jail_pooled_pooled_p25"]

ID_COLUMNS = ["state", "county", "tract"]

@dataclass
#AI assisted scaffolding
class PreprocessingConfig:
    """Configuration for the preprocessing pipeline."""
    winsorize: bool = True
    winsor_quantile: float = 0.01
    imputation_strategy: str = "iterative"   # "iterative" | "mean" | "median"
    add_engineered_features: bool = True
    standardize: bool = True


@dataclass
#AI assisted scaffolding
class PreprocessingArtifacts:
    """Fitted preprocessing state so test data can be transformed identically."""
    winsor_bounds: dict = field(default_factory=dict)
    imputer: Optional[object] = None
    scaler: Optional[StandardScaler] = None
    feature_names: list = field(default_factory=list)
    # Z-score parameters for the concentrated-disadvantage index. Storing
    # these (instead of recomputing from the input batch) lets us transform
    # a single row at serving time correctly.
    z_params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data quality step 1: winsorization
# ---------------------------------------------------------------------------
#AI assisted scaffolding
def winsorize_fit_transform(
    df: pd.DataFrame,
    columns: list,
    lower_q: float,
    artifacts: PreprocessingArtifacts,
) -> pd.DataFrame:
    """Clip values to [q, 1-q] quantile bounds. Fits bounds on training data."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        lo = df[col].quantile(lower_q)
        hi = df[col].quantile(1 - lower_q)
        artifacts.winsor_bounds[col] = (lo, hi)
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df

#AI assisted scaffolding
def winsorize_transform(df: pd.DataFrame, artifacts: PreprocessingArtifacts) -> pd.DataFrame:
    df = df.copy()
    for col, (lo, hi) in artifacts.winsor_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df


# ---------------------------------------------------------------------------
# Data quality step 2: missing-value imputation
# ---------------------------------------------------------------------------

def fit_imputer(
    df: pd.DataFrame,
    strategy: str,
    feature_cols: list,
) -> object:
    """Fit an imputer on the numeric feature columns."""
    X = df[feature_cols].values
    if strategy == "iterative":
        imputer = IterativeImputer(max_iter=10, random_state=2025)
    elif strategy == "mean":
        imputer = SimpleImputer(strategy="mean")
    elif strategy == "median":
        imputer = SimpleImputer(strategy="median")
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")
    imputer.fit(X)
    return imputer


def apply_imputer(
    df: pd.DataFrame,
    imputer: object,
    feature_cols: list,
) -> pd.DataFrame:
    df = df.copy()
    df[feature_cols] = imputer.transform(df[feature_cols].values)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
#AI assisted scaffolding
def add_engineered_features(
    df: pd.DataFrame,
    z_params: Optional[dict] = None,
) -> tuple:
    """
    Create derived features that encode known sociological constructs:

      - `log_med_hhinc`, `log_popdensity`: tame heavy right tails
      - `inequality_index`: mean/median ratio, a crude inequality proxy
      - `concentrated_disadvantage`: composite index combining poverty,
        single-parent share, and unemployment (cf. Sampson 1997)
      - `poor_x_black`, `poor_x_hisp`: interaction terms flagged by
        sociological literature on intersectional disadvantage
      - `school_x_poverty`: interaction between school quality and poverty

    Parameters
    ----------
    df : input dataframe
    z_params : dict mapping column name -> {"mean": float, "std": float}
        for the variables used in the concentrated-disadvantage z-score.
        If None, computed from `df` (training-time / fit). If provided,
        used as-is (test-time / serving-time / single-row transform).

    Returns
    -------
    df_with_features, z_params  : the augmented dataframe and the (possibly
        newly fit) z-score parameters.
    """
    df = df.copy()

    # Log transforms (stateless)
    df["log_med_hhinc"] = np.log1p(df["med_hhinc"])
    df["log_popdensity"] = np.log1p(df["popdensity"])

    # Inequality index (stateless)
    df["inequality_index"] = df["hhinc_mean"] / (df["med_hhinc"] + 1e-9)

    # Sampson's concentrated-disadvantage composite (z-scored sum)
    # We track a derived variable "unemp_share" = 1 - emp_rate so the
    # z-params dict is keyed cleanly.
    derived = {
        "poor_share": df["poor_share"].astype(float),
        "singleparent_share": df["singleparent_share"].astype(float),
        "unemp_share": (1.0 - df["emp_rate"]).astype(float),
    }

    fitting = z_params is None
    if fitting:
        z_params = {}
        for k, series in derived.items():
            mu = float(series.mean())
            sd = float(series.std())
            if not np.isfinite(sd) or sd < 1e-9:
                sd = 1.0
            z_params[k] = {"mean": mu, "std": sd}

    z_terms = []
    for k, series in derived.items():
        mu = z_params[k]["mean"]
        sd = z_params[k]["std"]
        z_terms.append((series - mu) / (sd + 1e-9))
    df["concentrated_disadvantage"] = sum(z_terms) / 3.0

    # Intersectional interaction terms (stateless)
    df["poor_x_black"] = df["poor_share"] * df["share_black"]
    df["poor_x_hisp"] = df["poor_share"] * df["share_hisp"]
    df["school_x_poverty"] = df["gsmn_math_pcst"] * df["poor_share"]

    return df, z_params


ENGINEERED_FEATURE_NAMES = [
    "log_med_hhinc", "log_popdensity", "inequality_index",
    "concentrated_disadvantage", "poor_x_black", "poor_x_hisp",
    "school_x_poverty",
]


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def fit_transform(
    df_train: pd.DataFrame,
    config: PreprocessingConfig,
) -> tuple[pd.DataFrame, PreprocessingArtifacts]:
    """
    Fit the full preprocessing pipeline on training data and transform it.

    Returns
    -------
    df_processed : pd.DataFrame
        Preprocessed training data.
    artifacts : PreprocessingArtifacts
        Fitted state for transforming test data.
    """
    artifacts = PreprocessingArtifacts()
    df = df_train.copy()

    # 1. Winsorize heavy-tailed columns (before imputation to reduce leverage).
    if config.winsorize:
        df = winsorize_fit_transform(
            df, HEAVY_TAILED_FEATURES, config.winsor_quantile, artifacts
        )
        logger.info(f"Winsorized {len(HEAVY_TAILED_FEATURES)} heavy-tailed features.")

    # 2. Impute missing numeric features.
    feature_cols_for_imputer = [c for c in NUMERIC_FEATURES if c in df.columns]
    artifacts.imputer = fit_imputer(df, config.imputation_strategy,
                                     feature_cols_for_imputer)
    df = apply_imputer(df, artifacts.imputer, feature_cols_for_imputer)
    logger.info(f"Imputed missing values using '{config.imputation_strategy}' strategy.")

    # 3. Feature engineering (after imputation).
    if config.add_engineered_features:
        df, z_params = add_engineered_features(df, z_params=None)
        artifacts.z_params = z_params
        logger.info(f"Added {len(ENGINEERED_FEATURE_NAMES)} engineered features.")

    # 4. Standardize numeric features.
    all_numeric = feature_cols_for_imputer + (
        ENGINEERED_FEATURE_NAMES if config.add_engineered_features else []
    )
    artifacts.feature_names = all_numeric
    if config.standardize:
        artifacts.scaler = StandardScaler().fit(df[all_numeric].values)
        df[all_numeric] = artifacts.scaler.transform(df[all_numeric].values)
        logger.info("Standardized all numeric features (StandardScaler).")

    return df, artifacts


def transform(
    df_test: pd.DataFrame,
    artifacts: PreprocessingArtifacts,
    config: PreprocessingConfig,
) -> pd.DataFrame:
    """Apply the fitted preprocessing to new data."""
    df = df_test.copy()
    if config.winsorize:
        df = winsorize_transform(df, artifacts)
    feature_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    df = apply_imputer(df, artifacts.imputer, feature_cols)
    if config.add_engineered_features:
        df, _ = add_engineered_features(df, z_params=artifacts.z_params or None)
    if config.standardize and artifacts.scaler is not None:
        df[artifacts.feature_names] = artifacts.scaler.transform(
            df[artifacts.feature_names].values
        )
    return df


if __name__ == "__main__":
    # Smoke test
    logging.basicConfig(level=logging.INFO)
    import sys
    sys.path.insert(0, ".")
    from data_loader import load_dataset

    df = load_dataset(use_synthetic=True)
    train = df.iloc[:50000]
    test = df.iloc[50000:]

    cfg = PreprocessingConfig()
    train_p, artifacts = fit_transform(train, cfg)
    test_p = transform(test, artifacts, cfg)

    print(f"Train processed shape: {train_p.shape}")
    print(f"Test processed shape: {test_p.shape}")
    print(f"Any NaNs left? train: {train_p.isna().any().any()}, "
          f"test: {test_p.isna().any().any()}")
    print(f"Engineered feature sample:\n"
          f"{train_p[['concentrated_disadvantage', 'poor_x_black']].head()}")
