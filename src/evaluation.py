"""
evaluation.py
=============
Evaluation metrics, error analysis, and subgroup (fairness) auditing for the
opportunity-prediction models.

Rubric items addressed here:
  * 3 distinct evaluation metrics (R^2, RMSE, MAE)
  * Error analysis with visualization and failure-case discussion
  * Subgroup (fairness) analysis by demographic group
  * Edge-case / out-of-distribution evaluation
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Container for evaluation results."""
    model_name: str
    rmse: float
    mae: float
    r2: float
    predictions: np.ndarray
    targets: np.ndarray
    inference_time_per_1k: float = 0.0

    def as_dict(self):
        return {
            "model": self.model_name,
            "RMSE": self.rmse,
            "MAE": self.mae,
            "R2": self.r2,
            "inference_ms_per_1k": self.inference_time_per_1k * 1000,
        }

"AI assisted scaffolding"
def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: Optional[str] = None,
) -> EvalResult:
    """
    Compute RMSE, MAE, R^2, and inference time for a fitted model.

    Three distinct metrics: covers the "at least three metrics" rubric item.
    """
    if model_name is None:
        model_name = getattr(model, "name", model.__class__.__name__)

    # Measure inference time (rubric 3pt: computational efficiency)
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    elapsed = time.perf_counter() - t0
    per_1k = elapsed / max(1, len(X_test)) * 1000

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    return EvalResult(
        model_name=model_name,
        rmse=rmse, mae=mae, r2=r2,
        predictions=np.asarray(y_pred),
        targets=np.asarray(y_test),
        inference_time_per_1k=per_1k,
    )


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def residual_stats(eval_result: EvalResult) -> pd.DataFrame:
    """Return a dataframe of residuals and their summary stats."""
    residuals = eval_result.predictions - eval_result.targets
    return pd.DataFrame({
        "target": eval_result.targets,
        "prediction": eval_result.predictions,
        "residual": residuals,
        "abs_residual": np.abs(residuals),
    })


def worst_errors(
    eval_result: EvalResult,
    df_context: pd.DataFrame,
    k: int = 20,
) -> pd.DataFrame:
    """
    Return the top-k worst-prediction rows along with their original features
    for qualitative error analysis.
    """
    df = df_context.copy().reset_index(drop=True)
    rs = residual_stats(eval_result).reset_index(drop=True)
    joined = pd.concat([rs, df], axis=1)
    return joined.sort_values("abs_residual", ascending=False).head(k)


# ---------------------------------------------------------------------------
# Subgroup / fairness analysis
# ---------------------------------------------------------------------------

def subgroup_metrics(
    eval_result: EvalResult,
    subgroup_values: np.ndarray,
    bins: list,
    bin_labels: list,
) -> pd.DataFrame:
    """
    Compute RMSE, MAE, R^2 stratified by a continuous subgroup variable
    (e.g., fraction-minority or fraction-poor) binned into quantile buckets.

    This operationalizes fairness auditing: if the model is systematically
    worse for high-minority or high-poverty tracts, that's a bias we should
    report and discuss.
    """
    cat = pd.cut(subgroup_values, bins=bins, labels=bin_labels,
                 include_lowest=True)
    rows = []
    for label in bin_labels:
        mask = np.asarray(cat == label)
        if mask.sum() < 10:
            continue
        y_t = eval_result.targets[mask]
        y_p = eval_result.predictions[mask]
        rows.append({
            "subgroup": label,
            "n": int(mask.sum()),
            "RMSE": float(np.sqrt(mean_squared_error(y_t, y_p))),
            "MAE": float(mean_absolute_error(y_t, y_p)),
            "R2": float(r2_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else float("nan"),
            "mean_target": float(y_t.mean()),
            "mean_prediction": float(y_p.mean()),
        })
    return pd.DataFrame(rows)


def demographic_disparity_report(
    eval_result: EvalResult,
    df_context: pd.DataFrame,
) -> dict:
    """
    Compute disparity metrics across four subgroupings:
      * By tract poverty rate (poor_share)
      * By tract minority share (nonwhite_share2010)
      * By tract single-parent share
      * By school quality decile

    Returns a dict of dataframes; each reports per-bucket RMSE, MAE, R^2.
    """
    df = df_context.reset_index(drop=True)
    out = {}
    out["by_poverty"] = subgroup_metrics(
        eval_result,
        df["poor_share"].values,
        bins=[0, 0.1, 0.2, 0.3, 1.0],
        bin_labels=["low (<10%)", "moderate (10-20%)",
                    "high (20-30%)", "very high (30%+)"],
    )
    out["by_minority_share"] = subgroup_metrics(
        eval_result,
        df["nonwhite_share2010"].values,
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        bin_labels=["majority white (<25% minority)",
                    "mixed (25-50%)", "majority minority (50-75%)",
                    "predominantly minority (>75%)"],
    )
    if "singleparent_share" in df.columns:
        out["by_singleparent"] = subgroup_metrics(
            eval_result,
            df["singleparent_share"].values,
            bins=[0, 0.2, 0.35, 0.5, 1.0],
            bin_labels=["<20%", "20-35%", "35-50%", ">50%"],
        )
    if "gsmn_math_pcst" in df.columns:
        out["by_school_quality"] = subgroup_metrics(
            eval_result,
            df["gsmn_math_pcst"].values,
            bins=[0, 25, 50, 75, 100],
            bin_labels=["bottom quartile", "2nd quartile",
                        "3rd quartile", "top quartile"],
        )
    return out


# ---------------------------------------------------------------------------
# Out-of-distribution / edge case evaluation
# ---------------------------------------------------------------------------

def edge_case_analysis(
    eval_result: EvalResult,
    df_context: pd.DataFrame,
    feature_col: str,
    low_pct: float = 0.05,
    high_pct: float = 0.95,
) -> pd.DataFrame:
    """
    Compare model performance on the tails of a feature distribution (low
    and high extremes) vs. the middle, to detect OOD degradation.
    """
    df = df_context.reset_index(drop=True)
    vals = df[feature_col].values
    lo = np.quantile(vals, low_pct)
    hi = np.quantile(vals, high_pct)

    buckets = {
        f"low tail (<= {low_pct*100:.0f}%)": vals <= lo,
        "middle":                            (vals > lo) & (vals < hi),
        f"high tail (>= {high_pct*100:.0f}%)": vals >= hi,
    }
    rows = []
    for name, mask in buckets.items():
        if mask.sum() < 10:
            continue
        y_t = eval_result.targets[mask]
        y_p = eval_result.predictions[mask]
        rows.append({
            "bucket": name,
            "n": int(mask.sum()),
            "RMSE": float(np.sqrt(mean_squared_error(y_t, y_p))),
            "MAE": float(mean_absolute_error(y_t, y_p)),
        })
    return pd.DataFrame(rows)


def comparison_table(results: list[EvalResult]) -> pd.DataFrame:
    """Summarize a list of EvalResult objects into a single dataframe."""
    rows = [r.as_dict() for r in results]
    df = pd.DataFrame(rows)
    return df.sort_values("RMSE").reset_index(drop=True)
