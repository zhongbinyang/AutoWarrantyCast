"""Single-claim ("unit") repair cost prediction models.

For a given development age (vehicle age in months), the unit repair cost is
defined as::

    unit_cost(age) = total claim amount at that age / total claim count at that age

aggregated across sale batches (optionally per system). This module fits one
of several interchangeable forecasting methods to the *historical* unit-cost
series and projects it onto future ages, together with a residual-bootstrap
mechanism used by the Monte Carlo engine to generate prediction intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from .data_io import TriangleBundle

METHOD_HISTORICAL_MEAN = "历史均值法"
METHOD_MOVING_AVERAGE = "移动平均法"
METHOD_EXP_SMOOTHING = "指数平滑法"
METHOD_XGBOOST = "XGBoost回归"
METHOD_CREDIBILITY_SHRINKAGE = "经验贝叶斯收缩法"

ALL_METHODS = [
    METHOD_HISTORICAL_MEAN,
    METHOD_CREDIBILITY_SHRINKAGE,
    METHOD_MOVING_AVERAGE,
    METHOD_EXP_SMOOTHING,
    METHOD_XGBOOST,
]



@dataclass
class UnitCostFitResult:
    """Fitted unit-cost model, ready to predict at arbitrary future ages."""

    method: str
    ages_hist: np.ndarray
    unit_cost_hist: np.ndarray
    weights_hist: np.ndarray
    fitted_hist: np.ndarray
    residuals: np.ndarray
    predict_fn: Callable[[np.ndarray], np.ndarray]
    r2: float
    mae: float

    def predict(self, ages: np.ndarray) -> np.ndarray:
        return np.clip(self.predict_fn(np.asarray(ages, dtype=float)), 0.0, None)


def compute_historical_unit_cost(bundle_count: TriangleBundle, bundle_amount: TriangleBundle) -> pd.DataFrame:
    """Aggregate observed cells into a per-age historical unit-cost series."""
    mis_cols = [f"MIS{a}" for a in bundle_count.mis_ages]
    rows = []
    for age, col in zip(bundle_count.mis_ages, mis_cols):
        mask = bundle_count.observed_mask[col]
        count_sum = float(bundle_count.count.loc[mask, col].fillna(0).sum())
        amount_sum = float(bundle_amount.amount.loc[mask, col].fillna(0).sum())
        if count_sum > 0:
            rows.append({"age": age, "count_sum": count_sum, "amount_sum": amount_sum,
                         "unit_cost": amount_sum / count_sum})
    return pd.DataFrame(rows).sort_values("age").reset_index(drop=True)


def _holt_linear_forecast(values: np.ndarray, alpha: float = 0.3, beta: float = 0.2):
    """Holt's linear (level + trend) exponential smoothing.

    Returns the in-sample fitted values plus a callable that extrapolates
    ``level + trend * horizon`` for any number of steps beyond the series.
    """
    level = values[0]
    trend = values[1] - values[0] if len(values) > 1 else 0.0
    fitted = [level]
    for y in values[1:]:
        last_level = level
        level = alpha * y + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        fitted.append(level)
    fitted = np.array(fitted)
    final_level, final_trend = level, trend
    n_hist = len(values)

    def predict(ages: np.ndarray) -> np.ndarray:
        horizon = ages - n_hist + 1
        horizon = np.clip(horizon, 1, None)
        return final_level + final_trend * horizon

    return fitted, predict


def _moving_average_forecast(values: np.ndarray, window: int = 3):
    """Trailing moving-average forecast; future ages repeat the last window mean."""
    fitted = np.array(
        [values[max(0, i - window + 1) : i + 1].mean() for i in range(len(values))]
    )
    last_mean = values[-window:].mean() if len(values) >= 1 else values.mean()

    def predict(ages: np.ndarray) -> np.ndarray:
        return np.full_like(ages, fill_value=last_mean, dtype=float)

    return fitted, predict


def _historical_mean_forecast(values: np.ndarray, weights: np.ndarray):
    """Flat forecast at the exposure-weighted historical mean."""
    mean_val = float(np.average(values, weights=weights)) if weights.sum() > 0 else float(values.mean())
    fitted = np.full_like(values, fill_value=mean_val, dtype=float)

    def predict(ages: np.ndarray) -> np.ndarray:
        return np.full_like(ages, fill_value=mean_val, dtype=float)

    return fitted, predict


def _credibility_shrinkage_forecast(
    values: np.ndarray,
    weights: np.ndarray,
    prior_unit_cost: float = 2000.0,
    credibility_k: float = 15.0,
):
    """Empirical Bayes (Bühlmann credibility) shrinkage forecast.

    Shrinks the exposure-weighted sample mean toward a baseline engineering
    prior unit cost according to credibility factor Z = n_claims / (n_claims + K).
    When historical claim counts are few, prediction leans on the baseline prior.
    When claim counts are abundant, Z -> 1, recovering the pure sample mean.
    """
    total_claims = float(weights.sum())
    sample_mean = float(np.average(values, weights=weights)) if total_claims > 0 else prior_unit_cost
    z = total_claims / (total_claims + max(credibility_k, 1.0))
    shrunk_mean = float(z * sample_mean + (1.0 - z) * prior_unit_cost)
    fitted = np.full_like(values, fill_value=shrunk_mean, dtype=float)

    def predict(ages: np.ndarray) -> np.ndarray:
        return np.full_like(ages, fill_value=shrunk_mean, dtype=float)

    return fitted, predict


def _xgboost_forecast(ages: np.ndarray, values: np.ndarray, weights: np.ndarray):
    """Gradient-boosted tree regression on (age, age^2) features.

    Falls back to :class:`sklearn.ensemble.GradientBoostingRegressor` when
    the optional ``xgboost`` package is not installed, so the app keeps
    working in a minimal environment.
    """
    features = np.column_stack([ages, ages**2])
    try:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            reg_lambda=1.0,
            random_state=0,
        )
        model.fit(features, values, sample_weight=weights)
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor

        model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.08, random_state=0)
        model.fit(features, values, sample_weight=weights)

    fitted = model.predict(features)

    def predict(future_ages: np.ndarray) -> np.ndarray:
        feat = np.column_stack([future_ages, future_ages**2])
        return model.predict(feat)

    return fitted, predict


def fit_unit_cost_model(
    hist_df: pd.DataFrame,
    method: str = METHOD_HISTORICAL_MEAN,
    prior_unit_cost: Optional[float] = None,
    credibility_k: float = 15.0,
) -> UnitCostFitResult:
    """Fit the chosen unit-cost forecasting method to the historical series."""
    if hist_df.empty:
        raise ValueError("没有可用于拟合单次维修成本模型的历史数据")

    ages = hist_df["age"].to_numpy(dtype=float)
    values = hist_df["unit_cost"].to_numpy(dtype=float)
    weights = hist_df["count_sum"].to_numpy(dtype=float)

    p_cost = float(prior_unit_cost) if prior_unit_cost is not None and prior_unit_cost > 0 else 2000.0

    if method == METHOD_CREDIBILITY_SHRINKAGE:
        fitted, predict_fn = _credibility_shrinkage_forecast(
            values, weights, prior_unit_cost=p_cost, credibility_k=credibility_k
        )
    elif method == METHOD_MOVING_AVERAGE:
        fitted, predict_fn = _moving_average_forecast(values)
    elif method == METHOD_EXP_SMOOTHING:
        fitted, predict_fn = _holt_linear_forecast(values)
    elif method == METHOD_XGBOOST:
        fitted, predict_fn = _xgboost_forecast(ages, values, weights)
    else:
        fitted, predict_fn = _historical_mean_forecast(values, weights)

    residuals = values - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((values - values.mean()) ** 2)) or 1e-9
    r2 = 1.0 - ss_res / ss_tot
    mae = float(np.mean(np.abs(residuals)))

    return UnitCostFitResult(
        method=method,
        ages_hist=ages,
        unit_cost_hist=values,
        weights_hist=weights,
        fitted_hist=fitted,
        residuals=residuals,
        predict_fn=predict_fn,
        r2=r2,
        mae=mae,
    )


def sample_unit_cost_paths(
    fit: UnitCostFitResult, ages: np.ndarray, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Residual-bootstrap sample of unit-cost paths, shape (n_samples, len(ages)).

    Guarantees reasonable physical variance on few-shot / short-history cases.
    """
    point = fit.predict(ages)
    if len(fit.residuals) == 0:
        std_est = np.maximum(point * 0.15, 50.0)
        draws = rng.normal(0.0, std_est, size=(n_samples, len(ages)))
        paths = point[np.newaxis, :] + draws
        return np.clip(paths, 0.0, None)

    # If only 1-2 historical points, synthesize mild dispersion around observed residuals
    if len(fit.residuals) <= 2:
        base_std = float(np.std(fit.residuals)) if len(fit.residuals) > 1 else 0.0
        min_std = float(np.mean(point) * 0.12)
        eff_std = max(base_std, min_std, 30.0)
        draws = rng.normal(0.0, eff_std, size=(n_samples, len(ages)))
    else:
        draws = rng.choice(fit.residuals, size=(n_samples, len(ages)), replace=True)

    paths = point[np.newaxis, :] + draws
    return np.clip(paths, 0.0, None)

