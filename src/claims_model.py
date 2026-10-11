"""Forecasting the claim-count and claim-amount triangles.

Claim count is always forecast via the bivariate Weibull competing-risk
model (see :mod:`weibull_model`). Claim amount can be forecast either as
``predicted count x predicted unit cost`` (the method emphasized by the
specification) or via a development-factor approach (classic chain ladder,
or a LightGBM-learned development factor as a drop-in replacement).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .cost_model import UnitCostFitResult
from .data_io import TriangleBundle
from .exposure import batch_max_ages
from .utils import mis_col, safe_divide
from .weibull_model import WeibullFitResult

AMOUNT_METHOD_COUNT_X_UNITCOST = "台次×单次成本"
AMOUNT_METHOD_CHAIN_LADDER = "链梯法"
AMOUNT_METHOD_LIGHTGBM = "LightGBM发展因子法"
AMOUNT_METHOD_BORNHUETTER_FERGUSON = "Bornhuetter-Ferguson(BF)法"



def full_age_range(
    bundle: TriangleBundle,
    sales_index: List[str],
    warranty_months: int,
    mileage_limit_km: Optional[float],
    mileage_by_month: Optional[Dict[str, float]],
) -> List[int]:
    """Determine the complete set of development ages to display/forecast."""
    max_ages = batch_max_ages(
        pd.DataFrame({"销售月份": sales_index, "销量": [1] * len(sales_index)}),
        warranty_months,
        mileage_limit_km,
        mileage_by_month,
    )
    overall_max = max([*bundle.mis_ages, *max_ages.values(), warranty_months])
    return list(range(0, int(overall_max) + 1))


def last_observed_age(bundle: TriangleBundle, sale_month: str) -> Optional[int]:
    """Return the last development age with actual (observed) data for a batch."""
    observed_ages = [
        age for age in bundle.mis_ages if bool(bundle.observed_mask.loc[sale_month, mis_col(age)])
    ]
    return max(observed_ages) if observed_ages else None


def forecast_counts_via_weibull(
    bundle: TriangleBundle,
    sales_df: pd.DataFrame,
    fit: WeibullFitResult,
    mileage_by_month: Optional[Dict[str, float]],
    warranty_months: int,
    mileage_limit_km: Optional[float],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    """Fill the incremental claim-count triangle using the Weibull fit.

    Future cumulative counts are calibrated to match the actual observed
    cumulative count at each batch's last observed age (additive offset),
    so the forecast connects smoothly to history rather than jumping to the
    model's unconditional curve.

    Returns ``(count_incremental, forecast_mask, ages)``.
    """
    sales_map = dict(zip(sales_df["销售月份"], sales_df["销量"]))
    ages = full_age_range(bundle, list(bundle.count.index), warranty_months, mileage_limit_km, mileage_by_month)
    mis_cols = [mis_col(a) for a in ages]
    max_ages = batch_max_ages(sales_df, warranty_months, mileage_limit_km, mileage_by_month)

    count_incremental = pd.DataFrame(index=bundle.count.index, columns=mis_cols, dtype=float)
    forecast_mask = pd.DataFrame(False, index=bundle.count.index, columns=mis_cols)

    for sm in bundle.count.index:
        sales = float(sales_map.get(sm, 0))
        mileage = float(mileage_by_month.get(sm, 0.0)) if mileage_by_month else 0.0
        max_age = max_ages.get(sm, warranty_months)
        last_obs_age = last_observed_age(bundle, sm)

        if last_obs_age is not None:
            actual_cum_last = float(
                bundle.count.loc[sm, [mis_col(a) for a in bundle.mis_ages if a <= last_obs_age]]
                .fillna(0)
                .sum()
            )
            model_cum_last = float(fit.cum_rate(np.array([last_obs_age]), mileage)[0]) * sales
            offset = actual_cum_last - model_cum_last
        else:
            offset = 0.0

        prev_cum = 0.0
        for age in ages:
            col = mis_col(age)
            if age in bundle.mis_ages and bool(bundle.observed_mask.loc[sm, col]):
                val = float(bundle.count.loc[sm, col]) if pd.notna(bundle.count.loc[sm, col]) else 0.0
                count_incremental.loc[sm, col] = val
                prev_cum += val
                continue
            if age > max_age:
                count_incremental.loc[sm, col] = np.nan
                continue
            pred_cum = float(fit.cum_rate(np.array([age]), mileage)[0]) * sales + offset
            pred_cum = max(pred_cum, prev_cum)
            inc = pred_cum - prev_cum
            count_incremental.loc[sm, col] = inc
            forecast_mask.loc[sm, col] = True
            prev_cum = pred_cum

    return count_incremental, forecast_mask, ages


def forecast_amount_via_unit_cost(
    bundle: TriangleBundle,
    count_incremental: pd.DataFrame,
    unit_cost_fit: UnitCostFitResult,
    ages: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fill the amount triangle as predicted count x predicted unit cost."""
    mis_cols = [mis_col(a) for a in ages]
    amount_incremental = pd.DataFrame(index=bundle.amount.index, columns=mis_cols, dtype=float)
    forecast_mask = pd.DataFrame(False, index=bundle.amount.index, columns=mis_cols)
    unit_cost_by_age = dict(zip(ages, unit_cost_fit.predict(np.array(ages, dtype=float))))

    for sm in bundle.amount.index:
        for age in ages:
            col = mis_col(age)
            if age in bundle.mis_ages and bool(bundle.observed_mask.loc[sm, col]):
                amount_incremental.loc[sm, col] = (
                    float(bundle.amount.loc[sm, col]) if pd.notna(bundle.amount.loc[sm, col]) else 0.0
                )
                continue
            count_val = count_incremental.loc[sm, col] if col in count_incremental.columns else np.nan
            if pd.isna(count_val):
                amount_incremental.loc[sm, col] = np.nan
                continue
            amount_incremental.loc[sm, col] = float(count_val) * float(unit_cost_by_age[age])
            forecast_mask.loc[sm, col] = True

    return amount_incremental, forecast_mask


def _cumulative_from_incremental(
    triangle_incremental: pd.DataFrame, observed_mask: pd.DataFrame, mis_cols: List[str]
) -> pd.DataFrame:
    inc_observed = triangle_incremental[mis_cols].where(observed_mask[mis_cols])
    return inc_observed.fillna(0).cumsum(axis=1)


def compute_development_factors(
    cum_df: pd.DataFrame, observed_mask: pd.DataFrame, mis_ages: List[int]
) -> Dict[int, float]:
    """Classic volume-weighted age-to-age development factors."""
    mis_cols = [mis_col(a) for a in mis_ages]
    factors: Dict[int, float] = {}
    for i in range(len(mis_cols) - 1):
        c0, c1 = mis_cols[i], mis_cols[i + 1]
        both = observed_mask[c0] & observed_mask[c1]
        den = float(cum_df.loc[both, c0].sum())
        num = float(cum_df.loc[both, c1].sum())
        factors[mis_ages[i + 1]] = (num / den) if den > 1e-9 else 1.0
    return factors


def chain_ladder_fill(
    triangle_incremental: pd.DataFrame,
    observed_mask: pd.DataFrame,
    mis_ages: List[int],
    all_ages: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fill a triangle's future cells with classic chain-ladder development."""
    mis_cols = [mis_col(a) for a in mis_ages]
    cum = _cumulative_from_incremental(triangle_incremental, observed_mask, mis_cols)
    factors = compute_development_factors(cum, observed_mask, mis_ages)

    all_cols = [mis_col(a) for a in all_ages]
    result_cum = cum.reindex(columns=all_cols)
    forecast_mask = pd.DataFrame(False, index=triangle_incremental.index, columns=all_cols)

    for sm in triangle_incremental.index:
        observed_ages = [a for a in mis_ages if bool(observed_mask.loc[sm, mis_col(a)])]
        last_age = max(observed_ages) if observed_ages else -1
        cur = float(cum.loc[sm, mis_col(last_age)]) if last_age >= 0 else 0.0
        for age in all_ages:
            if age <= last_age:
                continue
            f = factors.get(age, 1.0)
            cur = cur * f
            result_cum.loc[sm, mis_col(age)] = cur
            forecast_mask.loc[sm, mis_col(age)] = True

    inc_result = result_cum.diff(axis=1)
    inc_result.iloc[:, 0] = result_cum.iloc[:, 0]
    return inc_result, forecast_mask


def lightgbm_development_fill(
    triangle_incremental: pd.DataFrame,
    observed_mask: pd.DataFrame,
    mis_ages: List[int],
    all_ages: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """LightGBM-learned development factors as an alternative to chain ladder.

    Falls back to :func:`chain_ladder_fill` when the optional ``lightgbm``
    dependency is unavailable or there is too little training data for a
    stable fit.
    """
    mis_cols = [mis_col(a) for a in mis_ages]
    cum = _cumulative_from_incremental(triangle_incremental, observed_mask, mis_cols)

    X: List[List[float]] = []
    y: List[float] = []
    for i in range(len(mis_cols) - 1):
        c0, c1 = mis_cols[i], mis_cols[i + 1]
        both = observed_mask[c0] & observed_mask[c1]
        for sm in triangle_incremental.index[both]:
            v0 = float(cum.loc[sm, c0])
            v1 = float(cum.loc[sm, c1])
            if v0 <= 1e-9:
                continue
            X.append([mis_ages[i], np.log1p(v0)])
            y.append(v1 / v0)

    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        return chain_ladder_fill(triangle_incremental, observed_mask, mis_ages, all_ages)

    if len(X) < 8:
        return chain_ladder_fill(triangle_incremental, observed_mask, mis_ages, all_ages)

    model = LGBMRegressor(n_estimators=100, max_depth=3, min_child_samples=3, verbose=-1)
    model.fit(np.array(X), np.array(y))

    all_cols = [mis_col(a) for a in all_ages]
    result_cum = cum.reindex(columns=all_cols)
    forecast_mask = pd.DataFrame(False, index=triangle_incremental.index, columns=all_cols)

    for sm in triangle_incremental.index:
        observed_ages = [a for a in mis_ages if bool(observed_mask.loc[sm, mis_col(a)])]
        last_age = max(observed_ages) if observed_ages else -1
        cur = float(cum.loc[sm, mis_col(last_age)]) if last_age >= 0 else 0.0
        for age in all_ages:
            if age <= last_age:
                continue
            feat = np.array([[age - 1, np.log1p(max(cur, 0.0))]])
            factor = float(model.predict(feat)[0])
            factor = max(factor, 1.0)
            cur = cur * factor
            result_cum.loc[sm, mis_col(age)] = cur
            forecast_mask.loc[sm, mis_col(age)] = True

    inc_result = result_cum.diff(axis=1)
    inc_result.iloc[:, 0] = result_cum.iloc[:, 0]
    return inc_result, forecast_mask


def bornhuetter_ferguson_fill(
    triangle_incremental: pd.DataFrame,
    observed_mask: pd.DataFrame,
    sales_df: pd.DataFrame,
    mis_ages: List[int],
    all_ages: List[int],
    prior_expected_loss_per_vehicle: float = 150.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fill a triangle's future cells using the Bornhuetter-Ferguson (BF) actuarial method.

    Blends actual cumulative losses with an a priori expected ultimate loss for the
    unobserved tail, making it the industry standard for small-sample and short-history
    regimes where empirical development factors fluctuate wildly.
    """
    mis_cols = [mis_col(a) for a in mis_ages]
    cum = _cumulative_from_incremental(triangle_incremental, observed_mask, mis_cols)
    factors = compute_development_factors(cum, observed_mask, mis_ages)

    all_cols = [mis_col(a) for a in all_ages]
    cdf: Dict[int, float] = {}
    running_f = 1.0
    for a in reversed(all_ages):
        f = factors.get(a, 1.0)
        running_f *= max(f, 1.0)
        cdf[a] = running_f

    # Emergence fraction p(age) = 1.0 / CDF(age)
    p_emerged: Dict[int, float] = {}
    for a in all_ages:
        p = 1.0 / cdf[a] if cdf[a] > 1e-9 else 1.0
        p_emerged[a] = float(np.clip(p, 0.0, 1.0))

    cur_p = 0.0
    for a in all_ages:
        cur_p = max(cur_p, p_emerged[a])
        p_emerged[a] = cur_p

    sales_map = dict(zip(sales_df["销售月份"], sales_df["销量"])) if sales_df is not None else {}
    inc_result = triangle_incremental.reindex(columns=all_cols).copy()
    forecast_mask = pd.DataFrame(False, index=triangle_incremental.index, columns=all_cols)

    for sm in triangle_incremental.index:
        sales = float(sales_map.get(sm, 1000.0))
        prior_ultimate = sales * prior_expected_loss_per_vehicle
        observed_ages = [a for a in mis_ages if bool(observed_mask.loc[sm, mis_col(a)])]
        last_age = max(observed_ages) if observed_ages else -1

        for i, age in enumerate(all_ages):
            col = mis_col(age)
            if age <= last_age:
                continue
            p_prev = p_emerged[all_ages[i - 1]] if i > 0 else 0.0
            p_curr = p_emerged[age]
            delta_p = max(p_curr - p_prev, 0.0)
            inc_val = prior_ultimate * delta_p
            inc_result.loc[sm, col] = inc_val
            forecast_mask.loc[sm, col] = True

    return inc_result, forecast_mask


def forecast_amount(
    bundle: TriangleBundle,
    count_incremental: pd.DataFrame,
    unit_cost_fit: Optional[UnitCostFitResult],
    ages: List[int],
    method: str = AMOUNT_METHOD_COUNT_X_UNITCOST,
    sales_df: Optional[pd.DataFrame] = None,
    prior_expected_loss_per_vehicle: float = 150.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Dispatch to the selected claim-amount forecasting method."""
    if method == AMOUNT_METHOD_BORNHUETTER_FERGUSON:
        return bornhuetter_ferguson_fill(
            bundle.amount,
            bundle.observed_mask,
            sales_df=sales_df,
            mis_ages=bundle.mis_ages,
            all_ages=ages,
            prior_expected_loss_per_vehicle=prior_expected_loss_per_vehicle,
        )
    if method == AMOUNT_METHOD_CHAIN_LADDER:
        return chain_ladder_fill(bundle.amount, bundle.observed_mask, bundle.mis_ages, ages)
    if method == AMOUNT_METHOD_LIGHTGBM:
        return lightgbm_development_fill(bundle.amount, bundle.observed_mask, bundle.mis_ages, ages)
    if unit_cost_fit is None:
        raise ValueError("台次×单次成本法需要提供已拟合的单次成本模型")
    return forecast_amount_via_unit_cost(bundle, count_incremental, unit_cost_fit, ages)

