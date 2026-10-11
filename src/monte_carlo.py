"""Monte Carlo simulation of future claim-count and claim-amount totals.

Each simulation iteration draws a Weibull parameter vector from the fitted
asymptotic covariance and an independent unit-cost residual-bootstrap path,
then combines them ``count x unit_cost`` *within the same iteration* so the
resulting amount distribution correctly reflects the correlation between
count and cost uncertainty, per the specification.

The implementation is fully vectorized over (simulation, development-age)
pairs using NumPy broadcasting -- there is no explicit per-simulation Python
loop, only a loop over sale-month batches (a few dozen at most), which keeps
5,000+ simulation runs fast without needing process-based parallelism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .claims_model import last_observed_age
from .cost_model import UnitCostFitResult, sample_unit_cost_paths
from .data_io import TriangleBundle
from .exposure import batch_max_ages
from .weibull_model import WeibullFitResult, sample_parameters


@dataclass
class DistributionSummary:
    """Standard risk metrics for a scalar Monte Carlo output distribution."""

    mean: float
    std: float
    cv: float
    var90: float
    var95: float
    samples: np.ndarray


@dataclass
class MonteCarloResult:
    """Full Monte Carlo simulation output for one system."""

    total_count: DistributionSummary
    total_amount: DistributionSummary
    band_ages: np.ndarray
    band_lower: np.ndarray
    band_median: np.ndarray
    band_upper: np.ndarray
    n_simulations: int


def summarize_distribution(samples: np.ndarray) -> DistributionSummary:
    """Compute mean/std/CV/VaR90/VaR95 for a 1-D sample array.

    VaR here follows the common practical convention of "the p-th percentile
    of the outcome distribution" (i.e. VaR90 is the value only exceeded 10%
    of the time), rather than a loss-relative-to-mean definition.
    """
    mean = float(np.mean(samples))
    std = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
    cv = std / mean if abs(mean) > 1e-9 else 0.0
    var90 = float(np.percentile(samples, 90))
    var95 = float(np.percentile(samples, 95))
    return DistributionSummary(mean=mean, std=std, cv=cv, var90=var90, var95=var95, samples=samples)


def cumulative_hazard_time_matrix(
    ages: np.ndarray, breakpoints: List[int], beta_t: np.ndarray, eta_t: np.ndarray
) -> np.ndarray:
    """Vectorized piecewise time cumulative hazard for a batch of parameter draws.

    Args:
        ages: 1-D array of development ages, shape (A,).
        breakpoints: interior segment boundaries (length n_seg - 1).
        beta_t: shape (S, n_seg) sampled shape parameters.
        eta_t: shape (S, n_seg) sampled scale parameters.

    Returns:
        Cumulative hazard matrix of shape (S, A).
    """
    S, n_seg = beta_t.shape
    edges = np.array([0.0] + [float(b) for b in breakpoints])
    h_at_edge = np.zeros((S, n_seg))
    for k in range(n_seg - 1):
        seg_len = edges[k + 1] - edges[k]
        h_at_edge[:, k + 1] = h_at_edge[:, k] + (seg_len / eta_t[:, k]) ** beta_t[:, k]

    idx = np.searchsorted(np.array(breakpoints, dtype=float), ages, side="right")
    idx = np.clip(idx, 0, n_seg - 1)  # shape (A,)

    offset = ages - edges[idx]
    offset = np.clip(offset, 0.0, None)  # (A,)
    offset_b = np.broadcast_to(offset, (S, len(ages)))
    beta_sel = beta_t[:, idx]  # (S, A)
    eta_sel = eta_t[:, idx]  # (S, A)
    h_edge_sel = h_at_edge[:, idx]  # (S, A)
    return h_edge_sel + (offset_b / eta_sel) ** beta_sel


def run_monte_carlo(
    bundle: TriangleBundle,
    sales_df: pd.DataFrame,
    fit: WeibullFitResult,
    unit_cost_fit: UnitCostFitResult,
    mileage_by_month: Optional[Dict[str, float]],
    warranty_months: int,
    mileage_limit_km: Optional[float],
    ages: List[int],
    n_simulations: int = 5000,
    seed: int = 42,
) -> MonteCarloResult:
    """Run the full Monte Carlo simulation for one system's claims."""
    rng = np.random.default_rng(seed)
    ages_arr = np.array(ages, dtype=float)
    n_ages = len(ages_arr)
    n_seg = len(fit.beta_t)

    param_samples = sample_parameters(fit, n_simulations, rng)
    beta_t_s = param_samples[:, :n_seg]
    eta_t_s = param_samples[:, n_seg : 2 * n_seg]
    if fit.has_mileage:
        beta_m_s = param_samples[:, 2 * n_seg]
        eta_m_s = param_samples[:, 2 * n_seg + 1]

    unit_cost_samples = sample_unit_cost_paths(unit_cost_fit, ages_arr, n_simulations, rng)

    sales_map = dict(zip(sales_df["销售月份"], sales_df["销量"]))
    max_ages = batch_max_ages(sales_df, warranty_months, mileage_limit_km, mileage_by_month)

    total_count = np.zeros(n_simulations)
    total_amount = np.zeros(n_simulations)

    for sm in bundle.count.index:
        sales = float(sales_map.get(sm, 0))
        if sales <= 0:
            continue
        mileage = float(mileage_by_month.get(sm, 0.0)) if mileage_by_month else 0.0
        max_age_b = max_ages.get(sm, warranty_months)
        last_obs = last_observed_age(bundle, sm)

        h_t = cumulative_hazard_time_matrix(ages_arr, fit.breakpoints, beta_t_s, eta_t_s)
        if fit.has_mileage and mileage > 0:
            m = ages_arr * mileage
            h = h_t + (m[None, :] / eta_m_s[:, None]) ** beta_m_s[:, None]
        else:
            h = h_t
        cum_rate_s = 1.0 - np.exp(-h)

        if last_obs is not None:
            from .utils import mis_col

            observed_cols = [mis_col(a) for a in bundle.mis_ages if a <= last_obs]
            actual_cum_last = float(bundle.count.loc[sm, observed_cols].fillna(0).sum())
            model_cum_last = float(fit.cum_rate(np.array([last_obs]), mileage)[0]) * sales
            offset = actual_cum_last - model_cum_last
        else:
            offset = 0.0

        pred_cum = cum_rate_s * sales + offset
        pred_cum = np.maximum.accumulate(pred_cum, axis=1)
        inc = np.diff(pred_cum, axis=1, prepend=0.0)

        lower_bound = last_obs if last_obs is not None else -1
        age_mask = (ages_arr > lower_bound) & (ages_arr <= max_age_b)
        if not np.any(age_mask):
            continue
        inc_future = inc[:, age_mask]
        uc_future = unit_cost_samples[:, age_mask]
        total_count += inc_future.sum(axis=1)
        total_amount += (inc_future * uc_future).sum(axis=1)

    # Representative confidence band for the pooled cumulative-rate curve,
    # using the sales-weighted average monthly mileage across batches.
    if mileage_by_month:
        weights = np.array([sales_map.get(sm, 0) for sm in mileage_by_month])
        vals = np.array([mileage_by_month[sm] for sm in mileage_by_month])
        ref_mileage = float(np.average(vals, weights=weights)) if weights.sum() > 0 else 0.0
    else:
        ref_mileage = 0.0

    h_t_ref = cumulative_hazard_time_matrix(ages_arr, fit.breakpoints, beta_t_s, eta_t_s)
    if fit.has_mileage and ref_mileage > 0:
        m_ref = ages_arr * ref_mileage
        h_ref = h_t_ref + (m_ref[None, :] / eta_m_s[:, None]) ** beta_m_s[:, None]
    else:
        h_ref = h_t_ref
    curve_samples = 1.0 - np.exp(-h_ref)
    band_lower = np.percentile(curve_samples, 5, axis=0)
    band_upper = np.percentile(curve_samples, 95, axis=0)
    band_median = np.percentile(curve_samples, 50, axis=0)

    return MonteCarloResult(
        total_count=summarize_distribution(total_count),
        total_amount=summarize_distribution(total_amount),
        band_ages=ages_arr,
        band_lower=band_lower,
        band_median=band_median,
        band_upper=band_upper,
        n_simulations=n_simulations,
    )
