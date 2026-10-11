"""End-to-end orchestration: raw triangle -> fitted models -> forecast -> metrics.

This module has no Streamlit dependency so it can be unit-tested or reused
from a script; :mod:`app` is a thin UI layer on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from . import claims_model, cost_model, exposure, metrics, monte_carlo, utils, weibull_model
from .data_io import TriangleBundle


@dataclass
class SystemResult:
    """Everything computed for a single claim system, ready for display."""

    name: str
    bundle: TriangleBundle
    fit: "weibull_model.WeibullFitResult"
    unit_cost_fit: cost_model.UnitCostFitResult
    ages: List[int]
    count_filled: pd.DataFrame
    count_forecast_mask: pd.DataFrame
    amount_filled: pd.DataFrame
    amount_forecast_mask: pd.DataFrame
    batch_summary: pd.DataFrame
    monthly_summary: pd.DataFrame
    mc_result: monte_carlo.MonteCarloResult
    hist_unit_cost: pd.DataFrame
    has_mileage: bool


def run_system_pipeline(
    name: str,
    bundle: TriangleBundle,
    sales_df: pd.DataFrame,
    mileage_by_month: Optional[Dict[str, float]],
    warranty_months: int,
    mileage_limit_km: Optional[float],
    breakpoints: List[int],
    use_bayesian: bool,
    cost_method: str,
    amount_method: str,
    n_simulations: int,
    enable_few_shot: bool = True,
    prior_config: Optional[dict] = None,
    regularization_weight: float = 1.0,
    seed: int = 42,
) -> SystemResult:
    """Run the full model pipeline for one system and return all results."""
    has_mileage = bool(mileage_by_month) and any(v > 0 for v in mileage_by_month.values())

    # Resolve system empirical baseline priors for small-sample regularization
    prior = prior_config if prior_config is not None else utils.get_system_prior(name)
    p_beta = float(prior.get("beta_t", 1.8))
    p_eta = float(prior.get("eta_t", 60.0))
    p_unit_cost = float(prior.get("unit_cost", 2000.0))
    p_claim_rate = float(prior.get("claim_rate", 0.05))
    reg_w = float(regularization_weight) if enable_few_shot else 0.0

    points = weibull_model.collect_observed_points(bundle, sales_df, mileage_by_month)

    if use_bayesian:
        try:
            fit = weibull_model.fit_bayesian_weibull(
                points, breakpoints, has_mileage, prior_beta_t=p_beta, prior_eta_t=p_eta
            )
        except ImportError:
            fit = weibull_model.fit_bivariate_weibull(
                points,
                breakpoints,
                has_mileage,
                prior_beta_t=p_beta,
                prior_eta_t=p_eta,
                regularization_weight=reg_w,
            )
    else:
        fit = weibull_model.fit_bivariate_weibull(
            points,
            breakpoints,
            has_mileage,
            prior_beta_t=p_beta,
            prior_eta_t=p_eta,
            regularization_weight=reg_w,
        )

    hist_unit_cost = cost_model.compute_historical_unit_cost(bundle, bundle)
    unit_cost_fit = cost_model.fit_unit_cost_model(
        hist_unit_cost, method=cost_method, prior_unit_cost=p_unit_cost
    )

    count_filled, count_fmask, ages = claims_model.forecast_counts_via_weibull(
        bundle, sales_df, fit, mileage_by_month, warranty_months, mileage_limit_km
    )
    amount_filled, amount_fmask = claims_model.forecast_amount(
        bundle,
        count_filled,
        unit_cost_fit,
        ages,
        method=amount_method,
        sales_df=sales_df,
        prior_expected_loss_per_vehicle=p_claim_rate * p_unit_cost,
    )


    max_ages = exposure.batch_max_ages(sales_df, warranty_months, mileage_limit_km, mileage_by_month)
    batch_summary = metrics.compute_batch_summary(count_filled, amount_filled, sales_df, ages, max_ages)

    in_force_table = exposure.build_in_force_table(
        sales_df, ages, warranty_months, mileage_limit_km, mileage_by_month
    )
    monthly_summary = metrics.compute_monthly_cross_batch(count_filled, amount_filled, in_force_table, ages)

    mc_result = monte_carlo.run_monte_carlo(
        bundle,
        sales_df,
        fit,
        unit_cost_fit,
        mileage_by_month,
        warranty_months,
        mileage_limit_km,
        ages,
        n_simulations=n_simulations,
        seed=seed,
    )

    return SystemResult(
        name=name,
        bundle=bundle,
        fit=fit,
        unit_cost_fit=unit_cost_fit,
        ages=ages,
        count_filled=count_filled,
        count_forecast_mask=count_fmask,
        amount_filled=amount_filled,
        amount_forecast_mask=amount_fmask,
        batch_summary=batch_summary,
        monthly_summary=monthly_summary,
        mc_result=mc_result,
        hist_unit_cost=hist_unit_cost,
        has_mileage=has_mileage,
    )
