"""Bivariate (time + mileage) competing-risk Weibull claim-count model.

The survival function used throughout this module is::

    S(t, m) = exp[-(t/eta_t)^beta_t - (m/eta_m)^beta_m]

so the cumulative failure probability is ``F(t, m) = 1 - S(t, m)``.

Because the input data is aggregated (claim-count triangles rather than
individual failure times), the model is fit on the *cumulative failure
rate* curve ``cum_count(batch, age) / sales(batch)`` rather than by
per-unit maximum likelihood. Two building blocks make this tractable:

* The time hazard ``H_t(t) = (t/eta_t)^beta_t`` is allowed to be
  **piecewise**: shape parameters beta_t are independent per segment while
  the cumulative hazard is forced to be continuous at every breakpoint by
  construction (each segment's local scale is solved from a weighted
  log-log regression anchored at the hazard value inherited from the
  previous segment).
* The mileage hazard ``H_m(m) = (m/eta_m)^beta_m`` is fit as a single
  segment (mileage breakpoints are rarely as informative as calendar-time
  breakpoints for warranty data, and keeping it single-segment keeps the
  joint refinement well identified).

Given piecewise time parameters as a good starting point, a joint
nonlinear least-squares refinement (`scipy.optimize.least_squares`) is run
over *all* parameters simultaneously against every observed
(batch, age) point weighted by that batch's sales volume, correctly
reflecting the competing-risk sum inside the exponential. The resulting
Jacobian is used to build an approximate parameter covariance matrix that
downstream Monte Carlo simulation samples from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from .data_io import TriangleBundle
from .utils import SALE_MONTH_COL, SALES_COL, safe_divide

MIN_BETA, MAX_BETA = 0.15, 8.0
MIN_ETA, MAX_ETA = 1e-3, 1e7


@dataclass
class ObservedPoint:
    """One (sale batch, development age) observation of cumulative failure rate."""

    sale_month: str
    age: int
    cum_rate: float
    weight: float
    mileage_per_month: float


@dataclass
class WeibullFitResult:
    """Fitted bivariate Weibull parameters plus diagnostics."""

    breakpoints: List[int]
    beta_t: List[float]
    eta_t: List[float]
    beta_m: Optional[float]
    eta_m: Optional[float]
    has_mileage: bool
    r2: float
    mape: float
    param_vector: np.ndarray
    param_names: List[str]
    cov: np.ndarray
    n_points: int

    def cumulative_hazard(self, t: np.ndarray, mileage_per_month: float = 0.0) -> np.ndarray:
        """Total cumulative hazard H(t) = H_t(t) + H_m(mileage_per_month * t)."""
        h_t = cumulative_hazard_time(t, self.breakpoints, self.beta_t, self.eta_t)
        if self.has_mileage and mileage_per_month > 0:
            m = np.asarray(t, dtype=float) * mileage_per_month
            h_m = (m / self.eta_m) ** self.beta_m
            return h_t + h_m
        return h_t

    def cum_rate(self, t: np.ndarray, mileage_per_month: float = 0.0) -> np.ndarray:
        """Predicted cumulative failure rate F(t, m) = 1 - exp(-H(t, m))."""
        h = self.cumulative_hazard(t, mileage_per_month)
        return 1.0 - np.exp(-h)


def _segment_edges(breakpoints: Sequence[int]) -> List[float]:
    return [0.0] + [float(b) for b in breakpoints]


def cumulative_hazard_time(
    t, breakpoints: Sequence[int], beta_t: Sequence[float], eta_t: Sequence[float]
) -> np.ndarray:
    """Piecewise-continuous time cumulative hazard, vectorized over ``t``."""
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    edges = _segment_edges(breakpoints)
    n_seg = len(beta_t)
    h_at_edge = [0.0]
    for k in range(n_seg - 1):
        seg_len = edges[k + 1] - edges[k]
        h_at_edge.append(h_at_edge[-1] + (seg_len / eta_t[k]) ** beta_t[k])
    idx = np.searchsorted(np.array(breakpoints, dtype=float), t_arr, side="right")
    idx = np.clip(idx, 0, n_seg - 1)
    offset = t_arr - np.array(edges)[idx]
    offset = np.clip(offset, 0.0, None)
    h = np.array(h_at_edge)[idx] + (offset / np.array(eta_t)[idx]) ** np.array(beta_t)[idx]
    return h


def _weighted_linreg(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    """Weighted least-squares slope/intercept for y = slope * x + intercept."""
    w_sum = w.sum()
    x_mean = np.sum(w * x) / w_sum
    y_mean = np.sum(w * y) / w_sum
    sxx = np.sum(w * (x - x_mean) ** 2)
    sxy = np.sum(w * (x - x_mean) * (y - y_mean))
    slope = sxy / sxx if sxx > 1e-12 else 0.0
    intercept = y_mean - slope * x_mean
    return slope, intercept


def collect_observed_points(
    bundle: TriangleBundle,
    sales_df: pd.DataFrame,
    mileage_by_month: Optional[Dict[str, float]],
) -> List[ObservedPoint]:
    """Flatten a triangle bundle's observed cells into cumulative-rate points."""
    sales_map = dict(zip(sales_df[SALE_MONTH_COL], sales_df[SALES_COL]))
    mis_cols = [f"MIS{a}" for a in bundle.mis_ages]
    cum_count = bundle.count[mis_cols].fillna(0).cumsum(axis=1)
    points: List[ObservedPoint] = []
    for sale_month in bundle.count.index:
        sales = sales_map.get(sale_month, 0)
        if sales <= 0:
            continue
        mileage = float(mileage_by_month.get(sale_month, 0.0)) if mileage_by_month else 0.0
        for age, col in zip(bundle.mis_ages, mis_cols):
            if not bundle.observed_mask.loc[sale_month, col]:
                continue
            rate = float(cum_count.loc[sale_month, col]) / sales
            rate = min(max(rate, 0.0), 0.999999)
            points.append(ObservedPoint(sale_month, age, rate, float(sales), mileage))
    return points


def auto_select_breakpoints(points: Sequence[ObservedPoint], max_breakpoints: int = 2) -> List[int]:
    """Curvature-based automatic breakpoint selection on the pooled hazard curve.

    Aggregates observed points by age (sales-weighted mean cumulative rate),
    converts to cumulative hazard, and looks for local maxima in the
    magnitude of its discrete second derivative -- points where the hazard
    behavior changes most.

    Includes a few-shot / cold-start safety threshold: if fewer than 15 total
    observations exist or the maximum observed age is less than 12 months,
    piecewise segmentation is aborted to avoid high variance on small samples.
    """
    if not points or len(points) < 15:
        return []
    df = pd.DataFrame([p.__dict__ for p in points])
    agg = df.groupby("age").apply(
        lambda g: np.average(g["cum_rate"], weights=g["weight"])
    )
    agg = agg.sort_index()
    if len(agg) < 8 or int(agg.index.max()) < 12:
        return []
    h = -np.log(1.0 - np.clip(agg.values, 0, 0.999999))
    d2 = np.gradient(np.gradient(h))
    peaks, _ = find_peaks(np.abs(d2))
    if len(peaks) == 0:
        return []
    ranked = sorted(peaks, key=lambda i: abs(d2[i]), reverse=True)
    ages = agg.index.to_numpy()
    chosen: List[int] = []
    for i in ranked:
        age = int(ages[i])
        if age <= 1 or age >= ages.max() - 1:
            continue
        if all(abs(age - c) >= 6 for c in chosen):
            chosen.append(age)
        if len(chosen) >= max_breakpoints:
            break
    return sorted(chosen)



def _fit_piecewise_time_regression(
    points: Sequence[ObservedPoint], breakpoints: Sequence[int]
) -> Tuple[List[float], List[float]]:
    """Initial closed-form estimate for piecewise time (beta_t, eta_t)."""
    edges = _segment_edges(breakpoints)
    n_seg = len(breakpoints) + 1
    ages = np.array([p.age for p in points], dtype=float)
    rates = np.array([p.cum_rate for p in points], dtype=float)
    weights = np.array([p.weight for p in points], dtype=float)
    h_obs = -np.log(1.0 - rates)

    beta_t: List[float] = []
    eta_t: List[float] = []
    h_prev_edge = 0.0
    for k in range(n_seg):
        lo = edges[k]
        hi = edges[k + 1] if k + 1 < len(edges) else np.inf
        seg_sel = (ages >= lo) & (ages < hi) if np.isfinite(hi) else (ages >= lo)
        if seg_sel.sum() < 2:
            beta_t.append(2.0)
            eta_t.append(max(float(np.median(ages)) if len(ages) else 24.0, 1.0))
            if np.isfinite(hi):
                h_prev_edge += (max(hi - lo, 1e-6) / eta_t[-1]) ** beta_t[-1]
            continue
        seg_age = ages[seg_sel] - lo
        seg_h = h_obs[seg_sel] - h_prev_edge
        valid = seg_h > 1e-6
        seg_age = seg_age[valid]
        seg_h = seg_h[valid]
        seg_w = weights[seg_sel][valid]
        if len(seg_age) < 2:
            beta = 2.0
            eta = max(float(np.median(ages)), 1.0)
        else:
            x = np.log(np.clip(seg_age, 1e-6, None))
            y = np.log(seg_h)
            slope, intercept = _weighted_linreg(x, y, seg_w)
            beta = float(np.clip(slope, MIN_BETA, MAX_BETA))
            eta = float(np.clip(np.exp(-intercept / beta) if beta > 1e-6 else 24.0, MIN_ETA, MAX_ETA))
        beta_t.append(beta)
        eta_t.append(eta)
        if np.isfinite(hi):
            h_prev_edge += ((hi - lo) / eta) ** beta
    return beta_t, eta_t


def fit_bivariate_weibull(
    points: Sequence[ObservedPoint],
    breakpoints: Sequence[int],
    has_mileage: bool,
    prior_beta_t: Optional[float] = None,
    prior_eta_t: Optional[float] = None,
    regularization_weight: float = 1.0,
) -> WeibullFitResult:
    """Fit the bivariate (or time-only) Weibull competing-risk model.

    In few-shot / short-history regimes, incorporates an adaptive L2/MAP prior
    penalty toward domain-informed engineering baselines (prior_beta_t, prior_eta_t),
    preventing non-physical parameter explosion. The penalty smoothly decays to
    zero as sample size increases, achieving a continuous transition to pure
    data-driven least squares.
    """
    if len(points) < 3:
        raise ValueError("有效观测点过少，无法拟合威布尔模型（至少需要 3 个观测点）")

    beta_t0, eta_t0 = _fit_piecewise_time_regression(points, breakpoints)
    n_seg = len(beta_t0)

    # Resolve default baseline priors if not supplied
    p_beta = float(prior_beta_t) if prior_beta_t is not None and prior_beta_t > 0 else 1.8
    p_eta = float(prior_eta_t) if prior_eta_t is not None and prior_eta_t > 0 else 60.0

    # Effective regularization penalty: stronger for small sample sizes, decays as points increase
    n_pts = len(points)
    eff_reg = float(regularization_weight) * max(0.0, 1.0 - n_pts / 24.0)

    if has_mileage:
        mileages = np.array([p.mileage_per_month * p.age for p in points])
        beta_m0 = 2.0
        pos_mileage = mileages[mileages > 0]
        eta_m0 = float(np.median(pos_mileage)) if len(pos_mileage) else 20000.0
        eta_m0 = max(eta_m0, MIN_ETA)
    else:
        beta_m0, eta_m0 = None, None

    x0 = list(beta_t0) + list(eta_t0)
    param_names = [f"beta_t{i}" for i in range(n_seg)] + [f"eta_t{i}" for i in range(n_seg)]
    lower = [MIN_BETA] * n_seg + [MIN_ETA] * n_seg
    upper = [MAX_BETA] * n_seg + [MAX_ETA] * n_seg
    if has_mileage:
        x0 += [beta_m0, eta_m0]
        param_names += ["beta_m", "eta_m"]
        lower += [MIN_BETA, MIN_ETA]
        upper += [MAX_BETA, MAX_ETA]

    ages = np.array([p.age for p in points], dtype=float)
    obs = np.array([p.cum_rate for p in points], dtype=float)
    weights = np.sqrt(np.array([p.weight for p in points], dtype=float))
    weights = weights / weights.mean()
    mileage_rates = np.array([p.mileage_per_month for p in points], dtype=float)

    def unpack(x):
        bt = x[:n_seg]
        et = x[n_seg : 2 * n_seg]
        if has_mileage:
            bm, em = x[2 * n_seg], x[2 * n_seg + 1]
        else:
            bm, em = None, None
        return bt, et, bm, em

    def residuals(x):
        bt, et, bm, em = unpack(x)
        h_t = cumulative_hazard_time(ages, breakpoints, bt, et)
        if has_mileage:
            m = mileage_rates * ages
            h_m = (m / em) ** bm
            h = h_t + h_m
        else:
            h = h_t
        pred = 1.0 - np.exp(-h)
        data_res = weights * (pred - obs)

        if eff_reg > 0:
            # MAP penalty terms: shape beta normalized by sigma=0.5, log-scale eta by sigma=0.4
            penalty_beta = [eff_reg * (b - p_beta) / 0.5 for b in bt]
            penalty_eta = [eff_reg * (np.log(max(e, 1e-4)) - np.log(p_eta)) / 0.4 for e in et]
            return np.concatenate([data_res, penalty_beta, penalty_eta])
        return data_res

    result = least_squares(
        residuals, x0=np.array(x0), bounds=(lower, upper), method="trf", max_nfev=5000
    )
    bt, et, bm, em = unpack(result.x)

    jac = result.jac
    resid = result.fun
    dof = max(len(resid) - len(result.x), 1)
    s_sq = float(np.sum(resid**2) / dof)

    # Tikhonov regularization on J^T J prevents singular / explosive parameter covariance on small samples
    jtj = jac.T @ jac
    diag_scale = float(np.trace(jtj) / max(len(result.x), 1)) if len(result.x) > 0 else 1.0
    gamma = max(1e-4 * diag_scale, 1e-6)
    jtj_reg = jtj + gamma * np.eye(len(result.x))
    try:
        cov = s_sq * np.linalg.inv(jtj_reg)
    except np.linalg.LinAlgError:
        cov = np.eye(len(result.x)) * 1e-6


    pred_final = 1.0 - np.exp(
        -cumulative_hazard_time(ages, breakpoints, bt, et)
        - (
            ((mileage_rates * ages) / em) ** bm
            if has_mileage
            else np.zeros_like(ages)
        )
    )
    ss_res = float(np.sum((obs - pred_final) ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2)) or 1e-9
    r2 = 1.0 - ss_res / ss_tot
    mape = float(np.mean(np.abs(safe_divide(obs - pred_final, obs))[obs > 1e-6])) if np.any(obs > 1e-6) else 0.0

    return WeibullFitResult(
        breakpoints=list(breakpoints),
        beta_t=list(bt),
        eta_t=list(et),
        beta_m=bm,
        eta_m=em,
        has_mileage=has_mileage,
        r2=r2,
        mape=mape,
        param_vector=result.x,
        param_names=param_names,
        cov=cov,
        n_points=len(points),
    )


def sample_parameters(fit: WeibullFitResult, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n_samples`` parameter vectors from the fitted asymptotic covariance.

    Draws are truncated to stay within the same physical bounds used during
    fitting (positive beta/eta), simply by rejecting/clamping negative draws.
    """
    mean = fit.param_vector
    cov = fit.cov
    try:
        draws = rng.multivariate_normal(mean, cov, size=n_samples, method="eigh")
    except np.linalg.LinAlgError:
        draws = np.tile(mean, (n_samples, 1))
    n_seg = len(fit.beta_t)
    draws[:, :n_seg] = np.clip(draws[:, :n_seg], MIN_BETA, MAX_BETA)
    draws[:, n_seg : 2 * n_seg] = np.clip(draws[:, n_seg : 2 * n_seg], MIN_ETA, MAX_ETA)
    if fit.has_mileage:
        draws[:, 2 * n_seg] = np.clip(draws[:, 2 * n_seg], MIN_BETA, MAX_BETA)
        draws[:, 2 * n_seg + 1] = np.clip(draws[:, 2 * n_seg + 1], MIN_ETA, MAX_ETA)
    return draws


def fit_bayesian_weibull(
    points: Sequence[ObservedPoint],
    breakpoints: Sequence[int],
    has_mileage: bool,
    prior_beta_t: Optional[float] = None,
    prior_eta_t: Optional[float] = None,
    draws: int = 800,
    tune: int = 500,
    chains: int = 2,
) -> WeibullFitResult:
    """Bayesian refinement of the Weibull fit using PyMC (if installed).

    Priors are centered on the maximum-likelihood estimate from
    :func:`fit_bivariate_weibull` (LogNormal for the strictly-positive beta
    and eta parameters). The returned :class:`WeibullFitResult` reuses the
    posterior mean as its point estimate and the posterior sample covariance
    as ``cov`` -- so downstream Monte Carlo sampling (which draws from
    ``cov``) reflects genuine Bayesian parameter uncertainty rather than the
    asymptotic normal approximation used by the plain MLE fit.

    Raises ImportError if PyMC is not installed; callers should catch this
    and fall back to :func:`fit_bivariate_weibull`.
    """
    import pymc as pm
    import pytensor.tensor as pt

    mle_fit = fit_bivariate_weibull(
        points,
        breakpoints,
        has_mileage,
        prior_beta_t=prior_beta_t,
        prior_eta_t=prior_eta_t,
    )
    n_seg = len(mle_fit.beta_t)

    ages = np.array([p.age for p in points], dtype=float)
    obs = np.array([p.cum_rate for p in points], dtype=float)
    weights = np.array([p.weight for p in points], dtype=float)
    weights = weights / weights.mean()
    mileage_rates = np.array([p.mileage_per_month for p in points], dtype=float)

    # Segment index and within-segment offset per observation are fixed
    # (they only depend on known ages/breakpoints), so they are precomputed
    # as plain NumPy arrays; only beta_t/eta_t/beta_m/eta_m remain symbolic.
    edges = _segment_edges(breakpoints)
    seg_idx = np.clip(
        np.searchsorted(np.array(breakpoints, dtype=float), ages, side="right"), 0, n_seg - 1
    )
    seg_offset = np.clip(ages - np.array(edges)[seg_idx], 0.0, None)

    def hazard_time_symbolic(beta_t, eta_t):
        h_at_edge = [pt.as_tensor_variable(0.0)]
        for k in range(n_seg - 1):
            seg_len = edges[k + 1] - edges[k]
            h_at_edge.append(h_at_edge[-1] + (seg_len / eta_t[k]) ** beta_t[k])
        h_at_edge_stacked = pt.stack(h_at_edge)
        beta_sel = beta_t[seg_idx]
        eta_sel = eta_t[seg_idx]
        h_edge_sel = h_at_edge_stacked[seg_idx]
        return h_edge_sel + (seg_offset / eta_sel) ** beta_sel

    with pm.Model():
        beta_t = pm.LogNormal("beta_t", mu=np.log(mle_fit.beta_t), sigma=0.3, shape=n_seg)
        eta_t = pm.LogNormal("eta_t", mu=np.log(mle_fit.eta_t), sigma=0.3, shape=n_seg)
        h_total = hazard_time_symbolic(beta_t, eta_t)

        if has_mileage:
            beta_m = pm.LogNormal("beta_m", mu=np.log(mle_fit.beta_m), sigma=0.3)
            eta_m = pm.LogNormal("eta_m", mu=np.log(mle_fit.eta_m), sigma=0.3)
            m = mileage_rates * ages
            h_total = h_total + (m / eta_m) ** beta_m

        sigma = pm.HalfNormal("sigma", sigma=0.05)
        pred = 1.0 - pm.math.exp(-h_total)
        pm.Normal("obs", mu=pred, sigma=sigma / pt.sqrt(weights), observed=obs)

        trace = pm.sample(draws=draws, tune=tune, chains=chains, progressbar=False, target_accept=0.9)

    names = ["beta_t", "eta_t"] + (["beta_m", "eta_m"] if has_mileage else [])
    flat_samples = []
    for name in names:
        var = trace.posterior[name].to_numpy().reshape(chains * draws, -1)
        flat_samples.append(var)
    combined = np.concatenate(flat_samples, axis=1)
    post_mean = combined.mean(axis=0)
    post_cov = np.cov(combined, rowvar=False)

    bt = post_mean[:n_seg]
    et = post_mean[n_seg : 2 * n_seg]
    bm = post_mean[2 * n_seg] if has_mileage else None
    em = post_mean[2 * n_seg + 1] if has_mileage else None

    return WeibullFitResult(
        breakpoints=list(breakpoints),
        beta_t=list(bt),
        eta_t=list(et),
        beta_m=bm,
        eta_m=em,
        has_mileage=has_mileage,
        r2=mle_fit.r2,
        mape=mle_fit.mape,
        param_vector=post_mean,
        param_names=mle_fit.param_names,
        cov=post_cov,
        n_points=mle_fit.n_points,
    )


def unpack_param_vector(fit: WeibullFitResult, x: np.ndarray):
    """Split a raw parameter vector (as produced by :func:`sample_parameters`)."""
    n_seg = len(fit.beta_t)
    bt = x[:n_seg]
    et = x[n_seg : 2 * n_seg]
    if fit.has_mileage:
        bm, em = x[2 * n_seg], x[2 * n_seg + 1]
    else:
        bm, em = None, None
    return bt, et, bm, em
