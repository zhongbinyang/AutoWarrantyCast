"""Shared constants and small helper utilities used across the package.

This module intentionally contains no business logic beyond simple, pure
helper functions so that it can be imported from anywhere without creating
circular dependencies.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

#: The nine warranty claim subsystems supported by the multi-system import mode.
SYSTEM_LIST: List[str] = [
    "发动机系统",
    "底盘系统",
    "传动系统",
    "转向系统",
    "车身系统",
    "车身内饰",
    "车门天窗",
    "电气系统",
    "三电系统",
]

#: Industry empirical baseline priors for the 9 vehicle subsystems.
#: Used to regularize fitting and shrink estimates in few-shot/cold-start scenarios.
SYSTEM_PRIORS: dict[str, dict[str, float]] = {
    "发动机系统": {"beta_t": 2.2, "eta_t": 55.0, "unit_cost": 3200.0, "claim_rate": 0.08},
    "底盘系统": {"beta_t": 1.6, "eta_t": 60.0, "unit_cost": 1800.0, "claim_rate": 0.05},
    "传动系统": {"beta_t": 2.0, "eta_t": 58.0, "unit_cost": 2600.0, "claim_rate": 0.06},
    "转向系统": {"beta_t": 1.8, "eta_t": 62.0, "unit_cost": 1500.0, "claim_rate": 0.04},
    "车身系统": {"beta_t": 1.4, "eta_t": 70.0, "unit_cost": 1200.0, "claim_rate": 0.03},
    "车身内饰": {"beta_t": 1.3, "eta_t": 65.0, "unit_cost": 800.0, "claim_rate": 0.04},
    "车门天窗": {"beta_t": 1.5, "eta_t": 60.0, "unit_cost": 1100.0, "claim_rate": 0.04},
    "电气系统": {"beta_t": 1.7, "eta_t": 55.0, "unit_cost": 1400.0, "claim_rate": 0.07},
    "三电系统": {"beta_t": 1.9, "eta_t": 65.0, "unit_cost": 4500.0, "claim_rate": 0.05},
    "默认/通用": {"beta_t": 1.8, "eta_t": 60.0, "unit_cost": 2000.0, "claim_rate": 0.05},
}


def get_system_prior(system_name: str) -> dict[str, float]:
    """Retrieve the industry prior dictionary for a system, or general defaults."""
    for key, val in SYSTEM_PRIORS.items():
        if key in system_name:
            return val.copy()
    return SYSTEM_PRIORS["默认/通用"].copy()


#: Column name prefix used for development-period (age-in-months) columns.
MIS_PREFIX = "MIS"

#: Light-yellow fill (opaque ARGB) used to highlight forecast (vs. historical) cells.
FORECAST_FILL_HEX = "FFFFF2CC"

SALE_MONTH_COL = "销售月份"
SALES_COL = "销量"
MILEAGE_COL = "月均行驶里程"

_MIS_RE = re.compile(rf"^{MIS_PREFIX}(\d+)$", re.IGNORECASE)


def mis_col(age: int) -> str:
    """Build a development-period column name, e.g. ``mis_col(3) -> "MIS3"``."""
    return f"{MIS_PREFIX}{int(age)}"


def parse_mis_age(col: str) -> Optional[int]:
    """Parse a development-period column name into its integer age in months.

    Returns ``None`` if ``col`` does not match the ``MIS<n>`` pattern.
    """
    m = _MIS_RE.match(str(col).strip())
    return int(m.group(1)) if m else None


def sorted_mis_columns(columns) -> List[str]:
    """Return the subset of ``columns`` that look like MIS columns, sorted by age."""
    pairs = [(parse_mis_age(c), c) for c in columns]
    pairs = [p for p in pairs if p[0] is not None]
    pairs.sort(key=lambda p: p[0])
    return [c for _, c in pairs]


def sale_month_to_period(sale_month: str) -> pd.Period:
    """Parse a ``YYYY-MM`` string into a monthly :class:`pandas.Period`."""
    return pd.Period(str(sale_month), freq="M")


def months_between(sale_month: str, as_of_month: str) -> int:
    """Number of whole months between a sale month and a later reference month."""
    return (sale_month_to_period(as_of_month) - sale_month_to_period(sale_month)).n


def normalize_sale_month(value) -> str:
    """Normalize an arbitrary date-like value to a canonical ``YYYY-MM`` string."""
    if isinstance(value, pd.Period):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m")
    text = str(value).strip()
    # Accept YYYY-MM, YYYY/MM, YYYYMM
    text = text.replace("/", "-")
    if re.fullmatch(r"\d{6}", text):
        text = f"{text[:4]}-{text[4:]}"
    return pd.Period(text, freq="M").strftime("%Y-%m")


def next_n_months(start_month: str, n: int) -> List[str]:
    """Return ``n`` consecutive ``YYYY-MM`` month labels starting at ``start_month``."""
    p = sale_month_to_period(start_month)
    return [str(p + i) for i in range(n)]


def is_cumulative(triangle: pd.DataFrame, mis_columns: List[str]) -> bool:
    """Heuristically decide whether a triangle's values are cumulative.

    A triangle is treated as cumulative when, for the large majority of rows
    with at least two populated development periods, values are non-decreasing
    across increasing development age. Incremental triangles will typically
    fail this monotonicity check for a large share of rows.
    """
    if len(mis_columns) < 2:
        return False
    total_pairs = 0
    monotone_pairs = 0
    values = triangle[mis_columns].to_numpy(dtype=float)
    for row in values:
        valid = ~np.isnan(row)
        idx = np.where(valid)[0]
        if len(idx) < 2:
            continue
        seq = row[idx]
        diffs = np.diff(seq)
        total_pairs += len(diffs)
        monotone_pairs += int(np.sum(diffs >= -1e-6))
    if total_pairs == 0:
        return False
    return (monotone_pairs / total_pairs) > 0.9


def cumulative_to_incremental(triangle: pd.DataFrame, mis_columns: List[str]) -> pd.DataFrame:
    """Convert a cumulative triangle to incremental values along MIS columns."""
    out = triangle.copy()
    arr = out[mis_columns].to_numpy(dtype=float)
    inc = arr.copy()
    inc[:, 1:] = arr[:, 1:] - arr[:, :-1]
    out[mis_columns] = inc
    return out


def incremental_to_cumulative(triangle: pd.DataFrame, mis_columns: List[str]) -> pd.DataFrame:
    """Convert an incremental triangle to cumulative values along MIS columns."""
    out = triangle.copy()
    arr = out[mis_columns].fillna(0).to_numpy(dtype=float)
    out[mis_columns] = np.cumsum(arr, axis=1)
    return out


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Elementwise division that returns 0 where the denominator is (near) zero."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(denominator) > 1e-12, numerator / denominator, 0.0)
    return result
