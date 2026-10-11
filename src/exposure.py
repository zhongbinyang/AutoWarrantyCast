"""In-force (still-under-warranty) vehicle count calculations.

A vehicle exits warranty coverage at the *earlier* of the time limit and the
mileage limit ("以先到的时间或里程为准"). This module computes, for every
sale batch and development age, whether that batch is still in force and how
many vehicles remain in force.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .utils import SALE_MONTH_COL, SALES_COL


def max_in_force_age(
    warranty_months: int,
    mileage_limit_km: Optional[float],
    monthly_mileage_km: Optional[float],
) -> int:
    """Return the last development age (inclusive) a batch remains in force."""
    max_age = int(warranty_months)
    if mileage_limit_km is not None and monthly_mileage_km is not None and monthly_mileage_km > 0:
        mileage_age = int(np.floor(mileage_limit_km / monthly_mileage_km))
        max_age = min(max_age, mileage_age)
    return max(max_age, 0)


def build_in_force_table(
    sales_df: pd.DataFrame,
    ages: List[int],
    warranty_months: int,
    mileage_limit_km: Optional[float],
    mileage_by_month: Optional[Dict[str, float]],
) -> pd.DataFrame:
    """Build a (sale_month x age) table of in-force vehicle counts.

    Each cell holds the number of vehicles from that sale batch still under
    warranty at the given development age (0 once the batch has exited on
    either the time or mileage limit).
    """
    sale_months = list(sales_df[SALE_MONTH_COL])
    sales_map = dict(zip(sales_df[SALE_MONTH_COL], sales_df[SALES_COL]))
    table = pd.DataFrame(index=sale_months, columns=ages, dtype=float)
    for sm in sale_months:
        sales = float(sales_map.get(sm, 0))
        monthly_mileage = None
        if mileage_by_month is not None:
            monthly_mileage = mileage_by_month.get(sm)
        max_age = max_in_force_age(warranty_months, mileage_limit_km, monthly_mileage)
        for age in ages:
            table.loc[sm, age] = sales if age <= max_age else 0.0
    return table


def batch_max_ages(
    sales_df: pd.DataFrame,
    warranty_months: int,
    mileage_limit_km: Optional[float],
    mileage_by_month: Optional[Dict[str, float]],
) -> Dict[str, int]:
    """Return, per sale month, the last in-force development age."""
    out: Dict[str, int] = {}
    for sm in sales_df[SALE_MONTH_COL]:
        monthly_mileage = mileage_by_month.get(sm) if mileage_by_month else None
        out[sm] = max_in_force_age(warranty_months, mileage_limit_km, monthly_mileage)
    return out
