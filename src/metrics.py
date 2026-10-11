"""Batch-level and cross-batch monthly metric calculations.

All formulas follow section 四.5 of the specification exactly:

* Batch cumulative failure rate = batch total predicted claim count / sales.
* Batch unit cost per vehicle   = batch total predicted claim amount / sales.
* Monthly average failure rate (age j) = sum of predicted counts at age j
  across batches / total in-force vehicles at age j.
* Monthly unit repair cost (age j) = sum of predicted amounts at age j
  across batches / sum of predicted counts at age j.
* Monthly unit cost per vehicle (age j) = sum of predicted amounts at age j
  across batches / total in-force vehicles at age j.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .utils import SALE_MONTH_COL, SALES_COL, mis_col, safe_divide


def compute_batch_summary(
    count_filled: pd.DataFrame,
    amount_filled: pd.DataFrame,
    sales_df: pd.DataFrame,
    ages: List[int],
    max_ages: dict,
) -> pd.DataFrame:
    """Per sale-month-batch cumulative failure rate and unit cost."""
    sales_map = dict(zip(sales_df[SALE_MONTH_COL], sales_df[SALES_COL]))
    rows = []
    for sm in count_filled.index:
        sales = float(sales_map.get(sm, 0))
        max_age = max_ages.get(sm, max(ages))
        cols = [mis_col(a) for a in ages if a <= max_age]
        total_count = float(count_filled.loc[sm, cols].fillna(0).sum())
        total_amount = float(amount_filled.loc[sm, cols].fillna(0).sum())
        rows.append(
            {
                SALE_MONTH_COL: sm,
                SALES_COL: sales,
                "累积故障率": (total_count / sales) if sales > 0 else 0.0,
                "单台费用": (total_amount / sales) if sales > 0 else 0.0,
                "预测总台次": total_count,
                "预测总金额": total_amount,
            }
        )
    return pd.DataFrame(rows).sort_values(SALE_MONTH_COL).reset_index(drop=True)


def compute_monthly_cross_batch(
    count_filled: pd.DataFrame,
    amount_filled: pd.DataFrame,
    in_force_table: pd.DataFrame,
    ages: List[int],
) -> pd.DataFrame:
    """Cross-batch aggregation by development age (vehicle age in months)."""
    rows = []
    for age in ages:
        col = mis_col(age)
        count_sum = float(count_filled[col].fillna(0).sum()) if col in count_filled.columns else 0.0
        amount_sum = float(amount_filled[col].fillna(0).sum()) if col in amount_filled.columns else 0.0
        in_force = float(in_force_table[age].sum()) if age in in_force_table.columns else 0.0
        rows.append(
            {
                "车龄_月": age,
                "在保车辆数": in_force,
                "预测台次": count_sum,
                "预测金额": amount_sum,
                "月度平均故障率": safe_divide(np.array([count_sum]), np.array([in_force]))[0],
                "月度单次维修成本": safe_divide(np.array([amount_sum]), np.array([count_sum]))[0],
                "月度单台费用": safe_divide(np.array([amount_sum]), np.array([in_force]))[0],
            }
        )
    return pd.DataFrame(rows)


def compute_validation_metrics(
    predicted_df: pd.DataFrame,
    forecast_mask: pd.DataFrame,
    actual_df: pd.DataFrame,
) -> Optional[dict]:
    """Compare user-supplied real future values against forecast cells.

    ``actual_df`` must share the same index (sale month) and MIS columns as
    ``predicted_df``. Only cells marked as forecast in ``forecast_mask`` and
    present (non-null) in ``actual_df`` are compared. Returns ``None`` if no
    overlapping cells are found.
    """
    common_index = predicted_df.index.intersection(actual_df.index)
    common_cols = [c for c in predicted_df.columns if c in actual_df.columns]
    if not len(common_index) or not common_cols:
        return None

    pred = predicted_df.loc[common_index, common_cols]
    mask = forecast_mask.loc[common_index, common_cols]
    actual = actual_df.loc[common_index, common_cols]

    valid = mask & actual.notna() & pred.notna()
    if not valid.to_numpy().any():
        return None

    p = pred.to_numpy()[valid.to_numpy()]
    a = actual.to_numpy()[valid.to_numpy()]
    mae = float(np.mean(np.abs(p - a)))
    nonzero = np.abs(a) > 1e-9
    mape = float(np.mean(np.abs((p[nonzero] - a[nonzero]) / a[nonzero]))) if nonzero.any() else float("nan")
    return {"MAE": mae, "MAPE": mape, "n_cells": int(valid.to_numpy().sum())}
