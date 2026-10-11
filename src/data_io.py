"""Data import, validation and normalization for the warranty cost system.

Handles the three input file families described in the specification:

1. Monthly new-vehicle sales (CSV/Excel, columns ``销售月份`` / ``销量``).
2. Claim triangles (Excel, one sheet per system, rows = sale month, columns
   = development period ``MIS0, MIS1, ...``) for both claim count and claim
   amount.
3. Optional monthly mileage input (CSV, columns ``销售月份`` / ``月均行驶里程``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .utils import (
    MILEAGE_COL,
    SALE_MONTH_COL,
    SALES_COL,
    cumulative_to_incremental,
    is_cumulative,
    normalize_sale_month,
    sorted_mis_columns,
)

FileLike = Union[str, bytes, "object"]


@dataclass
class ValidationIssue:
    """A single human-readable validation warning or error."""

    level: str  # "warning" | "error" | "info"
    message: str


@dataclass
class TriangleBundle:
    """A fully validated, normalized claim triangle for one system.

    Attributes:
        name: System name (or "单系统" for single-system mode).
        count: Incremental claim-count triangle, index=sale month, columns=MIS ages.
        amount: Incremental claim-amount triangle, same shape as ``count``.
        observed_mask: Boolean frame, True where the cell is historical/actual.
        evaluation_month: The inferred "as of" month of the data snapshot.
        mis_ages: Sorted list of development-period ages (in months) present.
        issues: Validation messages collected while preparing this bundle.
    """

    name: str
    count: pd.DataFrame
    amount: pd.DataFrame
    observed_mask: pd.DataFrame
    evaluation_month: str
    mis_ages: List[int]
    issues: List[ValidationIssue] = field(default_factory=list)


def _read_table(file: FileLike, **kwargs) -> pd.DataFrame:
    """Read a CSV or Excel file-like object into a DataFrame."""
    name = getattr(file, "name", str(file))
    has_seek = hasattr(file, "seek")

    if str(name).lower().endswith(".csv"):
        try:
            return pd.read_csv(file, **kwargs)
        except Exception:
            if has_seek:
                file.seek(0)
            raise

    try:
        return pd.read_excel(file, **kwargs)
    except Exception as e:
        # Fallback to CSV if reading Excel fails (e.g. if a CSV was renamed to .xlsx)
        if has_seek:
            file.seek(0)
        try:
            return pd.read_csv(file, **kwargs)
        except Exception:
            # Raise the original Excel error if CSV fallback also fails
            raise e


def load_sales(file: FileLike) -> pd.DataFrame:
    """Load and validate the monthly sales file.

    Returns a DataFrame sorted by sale month with columns ``销售月份`` (str)
    and ``销量`` (int).
    """
    df = _read_table(file)
    missing = {SALE_MONTH_COL, SALES_COL} - set(df.columns)
    if missing:
        raise ValueError(f"销量文件缺少必需列: {missing}")
    df = df[[SALE_MONTH_COL, SALES_COL]].copy()
    df[SALE_MONTH_COL] = df[SALE_MONTH_COL].map(normalize_sale_month)
    df[SALES_COL] = pd.to_numeric(df[SALES_COL], errors="coerce").fillna(0).astype(int)
    df = df.groupby(SALE_MONTH_COL, as_index=False)[SALES_COL].sum()
    df = df.sort_values(SALE_MONTH_COL).reset_index(drop=True)
    return df


def load_mileage(file: FileLike) -> pd.DataFrame:
    """Load and validate the optional monthly-mileage-by-batch file."""
    df = _read_table(file)
    missing = {SALE_MONTH_COL, MILEAGE_COL} - set(df.columns)
    if missing:
        raise ValueError(f"月均行驶里程文件缺少必需列: {missing}")
    df = df[[SALE_MONTH_COL, MILEAGE_COL]].copy()
    df[SALE_MONTH_COL] = df[SALE_MONTH_COL].map(normalize_sale_month)
    df[MILEAGE_COL] = pd.to_numeric(df[MILEAGE_COL], errors="coerce")
    return df.sort_values(SALE_MONTH_COL).reset_index(drop=True)


def load_triangle_sheet(file: FileLike, sheet_name: Optional[Union[str, int]] = 0) -> pd.DataFrame:
    """Load a single triangle sheet into a DataFrame indexed by sale month.

    The sheet must have a ``销售月份`` column plus one or more ``MIS<n>``
    development-period columns.
    """
    has_seek = hasattr(file, "seek")
    try:
        df = pd.read_excel(file, sheet_name=sheet_name)
    except Exception as e:
        # Fallback to CSV if reading Excel fails (e.g. if a CSV was renamed to .xlsx)
        if has_seek:
            file.seek(0)
        try:
            df = pd.read_csv(file)
        except Exception:
            raise e

    if SALE_MONTH_COL not in df.columns:
        raise ValueError(f"三角形文件缺少 '{SALE_MONTH_COL}' 列")
    df[SALE_MONTH_COL] = df[SALE_MONTH_COL].map(normalize_sale_month)
    mis_cols = sorted_mis_columns(df.columns)
    if not mis_cols:
        raise ValueError("三角形文件未找到任何 MIS<n> 发展期列")
    df = df[[SALE_MONTH_COL] + mis_cols].copy()
    df = df.groupby(SALE_MONTH_COL, as_index=True)[mis_cols].sum(min_count=1)
    return df


def list_excel_sheets(file: FileLike) -> List[str]:
    """Return the sheet names contained in an uploaded Excel file."""
    xls = pd.ExcelFile(file)
    return list(xls.sheet_names)


def _infer_evaluation_month(df: pd.DataFrame, mis_cols: List[str]) -> str:
    """Infer the data snapshot ("as of") month from the populated cells."""
    ages = [int(c.replace("MIS", "")) if isinstance(c, str) else c for c in mis_cols]
    best_month = None
    for sale_month, row in df.iterrows():
        valid_idx = [i for i, c in enumerate(mis_cols) if pd.notna(row[c])]
        if not valid_idx:
            continue
        max_age = ages[max(valid_idx)]
        candidate = pd.Period(sale_month, freq="M") + max_age
        if best_month is None or candidate > best_month:
            best_month = candidate
    if best_month is None:
        raise ValueError("三角形数据全部为空，无法推断数据截止月份")
    return str(best_month)


def build_observed_mask(
    sale_months: List[str], mis_cols: List[str], evaluation_month: str
) -> pd.DataFrame:
    """Build a boolean mask: True where the (batch, age) cell is historical."""
    ages = [int(c.replace("MIS", "")) for c in mis_cols]
    eval_p = pd.Period(evaluation_month, freq="M")
    mask = pd.DataFrame(index=sale_months, columns=mis_cols, dtype=bool)
    for sm in sale_months:
        sm_p = pd.Period(sm, freq="M")
        max_observed_age = (eval_p - sm_p).n
        mask.loc[sm] = [age <= max_observed_age for age in ages]
    return mask


def align_to_sales(df: pd.DataFrame, sales_months: List[str], label: str) -> pd.DataFrame:
    """Reindex a triangle to the full set of sale months, flagging mismatches."""
    extra = [m for m in df.index if m not in sales_months]
    missing = [m for m in sales_months if m not in df.index]
    issues = []
    if extra:
        issues.append(
            ValidationIssue("warning", f"[{label}] 三角形中存在销量表未覆盖的销售月份: {extra}")
        )
    if missing:
        issues.append(
            ValidationIssue("warning", f"[{label}] 销量表中的以下销售月份在三角形中缺失，将按 0 处理: {missing}")
        )
    out = df.reindex(sales_months)
    return out, issues


def prepare_triangle_bundle(
    name: str,
    count_df: pd.DataFrame,
    amount_df: pd.DataFrame,
    sales_df: pd.DataFrame,
) -> TriangleBundle:
    """Validate, align and normalize a (count, amount) triangle pair.

    Detects whether each triangle is incremental or cumulative and converts
    cumulative triangles to incremental form. Aligns both triangles to the
    sales table's sale months and infers the evaluation ("as of") month.
    """
    issues: List[ValidationIssue] = []
    sales_months = list(sales_df[SALE_MONTH_COL])

    mis_cols = sorted_mis_columns(set(count_df.columns) | set(amount_df.columns))
    count_df = count_df.reindex(columns=mis_cols)
    amount_df = amount_df.reindex(columns=mis_cols)

    count_df, count_issues = align_to_sales(count_df, sales_months, f"{name}-台次")
    amount_df, amount_issues = align_to_sales(amount_df, sales_months, f"{name}-金额")
    issues.extend(count_issues)
    issues.extend(amount_issues)

    if is_cumulative(count_df, mis_cols):
        issues.append(ValidationIssue("info", f"[{name}-台次] 检测到累积口径，已自动转换为增量口径"))
        count_df = cumulative_to_incremental(count_df, mis_cols)
    if is_cumulative(amount_df, mis_cols):
        issues.append(ValidationIssue("info", f"[{name}-金额] 检测到累积口径，已自动转换为增量口径"))
        amount_df = cumulative_to_incremental(amount_df, mis_cols)

    n_missing_count = int(count_df[mis_cols].isna().sum().sum())
    n_missing_amount = int(amount_df[mis_cols].isna().sum().sum())

    evaluation_month = _infer_evaluation_month(count_df, mis_cols)
    observed_mask = build_observed_mask(sales_months, mis_cols, evaluation_month)

    # Any NaN inside the observed region is a genuine data gap -> fill with 0
    # and warn; NaN outside the observed region is expected (future, to be
    # forecast) and is left as NaN.
    for df, dname, n_missing in (
        (count_df, "台次", n_missing_count),
        (amount_df, "金额", n_missing_amount),
    ):
        gap_mask = df[mis_cols].isna() & observed_mask
        n_gaps = int(gap_mask.to_numpy().sum())
        if n_gaps > 0:
            issues.append(
                ValidationIssue(
                    "warning",
                    f"[{name}-{dname}] 历史区间内发现 {n_gaps} 个缺失值单元格，已按 0 填充",
                )
            )
            df[mis_cols] = df[mis_cols].where(~gap_mask, 0.0)

    mis_ages = [int(c.replace("MIS", "")) for c in mis_cols]
    return TriangleBundle(
        name=name,
        count=count_df,
        amount=amount_df,
        observed_mask=observed_mask,
        evaluation_month=evaluation_month,
        mis_ages=mis_ages,
        issues=issues,
    )
