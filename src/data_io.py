"""Unified Excel import, sheet mapping, validation and triangle normalization.

One workbook contains sales, historical claim triangles, optional mileage
and optional actual incremental triangles for forecast validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
import re
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .utils import (
    MILEAGE_COL,
    SALE_MONTH_COL,
    SALES_COL,
    SYSTEM_LIST,
    cumulative_to_incremental,
    is_cumulative,
    normalize_sale_month,
    parse_mis_age,
    sorted_mis_columns,
)

FileLike = Union[str, bytes, "object"]

SHEET_PURPOSES = ["忽略", "销量", "历史索赔", "月均里程", "验证数据"]
CLAIM_METRICS = ["台次", "金额"]


@dataclass(frozen=True)
class SheetMapping:
    sheet_name: str
    purpose: str = "忽略"
    system: str = ""
    metric: str = ""


@dataclass
class WorkbookImport:
    sales: pd.DataFrame
    bundles: Dict[str, "TriangleBundle"]
    mileage: Optional[pd.DataFrame]
    actuals: Dict[str, Dict[str, pd.DataFrame]]
    issues: List["ValidationIssue"]


def read_workbook(file: FileLike) -> Dict[str, pd.DataFrame]:
    """Read all sheets once, preserving duplicate headers for validation."""
    if isinstance(file, bytes):
        file = BytesIO(file)
    elif hasattr(file, "seek"):
        file.seek(0)
    with pd.ExcelFile(file, engine="openpyxl") as workbook:
        sheets = {}
        for name in workbook.sheet_names:
            raw = pd.read_excel(workbook, sheet_name=name, header=None)
            if raw.empty:
                sheets[name] = pd.DataFrame()
                continue
            df = raw.iloc[1:].copy()
            df.columns = [str(c).strip() if pd.notna(c) else "" for c in raw.iloc[0]]
            sheets[name] = df.reset_index(drop=True)
    return sheets


def suggest_sheet_mappings(sheets: Dict[str, pd.DataFrame]) -> List[SheetMapping]:
    """Use explicit sheet names for claims; never infer metrics from values."""
    mappings = []
    for name, df in sheets.items():
        compact = re.sub(r"[\s_\-—－（）()·.]+", "", name)
        columns = {str(c).strip() for c in df.columns}
        mapping = SheetMapping(name)
        if "说明" not in compact:
            systems = [s for s in SYSTEM_LIST if s in compact]
            metrics = [m for m in CLAIM_METRICS if m in compact]
            if len(systems) == 1 and len(metrics) == 1:
                purpose = "验证数据" if any(w in compact for w in ("验证", "实际", "真实")) else "历史索赔"
                mapping = SheetMapping(name, purpose, systems[0], metrics[0])
            elif not systems and not metrics:
                is_sales = compact in ("销量", "月度销量") or {SALE_MONTH_COL, SALES_COL} <= columns
                is_mileage = compact in ("月均里程", "月均行驶里程") or {SALE_MONTH_COL, MILEAGE_COL} <= columns
                if is_sales and not is_mileage:
                    mapping = SheetMapping(name, "销量")
                elif is_mileage and not is_sales:
                    mapping = SheetMapping(name, "月均里程")
        mappings.append(mapping)
    return mappings


def _checked_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.replace(r"^\s*$", np.nan, regex=True).dropna(how="all")
    df = df.loc[:, ~((df.columns == "") & df.isna().all().to_numpy())]
    if df.columns.duplicated().any():
        raise ValueError("存在重复列名")
    if df.empty:
        raise ValueError("数据为空")
    if SALE_MONTH_COL not in df:
        raise ValueError(f"缺少 '{SALE_MONTH_COL}' 列")
    months = []
    for row, value in df[SALE_MONTH_COL].items():
        try:
            if pd.isna(value):
                raise ValueError("空月份")
            month = normalize_sale_month(value)
            if month == "NaT":
                raise ValueError("无效月份")
            months.append(month)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"第 {row + 2} 行销售月份无效: {value}") from exc
    df[SALE_MONTH_COL] = months
    return df


def _numeric_columns(df: pd.DataFrame, columns: List[str], *, integer=False, allow_blank=False) -> pd.DataFrame:
    for col in columns:
        if col not in df:
            raise ValueError(f"缺少 '{col}' 列")
        values = pd.to_numeric(df[col], errors="coerce")
        invalid = values.isna() & (df[col].notna() if allow_blank else True)
        invalid |= values.notna() & (~np.isfinite(values) | (values < 0))
        if integer:
            invalid |= values.notna() & (values % 1 != 0)
        if invalid.any():
            row = invalid[invalid].index[0]
            rule = "非负整数" if integer else "非负数值"
            raise ValueError(f"第 {row + 2} 行 '{col}' 必须为有限的{rule}")
        df[col] = values
    return df


def _normalize_triangle(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    df = _checked_table(df)
    df = df.rename(columns={c: f"MIS{parse_mis_age(c)}" for c in df.columns if parse_mis_age(c) is not None})
    if df.columns.duplicated().any():
        raise ValueError("存在重复的 MIS 发展期列")
    columns = sorted_mis_columns(df.columns)
    if not columns:
        raise ValueError("未找到任何 MIS<n> 发展期列")
    df = _numeric_columns(df, columns, integer=metric == "台次", allow_blank=True)
    if not df[columns].notna().to_numpy().any():
        raise ValueError("三角形数据全部为空")
    return df.groupby(SALE_MONTH_COL)[columns].sum(min_count=1)


def load_workbook_data(sheets: Dict[str, pd.DataFrame], mappings: List[SheetMapping]) -> WorkbookImport:
    """Validate the whole selection before returning a new import snapshot.

    Validation triangles are explicitly incremental and are kept separate
    from the historical bundles and their inferred evaluation months.
    """
    errors = []
    selected = {}
    seen_sheets = set()
    for mapping in mappings:
        if mapping.purpose == "忽略":
            continue
        label = f"Sheet「{mapping.sheet_name}」"
        if mapping.sheet_name not in sheets:
            errors.append(f"{label}不存在")
        if mapping.sheet_name in seen_sheets:
            errors.append(f"{label}被重复使用")
        seen_sheets.add(mapping.sheet_name)
        if mapping.purpose not in SHEET_PURPOSES:
            errors.append(f"{label}用途无效")
        claim = mapping.purpose in ("历史索赔", "验证数据")
        if claim and (mapping.system not in SYSTEM_LIST or mapping.metric not in CLAIM_METRICS):
            errors.append(f"{label}请选择有效的系统及台次/金额指标")
        key = (mapping.purpose, mapping.system if claim else "", mapping.metric if claim else "")
        if key in selected:
            errors.append(f"{label}与 Sheet「{selected[key].sheet_name}」映射重复")
        selected[key] = mapping
    if ("销量", "", "") not in selected:
        errors.append("必须选择一个销量 Sheet")
    systems = {key[1] for key in selected if key[0] == "历史索赔"}
    if not systems:
        errors.append("至少选择一个系统的历史台次和金额 Sheet")
    for system in sorted(systems):
        for metric in CLAIM_METRICS:
            if ("历史索赔", system, metric) not in selected:
                errors.append(f"[{system}] 缺少历史{metric} Sheet")
    for key, mapping in selected.items():
        if key[0] == "验证数据" and key[1] not in systems:
            errors.append(f"Sheet「{mapping.sheet_name}」对应系统未导入历史索赔")
    if errors:
        raise ValueError("\n".join(errors))

    tables = {}
    for key, mapping in selected.items():
        try:
            raw = sheets[mapping.sheet_name]
            if key[0] in ("历史索赔", "验证数据"):
                tables[key] = _normalize_triangle(raw, key[2])
            else:
                df = _checked_table(raw)
                column = SALES_COL if key[0] == "销量" else MILEAGE_COL
                df = _numeric_columns(df, [column], integer=column == SALES_COL)
                if column == MILEAGE_COL and df[SALE_MONTH_COL].duplicated().any():
                    raise ValueError("同一销售月份存在多条里程记录")
                df = df.groupby(SALE_MONTH_COL, as_index=False)[[column]].sum()
                if column == SALES_COL:
                    if df[column].sum() <= 0:
                        raise ValueError("总销量必须大于 0")
                    df[column] = df[column].astype(int)
                tables[key] = df
        except (ValueError, TypeError, OverflowError) as exc:
            errors.append(f"Sheet「{mapping.sheet_name}」: {exc}")
    if errors:
        raise ValueError("\n".join(errors))

    sales = tables[("销量", "", "")]
    mileage = tables.get(("月均里程", "", ""))
    issues = []
    if mileage is not None:
        missing = sorted(set(sales[SALE_MONTH_COL]) - set(mileage[SALE_MONTH_COL]))
        if missing:
            issues.append(ValidationIssue("warning", f"月均里程缺少销售月份 {missing}，这些批次按 0 公里/月处理"))
    bundles = {}
    for system in SYSTEM_LIST:
        if system not in systems:
            continue
        count_key, amount_key = ("历史索赔", system, "台次"), ("历史索赔", system, "金额")
        try:
            bundles[system] = prepare_triangle_bundle(system, tables[count_key], tables[amount_key], sales)
        except (ValueError, TypeError, OverflowError) as exc:
            errors.append(f"Sheet「{selected[count_key].sheet_name}」/「{selected[amount_key].sheet_name}」: {exc}")
    if errors:
        raise ValueError("\n".join(errors))
    actuals = {}
    for (purpose, system, metric), table in tables.items():
        if purpose == "验证数据":
            actuals.setdefault(system, {})[metric] = table
    return WorkbookImport(sales, bundles, mileage, actuals, issues)


@dataclass
class ValidationIssue:
    """A single human-readable validation warning or error."""

    level: str  # "warning" | "error" | "info"
    message: str


@dataclass
class TriangleBundle:
    """A fully validated, normalized claim triangle for one system.

    Attributes:
        name: Warranty system name.
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
