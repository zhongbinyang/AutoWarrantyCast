"""Generate realistic synthetic sample data for the warranty cost system.

Produces, under ``sample_data/``:

* ``sales.csv`` -- monthly new-vehicle sales for 30 sale-month batches.
* ``engine_claim_count.xlsx`` / ``engine_claim_amount.xlsx`` -- 发动机系统 triangle.
* ``chassis_claim_count.xlsx`` / ``chassis_claim_amount.xlsx`` -- 底盘系统 triangle.
* ``monthly_mileage.csv`` -- optional per-batch monthly mileage.

Each system uses a distinct true Weibull hazard so the two triangles look
meaningfully different, with Poisson claim-count noise and lognormal cost
noise layered on top. Cells beyond each batch's observed development window
(governed by a single fixed "data evaluation month") are left blank, exactly
matching the ragged-triangle shape of real warranty data.

Run with: ``python scripts/generate_sample_data.py``
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")

N_BATCHES = 30
START_MONTH = "2022-01"
EVAL_MONTH = "2024-07"  # data snapshot date

SYSTEMS = {
    "engine": {
        "sheet_label": "发动机系统",
        "beta_t": 2.3,
        "eta_t": 55.0,
        "base_unit_cost": 3400.0,
        "cost_growth": 0.01,
        # Engine wear is genuinely mileage-driven in this sample, so the
        # bivariate time+mileage model has real signal to recover.
        "beta_m": 1.8,
        "eta_m": 45000.0,
    },
    "chassis": {
        "sheet_label": "底盘系统",
        "beta_t": 1.6,
        "eta_t": 58.0,
        "base_unit_cost": 1800.0,
        "cost_growth": 0.006,
        # Chassis failures are treated as calendar-time driven only.
        "beta_m": None,
        "eta_m": None,
    },
}

MAX_AGE = 36


def month_range(start: str, n: int):
    p = pd.Period(start, freq="M")
    return [str(p + i) for i in range(n)]


def build_sales() -> pd.DataFrame:
    months = month_range(START_MONTH, N_BATCHES)
    base = 4000
    trend = np.linspace(0, 2500, N_BATCHES)
    seasonal = 800 * np.sin(np.linspace(0, 6 * np.pi, N_BATCHES))
    noise = RNG.normal(0, 250, N_BATCHES)
    sales = np.clip(base + trend + seasonal + noise, 500, None).round().astype(int)
    return pd.DataFrame({"销售月份": months, "销量": sales})


def observed_age_limit(sale_month: str) -> int:
    age = (pd.Period(EVAL_MONTH, freq="M") - pd.Period(sale_month, freq="M")).n
    return int(np.clip(age, 0, MAX_AGE))


def build_system_triangles(sales_df: pd.DataFrame, params: dict, mileage_df: pd.DataFrame = None):
    beta_t = params["beta_t"]
    eta_t = params["eta_t"]
    base_cost = params["base_unit_cost"]
    cost_growth = params["cost_growth"]
    beta_m = params.get("beta_m")
    eta_m = params.get("eta_m")

    mileage_map = {}
    if mileage_df is not None:
        mileage_map = dict(zip(mileage_df["销售月份"], mileage_df["月均行驶里程"]))

    ages = list(range(0, MAX_AGE + 1))
    count_rows = []
    amount_rows = []

    for _, row in sales_df.iterrows():
        sm = row["销售月份"]
        sales = int(row["销量"])
        max_obs_age = observed_age_limit(sm)
        batch_factor = float(np.exp(RNG.normal(0, 0.08)))
        monthly_mileage = float(mileage_map.get(sm, 0.0))

        cum_prev = 0.0
        count_row = {"销售月份": sm}
        amount_row = {"销售月份": sm}
        for age in ages:
            col = f"MIS{age}"
            if age > max_obs_age:
                count_row[col] = np.nan
                amount_row[col] = np.nan
                continue
            hazard = (age / eta_t) ** beta_t
            if beta_m is not None and eta_m is not None and monthly_mileage > 0:
                cumulative_mileage = age * monthly_mileage
                hazard += (cumulative_mileage / eta_m) ** beta_m
            true_cum_rate = 1.0 - np.exp(-hazard)
            true_cum_rate *= batch_factor
            true_cum_rate = min(true_cum_rate, 0.98)
            expected_cum_count = true_cum_rate * sales
            expected_inc = max(expected_cum_count - cum_prev, 0.0)
            inc_count = RNG.poisson(lam=expected_inc) if expected_inc > 0 else 0
            cum_prev += inc_count

            unit_cost = base_cost * (1 + cost_growth * age) * float(np.exp(RNG.normal(0, 0.07)))
            inc_amount = inc_count * unit_cost * float(np.exp(RNG.normal(0, 0.05)))

            count_row[col] = int(inc_count)
            amount_row[col] = round(float(inc_amount), 2)

        count_rows.append(count_row)
        amount_rows.append(amount_row)

    return pd.DataFrame(count_rows), pd.DataFrame(amount_rows)


def build_mileage(sales_df: pd.DataFrame) -> pd.DataFrame:
    n = len(sales_df)
    base = 1500
    drift = np.linspace(-150, 150, n)
    noise = RNG.normal(0, 120, n)
    mileage = np.clip(base + drift + noise, 800, None).round().astype(int)
    return pd.DataFrame({"销售月份": sales_df["销售月份"], "月均行驶里程": mileage})


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    sales_df = build_sales()
    sales_df.to_csv(os.path.join(OUT_DIR, "sales.csv"), index=False, encoding="utf-8-sig")

    mileage_df = build_mileage(sales_df)
    mileage_df.to_csv(os.path.join(OUT_DIR, "monthly_mileage.csv"), index=False, encoding="utf-8-sig")

    for key, params in SYSTEMS.items():
        count_df, amount_df = build_system_triangles(sales_df, params, mileage_df)
        count_df.to_excel(os.path.join(OUT_DIR, f"{key}_claim_count.xlsx"), index=False)
        amount_df.to_excel(os.path.join(OUT_DIR, f"{key}_claim_amount.xlsx"), index=False)
        print(f"生成 {params['sheet_label']} 三角形: {count_df.shape}")

    print(f"样例数据已生成到: {OUT_DIR}")


if __name__ == "__main__":
    main()
