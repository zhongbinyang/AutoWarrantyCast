"""Streamlit application: vehicle warranty cost actuarial and forecasting system.

Run with ``streamlit run app.py`` (see start.bat for a one-click launcher).
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

matplotlib.rcParams["axes.unicode_minus"] = False
for _font in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"):
    matplotlib.rcParams["font.sans-serif"] = [_font] + matplotlib.rcParams.get("font.sans-serif", [])

from src import claims_model, cost_model, data_io, excel_export, metrics, monte_carlo, weibull_model
from src.pipeline import SystemResult, run_system_pipeline
from src.utils import SALE_MONTH_COL, SALES_COL, SYSTEM_LIST, mis_col

st.set_page_config(page_title="AutoWarrantyCast - 车辆保修成本精算与预测系统", layout="wide")


# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "sales_df": None,
        "bundles": {},  # name -> TriangleBundle
        "results": {},  # name -> SystemResult
        "import_issues": {},  # name -> list[ValidationIssue]
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


# --------------------------------------------------------------------------
# Small rendering helpers
# --------------------------------------------------------------------------
def style_forecast(df: pd.DataFrame, mask: pd.DataFrame):
    """Return a pandas Styler highlighting forecast cells in light yellow."""

    def highlight(_frame: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for col in df.columns:
            if col in mask.columns:
                col_mask = mask[col].reindex(df.index).fillna(False)
                styles[col] = np.where(col_mask, "background-color: #FFF2CC", "")
        return styles

    return df.style.apply(highlight, axis=None).format(precision=1, na_rep="")


def build_mileage_dict(
    sales_df: pd.DataFrame,
    fixed_mileage: Optional[float],
    mileage_upload_df: Optional[pd.DataFrame],
) -> Optional[Dict[str, float]]:
    """Resolve the effective monthly-mileage-by-batch dict, or None for a pure time model."""
    if mileage_upload_df is not None:
        mapping = dict(zip(mileage_upload_df[SALE_MONTH_COL], mileage_upload_df["月均行驶里程"]))
        return {sm: float(mapping.get(sm, 0.0) or 0.0) for sm in sales_df[SALE_MONTH_COL]}
    if fixed_mileage is not None and fixed_mileage > 0:
        return {sm: float(fixed_mileage) for sm in sales_df[SALE_MONTH_COL]}
    return None


def parse_breakpoints(text: str) -> list:
    text = text.strip()
    if not text:
        return []
    try:
        values = sorted({int(v.strip()) for v in text.split(",") if v.strip()})
        return values
    except ValueError:
        st.sidebar.error("分段点格式错误，应为逗号分隔的整数，如 12,60")
        return []


# --------------------------------------------------------------------------
# Sidebar: global settings
# --------------------------------------------------------------------------
st.sidebar.header("⚙️ 全局设置")

st.sidebar.subheader("保修期设定")
warranty_months = st.sidebar.number_input("保修期月数", min_value=1, max_value=120, value=36, step=1)
use_mileage_limit = st.sidebar.checkbox("设置保修期里程限制", value=False)
mileage_limit_km = None
if use_mileage_limit:
    mileage_limit_km = st.sidebar.number_input(
        "保修期里程限制（公里）", min_value=1000, value=100000, step=1000
    )

st.sidebar.subheader("每月行驶里程")
mileage_mode = st.sidebar.radio(
    "里程输入方式", ["不提供（纯时间模型）", "固定值（公里/月）", "按销售批次上传"]
)
fixed_mileage = None
mileage_upload_df = None
if mileage_mode == "固定值（公里/月）":
    fixed_mileage = st.sidebar.number_input("月均行驶里程", min_value=100, value=1500, step=100)
elif mileage_mode == "按销售批次上传":
    mileage_file = st.sidebar.file_uploader("上传月均行驶里程 CSV", type=["csv"], key="mileage_upload")
    if mileage_file is not None:
        try:
            mileage_upload_df = data_io.load_mileage(mileage_file)
            st.sidebar.success(f"已加载 {len(mileage_upload_df)} 个批次的里程数据")
        except Exception as exc:  # noqa: BLE001 - surface parse errors to the user
            st.sidebar.error(f"里程文件解析失败: {exc}")

st.sidebar.subheader("小样本 / 冷启动优化设置")
enable_few_shot = st.sidebar.checkbox(
    "启用小样本精度优化",
    value=True,
    help="针对新车上市初期、数据稀疏或零星偶发索赔系统，引入行业物理先验惩罚与经验贝叶斯收缩，防止参数过拟合与因子爆炸",
)
reg_weight = 1.0
if enable_few_shot:
    prior_strength = st.sidebar.select_slider(
        "先验约束强度",
        options=["温和 (0.5x)", "标准 (1.0x)", "较强 (2.0x)"],
        value="标准 (1.0x)",
        help="样本少时约束参数靠近行业基准，样本充裕后自动衰减归零",
    )
    reg_weight = {"温和 (0.5x)": 0.5, "标准 (1.0x)": 1.0, "较强 (2.0x)": 2.0}.get(prior_strength, 1.0)

st.sidebar.subheader("台次预测方法")
weibull_method = st.sidebar.radio("威布尔拟合方法", ["极大似然估计 (MLE)", "贝叶斯威布尔 (PyMC)"])
use_bayesian = weibull_method.startswith("贝叶斯")

st.sidebar.subheader("威布尔分段设置")
segment_mode = st.sidebar.radio("分段方式", ["不分段（单段）", "手动分段", "自适应智能分段"])
manual_breakpoints_text = ""
if segment_mode == "手动分段":
    manual_breakpoints_text = st.sidebar.text_input("分段点（逗号分隔，如 12,60）", value="12,60")

st.sidebar.subheader("单次维修成本预测方法")
cost_method = st.sidebar.selectbox(
    "方法选择",
    cost_model.ALL_METHODS,
    index=0,
    help="默认推荐：历史均值法或经验贝叶斯收缩法（小样本防过拟合最佳）；数据充足时可选 XGBoost",
)

st.sidebar.subheader("索赔金额预测方法")
amount_method = st.sidebar.selectbox(
    "方法选择",
    [
        claims_model.AMOUNT_METHOD_COUNT_X_UNITCOST,
        claims_model.AMOUNT_METHOD_BORNHUETTER_FERGUSON,
        claims_model.AMOUNT_METHOD_CHAIN_LADDER,
        claims_model.AMOUNT_METHOD_LIGHTGBM,
    ],
    index=0,
    help="推荐：台次×单次成本法或Bornhuetter-Ferguson(BF)法（精算行业小样本防因子爆炸最佳）",
)

st.sidebar.subheader("蒙特卡洛模拟")
n_simulations = st.sidebar.number_input("模拟次数", min_value=200, max_value=50000, value=5000, step=100)



# --------------------------------------------------------------------------
# Main layout
# --------------------------------------------------------------------------
st.title("🚗 车辆保修成本精算与预测系统")
st.caption("基于双变量（时间+里程）竞争风险威布尔模型的保修索赔台次/金额预测与可信度评估")

tab_import, tab_results, tab_download = st.tabs(["📥 数据导入区", "📊 预测结果展示区", "⬇️ 下载区"])


# --------------------------------------------------------------------------
# Tab 1: data import
# --------------------------------------------------------------------------
with tab_import:
    st.subheader("1. 月度销量导入")
    sales_file = st.file_uploader("上传销量文件（CSV/Excel，列：销售月份、销量）", type=["csv", "xlsx", "xls"])
    if sales_file is not None:
        try:
            st.session_state["sales_df"] = data_io.load_sales(sales_file)
            st.success(f"成功加载 {len(st.session_state['sales_df'])} 个销售月份")
        except Exception as exc:  # noqa: BLE001
            st.error(f"销量文件解析失败: {exc}")

    if st.session_state["sales_df"] is not None:
        st.dataframe(st.session_state["sales_df"], use_container_width=True, height=200)

    st.divider()
    st.subheader("2. 索赔数据导入")
    import_mode = st.radio("导入模式", ["单系统导入", "多系统导入"], horizontal=True)

    new_bundles: Dict[str, "data_io.TriangleBundle"] = {}
    new_issues = {}

    def _try_build_bundle(system_name: str, count_file, amount_file, sales_df) -> None:
        try:
            count_df = data_io.load_triangle_sheet(count_file)
            amount_df = data_io.load_triangle_sheet(amount_file)
            bundle = data_io.prepare_triangle_bundle(system_name, count_df, amount_df, sales_df)
            new_bundles[system_name] = bundle
            new_issues[system_name] = bundle.issues
        except Exception as exc:  # noqa: BLE001
            st.error(f"[{system_name}] 三角形数据处理失败: {exc}")

    sales_df_current = st.session_state["sales_df"]

    if import_mode == "单系统导入":
        col1, col2 = st.columns(2)
        with col1:
            count_file = st.file_uploader("上传索赔台次（Excel）", type=["xlsx", "xls"], key="single_count")
        with col2:
            amount_file = st.file_uploader("上传索赔金额（Excel）", type=["xlsx", "xls"], key="single_amount")
        if count_file is not None and amount_file is not None and sales_df_current is not None:
            _try_build_bundle("单系统", count_file, amount_file, sales_df_current)
        elif (count_file is not None or amount_file is not None) and sales_df_current is None:
            st.warning("请先上传销量文件")
    else:
        if sales_df_current is None:
            st.warning("请先上传销量文件，再导入各系统三角形数据")
        for system in SYSTEM_LIST:
            with st.expander(f"📁 {system}"):
                c1, c2 = st.columns(2)
                with c1:
                    cf = st.file_uploader(f"上传台次 - {system}", type=["xlsx", "xls"], key=f"count_{system}")
                with c2:
                    af = st.file_uploader(f"上传金额 - {system}", type=["xlsx", "xls"], key=f"amount_{system}")
                if cf is not None and af is not None and sales_df_current is not None:
                    _try_build_bundle(system, cf, af, sales_df_current)
                    st.success(f"{system} 数据已导入")
                    with st.container():
                        preview_cols = [mis_col(a) for a in new_bundles[system].mis_ages[:8]]
                        st.caption("原始三角形预览（增量口径，前 8 期）")
                        st.dataframe(new_bundles[system].count[preview_cols].head(10), use_container_width=True)

    if new_bundles:
        st.session_state["bundles"] = new_bundles
        st.session_state["import_issues"] = new_issues

    if st.session_state["bundles"]:
        st.divider()
        st.subheader("3. 数据校验结果")
        for name, issues in st.session_state["import_issues"].items():
            for issue in issues:
                if issue.level == "warning":
                    st.warning(issue.message)
                elif issue.level == "error":
                    st.error(issue.message)
                else:
                    st.info(issue.message)

        # Few-shot data diagnostic hint
        for name, b in st.session_state["bundles"].items():
            max_obs_age = max(b.mis_ages) if b.mis_ages else 0
            n_obs_cells = int(b.observed_mask.to_numpy().sum())
            if max_obs_age < 12 or n_obs_cells < 15:
                st.info(
                    f"💡 [{name}] 检测到历史观测期较短（最大车龄 MIS{max_obs_age}，有效观测点 {n_obs_cells} 个）。"
                    "已激活小样本行业先验正则化与经验贝叶斯收缩防护，保障外推平滑稳定。"
                )

        st.divider()
        st.subheader("4. 三角形热力图预览")
        heat_system = st.selectbox("选择系统查看热力图", list(st.session_state["bundles"].keys()))
        bundle = st.session_state["bundles"][heat_system]
        mis_cols_preview = [mis_col(a) for a in bundle.mis_ages]
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        sns.heatmap(bundle.count[mis_cols_preview].astype(float), ax=axes[0], cmap="YlOrRd", cbar=True)
        axes[0].set_title(f"{heat_system} - 索赔台次（增量）")
        sns.heatmap(bundle.amount[mis_cols_preview].astype(float), ax=axes[1], cmap="YlGnBu", cbar=True)
        axes[1].set_title(f"{heat_system} - 索赔金额（增量）")
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    run_clicked = st.button(
        "🚀 运行预测模型", type="primary", disabled=not st.session_state["bundles"] or sales_df_current is None
    )

    if run_clicked:
        sales_df_current = st.session_state["sales_df"]
        mileage_by_month = build_mileage_dict(sales_df_current, fixed_mileage, mileage_upload_df)
        results: Dict[str, SystemResult] = {}
        progress = st.progress(0.0, text="正在拟合模型...")
        names = list(st.session_state["bundles"].keys())
        for i, name in enumerate(names):
            bundle = st.session_state["bundles"][name]
            try:
                if segment_mode == "自适应智能分段":
                    points = weibull_model.collect_observed_points(bundle, sales_df_current, mileage_by_month)
                    breakpoints = weibull_model.auto_select_breakpoints(points, max_breakpoints=2)
                elif segment_mode == "手动分段":
                    breakpoints = parse_breakpoints(manual_breakpoints_text)
                else:
                    breakpoints = []

                result = run_system_pipeline(
                    name=name,
                    bundle=bundle,
                    sales_df=sales_df_current,
                    mileage_by_month=mileage_by_month,
                    warranty_months=int(warranty_months),
                    mileage_limit_km=mileage_limit_km,
                    breakpoints=breakpoints,
                    use_bayesian=use_bayesian,
                    cost_method=cost_method,
                    amount_method=amount_method,
                    n_simulations=int(n_simulations),
                    enable_few_shot=enable_few_shot,
                    regularization_weight=reg_weight,
                )
                results[name] = result
            except Exception as exc:  # noqa: BLE001
                st.error(f"[{name}] 模型拟合失败: {exc}")
            progress.progress((i + 1) / len(names), text=f"已完成 {name}")
        progress.empty()
        st.session_state["results"] = results
        if results:
            st.success(f"预测完成，共 {len(results)} 个系统")



# --------------------------------------------------------------------------
# Tab 2: results
# --------------------------------------------------------------------------
with tab_results:
    results: Dict[str, SystemResult] = st.session_state["results"]
    if not results:
        st.info("请先在「数据导入区」上传数据并点击「运行预测模型」")
    else:
        system_tabs = st.tabs(list(results.keys()))
        for tab, name in zip(system_tabs, results.keys()):
            result = results[name]
            with tab:
                sub_tabs = st.tabs(
                    ["三角形", "故障率曲线", "单次成本趋势", "批次汇总", "跨批次月度汇总", "预测可信度报告"]
                )

                # --- 三角形 ---
                with sub_tabs[0]:
                    st.caption("浅黄色底色为预测值")
                    view = st.radio("查看", ["索赔台次", "索赔金额"], horizontal=True, key=f"tri_view_{name}")
                    if view == "索赔台次":
                        st.dataframe(
                            style_forecast(result.count_filled, result.count_forecast_mask),
                            use_container_width=True,
                        )
                    else:
                        st.dataframe(
                            style_forecast(result.amount_filled, result.amount_forecast_mask),
                            use_container_width=True,
                        )

                # --- 故障率曲线 ---
                with sub_tabs[1]:
                    fit = result.fit
                    max_age = max(result.ages)
                    ages_dense = np.linspace(0, max_age, 200)

                    fig, ax = plt.subplots(figsize=(9, 5))
                    sales_map = dict(zip(result.batch_summary[SALE_MONTH_COL], result.batch_summary[SALES_COL]))
                    for sm in result.bundle.count.index:
                        sales = sales_map.get(sm, 0)
                        if sales <= 0:
                            continue
                        cum = result.bundle.count.loc[sm, [mis_col(a) for a in result.bundle.mis_ages]].fillna(0).cumsum()
                        obs_ages = [
                            a for a in result.bundle.mis_ages if bool(result.bundle.observed_mask.loc[sm, mis_col(a)])
                        ]
                        if not obs_ages:
                            continue
                        rates = [cum[mis_col(a)] / sales for a in obs_ages]
                        ax.scatter(obs_ages, rates, s=10, alpha=0.3, color="steelblue")

                    curve = fit.cum_rate(ages_dense, mileage_per_month=0.0)
                    ax.plot(ages_dense, curve, color="darkorange", linewidth=2, label="威布尔拟合曲线")
                    ax.fill_between(
                        result.mc_result.band_ages,
                        result.mc_result.band_lower,
                        result.mc_result.band_upper,
                        color="darkorange",
                        alpha=0.2,
                        label="90% 置信带",
                    )
                    ax.set_xlabel("车龄（月）")
                    ax.set_ylabel("累积故障率")
                    ax.set_title(f"{name} - 故障率曲线（实际 vs 预测）")
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)
                    st.caption(f"拟合优度: R² = {fit.r2:.3f}，MAPE = {fit.mape:.2%}，分段点: {fit.breakpoints or '无（单段）'}")

                # --- 单次成本趋势 ---
                with sub_tabs[2]:
                    hist = result.hist_unit_cost
                    ages_dense = np.array(result.ages, dtype=float)
                    point_pred = result.unit_cost_fit.predict(ages_dense)
                    rng = np.random.default_rng(0)
                    sample_paths = cost_model.sample_unit_cost_paths(result.unit_cost_fit, ages_dense, 300, rng)
                    lower = np.percentile(sample_paths, 5, axis=0)
                    upper = np.percentile(sample_paths, 95, axis=0)

                    fig, ax = plt.subplots(figsize=(9, 5))
                    ax.scatter(hist["age"], hist["unit_cost"], s=20, color="seagreen", label="历史单次成本")
                    ax.plot(ages_dense, point_pred, color="firebrick", linewidth=2, label=f"预测（{result.unit_cost_fit.method}）")
                    ax.fill_between(ages_dense, lower, upper, color="firebrick", alpha=0.15, label="90% 置信区间")
                    ax.set_xlabel("车龄（月）")
                    ax.set_ylabel("单次维修成本")
                    ax.set_title(f"{name} - 单次维修成本趋势")
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)
                    st.caption(f"拟合优度: R² = {result.unit_cost_fit.r2:.3f}，MAE = {result.unit_cost_fit.mae:.2f}")

                # --- 批次汇总 ---
                with sub_tabs[3]:
                    st.dataframe(result.batch_summary, use_container_width=True)
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    axes[0].plot(result.batch_summary[SALE_MONTH_COL], result.batch_summary["累积故障率"], marker="o")
                    axes[0].set_title("累积故障率趋势")
                    axes[0].tick_params(axis="x", rotation=60)
                    axes[1].plot(result.batch_summary[SALE_MONTH_COL], result.batch_summary["单台费用"], marker="o", color="darkorange")
                    axes[1].set_title("单台费用趋势")
                    axes[1].tick_params(axis="x", rotation=60)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                # --- 跨批次月度汇总 ---
                with sub_tabs[4]:
                    st.dataframe(result.monthly_summary, use_container_width=True)
                    fig, ax = plt.subplots(figsize=(9, 4))
                    ax.plot(result.monthly_summary["车龄_月"], result.monthly_summary["月度平均故障率"], label="月度平均故障率")
                    ax2 = ax.twinx()
                    ax2.plot(
                        result.monthly_summary["车龄_月"],
                        result.monthly_summary["月度单次维修成本"],
                        color="firebrick",
                        label="月度单次维修成本",
                    )
                    ax.set_xlabel("车龄（月）")
                    ax.set_ylabel("月度平均故障率")
                    ax2.set_ylabel("月度单次维修成本")
                    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
                    st.pyplot(fig)
                    plt.close(fig)

                # --- 预测可信度报告 ---
                with sub_tabs[5]:
                    mc = result.mc_result
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**未来总台次分布**")
                        st.metric("期望值", f"{mc.total_count.mean:,.0f}")
                        st.write(
                            f"标准差: {mc.total_count.std:,.0f} | CV: {mc.total_count.cv:.2%} | "
                            f"VaR90: {mc.total_count.var90:,.0f} | VaR95: {mc.total_count.var95:,.0f}"
                        )
                        fig, ax = plt.subplots(figsize=(6, 3.5))
                        ax.hist(mc.total_count.samples, bins=40, color="steelblue", alpha=0.8)
                        ax.axvline(mc.total_count.var90, color="orange", linestyle="--", label="VaR90")
                        ax.axvline(mc.total_count.var95, color="red", linestyle="--", label="VaR95")
                        ax.legend()
                        ax.set_title("未来总台次蒙特卡洛分布")
                        st.pyplot(fig)
                        plt.close(fig)
                    with c2:
                        st.markdown("**未来总金额分布**")
                        st.metric("期望值", f"{mc.total_amount.mean:,.0f}")
                        st.write(
                            f"标准差: {mc.total_amount.std:,.0f} | CV: {mc.total_amount.cv:.2%} | "
                            f"VaR90: {mc.total_amount.var90:,.0f} | VaR95: {mc.total_amount.var95:,.0f}"
                        )
                        fig, ax = plt.subplots(figsize=(6, 3.5))
                        ax.hist(mc.total_amount.samples, bins=40, color="seagreen", alpha=0.8)
                        ax.axvline(mc.total_amount.var90, color="orange", linestyle="--", label="VaR90")
                        ax.axvline(mc.total_amount.var95, color="red", linestyle="--", label="VaR95")
                        ax.legend()
                        ax.set_title("未来总金额蒙特卡洛分布")
                        st.pyplot(fig)
                        plt.close(fig)

                    st.divider()
                    st.markdown("**验证数据（可选）**")
                    with st.expander("上传真实未来数据以计算 MAE / MAPE"):
                        vc1, vc2 = st.columns(2)
                        with vc1:
                            val_count_file = st.file_uploader(
                                "上传实际索赔台次（Excel）", type=["xlsx", "xls"], key=f"val_count_{name}"
                            )
                        with vc2:
                            val_amount_file = st.file_uploader(
                                "上传实际索赔金额（Excel）", type=["xlsx", "xls"], key=f"val_amount_{name}"
                            )
                        if val_count_file is not None:
                            try:
                                actual_count = data_io.load_triangle_sheet(val_count_file)
                                m = metrics.compute_validation_metrics(
                                    result.count_filled, result.count_forecast_mask, actual_count
                                )
                                if m:
                                    st.success(f"台次验证：MAE={m['MAE']:.2f}, MAPE={m['MAPE']:.2%}（{m['n_cells']} 个单元格）")
                                else:
                                    st.warning("未找到可比较的预测/实际数据重叠单元格")
                            except Exception as exc:  # noqa: BLE001
                                st.error(f"验证数据解析失败: {exc}")
                        if val_amount_file is not None:
                            try:
                                actual_amount = data_io.load_triangle_sheet(val_amount_file)
                                m = metrics.compute_validation_metrics(
                                    result.amount_filled, result.amount_forecast_mask, actual_amount
                                )
                                if m:
                                    st.success(f"金额验证：MAE={m['MAE']:.2f}, MAPE={m['MAPE']:.2%}（{m['n_cells']} 个单元格）")
                                else:
                                    st.warning("未找到可比较的预测/实际数据重叠单元格")
                            except Exception as exc:  # noqa: BLE001
                                st.error(f"验证数据解析失败: {exc}")
                        if val_count_file is None and val_amount_file is None:
                            st.info("无验证数据")


# --------------------------------------------------------------------------
# Tab 3: downloads
# --------------------------------------------------------------------------
with tab_download:
    results: Dict[str, SystemResult] = st.session_state["results"]
    if not results:
        st.info("请先运行预测模型")
    else:
        st.subheader("按系统下载填充三角形（Excel，四个 Sheet）")
        for name, result in results.items():
            fail_rate, unit_cost_pv = excel_export.build_derived_triangles(
                result.count_filled, result.amount_filled, st.session_state["sales_df"], result.ages
            )
            sheets = {
                "台次": result.count_filled,
                "金额": result.amount_filled,
                "单台费用": unit_cost_pv,
                "故障率": fail_rate,
            }
            masks = {
                "台次": result.count_forecast_mask,
                "金额": result.amount_forecast_mask,
                "单台费用": result.amount_forecast_mask,
                "故障率": result.count_forecast_mask,
            }
            excel_bytes = excel_export.export_triangles_workbook(sheets, masks)
            st.download_button(
                f"⬇️ 下载「{name}」三角形 Excel",
                data=excel_bytes,
                file_name=f"{name}_预测三角形.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_xlsx_{name}",
            )

        st.divider()
        st.subheader("汇总报告 CSV（含批次累积故障率与单台费用）")
        combined = []
        for name, result in results.items():
            df = result.batch_summary.copy()
            df.insert(0, "系统", name)
            combined.append(df)
        combined_df = pd.concat(combined, ignore_index=True)
        st.dataframe(combined_df, use_container_width=True)
        csv_bytes = excel_export.export_summary_csv(combined_df)
        st.download_button(
            "⬇️ 下载汇总报告 CSV", data=csv_bytes, file_name="批次汇总报告.csv", mime="text/csv"
        )
