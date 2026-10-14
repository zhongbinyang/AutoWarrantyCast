"""The application's only upload entry point and its import lifecycle."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from . import data_io
from .utils import SYSTEM_LIST

DEMO_NAME = "保修预测导入示例.xlsx"
DEMO_PATH = Path(__file__).resolve().parent.parent / "sample_data" / DEMO_NAME
NO_MILEAGE = "不提供（纯时间模型）"
FIXED_MILEAGE = "固定值（公里/月）"
WORKBOOK_MILEAGE = "使用工作簿里程"


def clear_import() -> None:
    """Invalidate the complete snapshot, including forecasts and downloads."""
    st.session_state.update(
        workbook_data=None, sales_df=None, bundles={}, import_issues={},
        results={}, imported_signature=None,
    )


@st.cache_data(show_spinner=False, max_entries=4)
def _read_sheets(content: bytes):
    return data_io.read_workbook(content)


def render_workbook_import() -> None:
    st.subheader("1. Excel 统一导入")
    st.caption("一次上传销量、各系统台次与金额；月均里程和实际验证数据可放在同一文件中。")
    if DEMO_PATH.is_file():
        st.download_button(
            "⬇️ 下载 Excel Demo", DEMO_PATH.read_bytes(), file_name=DEMO_NAME,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_import_demo",
        )
    with st.expander("查看 Excel 格式说明"):
        st.markdown(
            "- **销量**：销售月份、销量。\n"
            "- **系统名_台次 / 系统名_金额**：销售月份、MIS0、MIS1……，每个系统需成对提供。\n"
            "- **月均里程**（可选）：销售月份、月均行驶里程（公里/月）。\n"
            "- **系统名_验证台次 / 系统名_验证金额**（可选）：实际增量三角形，仅用于预测验证。\n"
            "- 第一行必须为表头；月份采用 YYYY-MM，金额单位为元，未来未知值留空。\n"
            "- 自动识别后可调整映射；不需要的数据设为“忽略”。"
        )
    uploaded = st.file_uploader(
        "上传完整数据 Excel", type=["xlsx"], key="workbook_upload", on_change=clear_import,
    )
    if uploaded is None:
        clear_import()
        st.session_state["workbook_file_hash"] = None
        return
    content = uploaded.getvalue()
    file_hash = sha256(content).hexdigest()
    if st.session_state.get("workbook_file_hash") != file_hash:
        clear_import()
        st.session_state["workbook_file_hash"] = file_hash
    try:
        sheets = _read_sheets(content)
    except Exception as exc:
        clear_import()
        st.error(f"Excel 读取失败：{exc}")
        return
    suggested = data_io.suggest_sheet_mappings(sheets)
    original = pd.DataFrame([
        {"Sheet": m.sheet_name, "用途": m.purpose, "系统": m.system, "指标": m.metric}
        for m in suggested
    ])
    unknown = [m.sheet_name for m in suggested if m.purpose == "忽略" and "说明" not in m.sheet_name]
    if unknown:
        st.info(f"以下 Sheet 未自动识别，默认忽略；如需导入，请调整用途、系统和指标：{'、'.join(unknown)}")
    st.markdown("**2. 确认 Sheet 映射**")
    edited = st.data_editor(
        original, hide_index=True, use_container_width=True, num_rows="fixed",
        disabled=["Sheet"], key=f"sheet_mapping_{file_hash}", on_change=clear_import,
        column_config={
            "用途": st.column_config.SelectboxColumn(options=data_io.SHEET_PURPOSES, required=True),
            "系统": st.column_config.SelectboxColumn(options=[""] + SYSTEM_LIST),
            "指标": st.column_config.SelectboxColumn(options=[""] + data_io.CLAIM_METRICS),
        },
    )
    mappings = [
        data_io.SheetMapping(row["Sheet"], row["用途"], row["系统"], row["指标"])
        for row in edited.fillna("").to_dict("records")
    ]
    signature = file_hash + json.dumps([asdict(m) for m in mappings], ensure_ascii=False, sort_keys=True)
    if st.session_state.get("imported_signature") != signature:
        clear_import()
    if st.button("导入数据", key="commit_workbook"):
        try:
            with st.spinner("正在校验并导入全部数据…"):
                imported = data_io.load_workbook_data(sheets, mappings)
            clear_import()
            st.session_state.update(
                workbook_data=imported, sales_df=imported.sales, bundles=imported.bundles,
                import_issues={name: bundle.issues for name, bundle in imported.bundles.items()},
                imported_signature=signature,
                mileage_mode=WORKBOOK_MILEAGE if imported.mileage is not None else NO_MILEAGE,
            )
        except Exception as exc:
            clear_import()
            st.error(f"导入失败，请修正后重试：\n\n{exc}")
    imported = st.session_state.get("workbook_data")
    if imported is None:
        st.info("数据尚未导入或已经修改，请确认映射后点击“导入数据”。")
        return
    st.success(f"已导入 {len(imported.sales)} 个销售月份、{len(imported.bundles)} 个系统：{'、'.join(imported.bundles)}")
    for issue in imported.issues:
        st.warning(issue.message)
    with st.expander("销量预览"):
        st.dataframe(imported.sales, use_container_width=True, hide_index=True)


def render_mileage_settings():
    st.sidebar.subheader("每月行驶里程")
    imported = st.session_state.get("workbook_data")
    has_mileage = imported is not None and imported.mileage is not None
    options = [NO_MILEAGE, FIXED_MILEAGE] + ([WORKBOOK_MILEAGE] if has_mileage else [])
    if st.session_state.get("mileage_mode") not in options:
        st.session_state["mileage_mode"] = WORKBOOK_MILEAGE if has_mileage else NO_MILEAGE
    def clear_results():
        st.session_state["results"] = {}
    mode = st.sidebar.radio("里程输入方式", options, key="mileage_mode", on_change=clear_results)
    if mode == FIXED_MILEAGE:
        value = st.sidebar.number_input(
            "月均行驶里程", min_value=100, value=1500, step=100, on_change=clear_results,
        )
        return value, None
    if mode == WORKBOOK_MILEAGE:
        st.sidebar.caption(f"已从工作簿加载 {len(imported.mileage)} 个批次的里程数据")
        return None, imported.mileage
    return None, None
