"""Exercise Streamlit import lifecycle with the upload transport mocked."""

from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from test_workbook_import import input_sheets

ROOT = Path(__file__).resolve().parents[1]
HARNESS = """
import streamlit as st
from src.import_ui import render_workbook_import, render_mileage_settings
render_workbook_import()
render_mileage_settings()
st.button('运行预测', disabled=not st.session_state.get('bundles'), key='forecast')
"""


class ImportUITests(unittest.TestCase):
    def test_full_app_import_forecast_validation_downloads(self):
        # Only mock file selection: real demo bytes go through the Excel parser,
        # sheet detection, mapping editor, normalization and prediction pipeline.
        content = (ROOT / "sample_data" / "保修预测导入示例.xlsx").read_bytes()
        with patch("streamlit.file_uploader", return_value=BytesIO(content)):
            at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
            at.button(key="commit_workbook").click().run()
            self.assertEqual(len(at.exception), 0)
            self.assertEqual(len(at.session_state["bundles"]), 2)
            next(n for n in at.number_input if n.label == "模拟次数").set_value(200)
            next(b for b in at.button if "运行预测" in b.label).click().run()
            self.assertEqual(len(at.exception), 0, [e.message for e in at.exception])
            self.assertEqual(len(at.error), 0, [e.value for e in at.error])
            self.assertEqual(len(at.session_state["results"]), 2)
            self.assertEqual(sum("验证：MAE=" in s.value for s in at.success), 4)
            downloads = at.get("download_button")
            self.assertEqual(len(downloads), 4)
            self.assertTrue(any(d.proto.label == "⬇️ 下载 Excel Demo" for d in downloads))
            self.assertTrue(all(result.bundle.evaluation_month == "2024-07" for result in at.session_state["results"].values()))
            # Changing result views should retain committed input and predictions.
            at.radio(key="tri_view_发动机系统").set_value("索赔金额").run()
            self.assertEqual(len(at.exception), 0)
            self.assertEqual(len(at.session_state["results"]), 2)

    def test_workbook_mileage_default_and_user_override(self):
        sheets = input_sheets()
        sheets["月均里程"] = pd.DataFrame({"销售月份": ["2024-01", "2024-02"], "月均行驶里程": [1200, 1500]})
        with patch("streamlit.file_uploader", return_value=BytesIO(b"with mileage")), \
             patch("src.import_ui._read_sheets", return_value=sheets):
            at = AppTest.from_string(HARNESS, default_timeout=20).run()
            at.button(key="commit_workbook").click().run()
            self.assertEqual(len(at.exception), 0)
            self.assertEqual(at.radio(key="mileage_mode").value, "使用工作簿里程")
            at.session_state["results"] = {"sentinel": "old mileage"}
            at.radio(key="mileage_mode").set_value("不提供（纯时间模型）").run()
            self.assertEqual(at.session_state["results"], {})
            self.assertFalse(at.button(key="forecast").disabled)
            at.run()
            self.assertEqual(at.radio(key="mileage_mode").value, "不提供（纯时间模型）")

    def test_empty_app_has_one_uploader_and_no_prediction(self):
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.get("file_uploader")), 1)
        self.assertTrue(next(b for b in at.button if "运行预测" in b.label).disabled)

    def test_commit_rerun_mapping_file_failure_and_removal(self):
        content = BytesIO(b"test workbook transport")
        with patch("streamlit.file_uploader", return_value=content) as upload, \
             patch("src.import_ui._read_sheets", return_value=input_sheets()) as read:
            at = AppTest.from_string(HARNESS, default_timeout=20).run()
            self.assertEqual(len(at.exception), 0)
            self.assertTrue(at.button(key="forecast").disabled)
            at.button(key="commit_workbook").click().run()
            self.assertEqual(len(at.exception), 0)
            self.assertFalse(at.button(key="forecast").disabled)
            at.session_state["results"] = {"sentinel": "unchanged"}
            at.run()
            self.assertEqual(at.session_state["results"], {"sentinel": "unchanged"})

            # AppTest does not support editing a data_editor yet; mock that
            # widget's returned value while keeping validation and state real.
            def edit_mapping(frame, **kwargs):
                edited = frame.copy()
                edited.loc[1, "用途"] = "忽略"
                return edited
            with patch("streamlit.data_editor", side_effect=edit_mapping):
                at.run()
                self.assertEqual(len(at.exception), 0)
                self.assertTrue(at.button(key="forecast").disabled)
                self.assertEqual(at.session_state["results"], {})
                at.button(key="commit_workbook").click().run()
                self.assertTrue(at.button(key="forecast").disabled)
                self.assertTrue(any("缺少历史台次" in e.value for e in at.error))

            upload.return_value = BytesIO(b"replacement workbook")
            at.run()
            at.button(key="commit_workbook").click().run()
            self.assertFalse(at.button(key="forecast").disabled)
            at.session_state["results"] = {"sentinel": "old"}
            read.side_effect = ValueError("corrupt workbook")
            upload.return_value = BytesIO(b"broken")
            at.run()
            self.assertTrue(at.button(key="forecast").disabled)
            self.assertEqual(at.session_state["results"], {})

            upload.return_value = None
            at.run()
            self.assertIsNone(at.session_state["sales_df"])
            self.assertTrue(at.button(key="forecast").disabled)


if __name__ == "__main__":
    unittest.main()
