"""Standalone executable launcher for the warranty cost system.

Acts as the entrypoint for PyInstaller. Handles sys._MEIPASS path resolution,
automatic free port allocation (preventing port 8501 conflicts), automatic browser
launch, and first-run extraction of sample_data for end-user convenience.
"""

from __future__ import annotations

import os
import shutil
import socket
import sys
import time
import webbrowser

if False:  # pragma: no cover
    # Static import hints for PyInstaller AST analysis
    import matplotlib.pyplot
    import numpy
    import openpyxl
    import pandas
    import scipy
    import scipy.optimize
    import scipy.signal
    import scipy.special
    import scipy.stats
    import seaborn
    import sklearn
    import sklearn.ensemble
    import sklearn.metrics
    import streamlit.runtime.scriptrunner.magic
    import streamlit.runtime.scriptrunner.magic_funcs
    import streamlit.runtime.scriptrunner.script_runner
    import streamlit.runtime.scriptrunner_utils.script_run_context
    import src.claims_model
    import src.cost_model
    import src.data_io
    import src.excel_export
    import src.exposure
    import src.metrics
    import src.monte_carlo
    import src.pipeline
    import src.utils
    import src.weibull_model


def find_free_port(start_port: int = 8501, max_tries: int = 50) -> int:
    """Find the first available TCP port on localhost starting from start_port."""
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return port
    return start_port


def ensure_sample_data(exe_dir: str, internal_dir: str) -> None:
    """On first run, extract sample_data into the folder beside the executable."""
    target_dir = os.path.join(exe_dir, "sample_data")
    source_dir = os.path.join(internal_dir, "sample_data")
    if not os.path.exists(target_dir) and os.path.exists(source_dir):
        try:
            shutil.copytree(source_dir, target_dir)
            print(f"[提示] 首次启动：已在程序同级目录提取示例数据 -> {target_dir}")
        except Exception as exc:
            print(f"[警告] 提取示例数据失败（不影响主程序运行）: {exc}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if getattr(sys, "frozen", False):
        base_dir = sys._MEIPASS
        exe_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        exe_dir = base_dir

    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    ensure_sample_data(exe_dir, base_dir)

    import streamlit.web.bootstrap
    from streamlit import config as _config

    port = find_free_port(8501)
    app_path = os.path.join(base_dir, "app.py")

    flag_options = {
        "server.port": port,
        "server.address": "localhost",
        "server.headless": False,
        "global.developmentMode": False,
        "browser.gatherUsageStats": False,
        "browser.serverAddress": "localhost",
    }
    for k, v in flag_options.items():
        _config.set_option(k, v)

    url = f"http://localhost:{port}/"
    print("=" * 65)
    print(" [系统] AutoWarrantyCast 车辆保修成本精算与预测系统 正在启动...")
    print(f" 本地服务访问地址: {url}")
    print(" 系统将自动打开默认浏览器；使用期间请勿关闭本控制台窗口。")
    print(" 如需退出系统，请在此窗口中按 Ctrl + C 或直接关闭窗口。")
    print("=" * 65)

    # Launch Streamlit application loop
    streamlit.web.bootstrap.run(app_path, False, [], flag_options)


if __name__ == "__main__":
    main()
