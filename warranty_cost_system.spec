# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, copy_metadata, collect_submodules

block_cipher = None

# Collect Streamlit and dependent libraries data files and metadata
datas = []
datas += collect_data_files('streamlit')
datas += collect_data_files('altair')
datas += copy_metadata('streamlit')
datas += copy_metadata('altair')
datas += copy_metadata('scikit-learn')

# Application source code, entrypoint, and sample data
datas += [
    ('app.py', '.'),
    ('src', 'src'),
    ('sample_data', 'sample_data'),
]

# Recursively collect all submodules to avoid any missing dynamic imports
hiddenimports = []
hiddenimports += collect_submodules('streamlit')
hiddenimports += collect_submodules('src')
hiddenimports += collect_submodules('starlette')
hiddenimports += collect_submodules('uvicorn')
hiddenimports += collect_submodules('altair')
hiddenimports += [
    'pandas',
    'numpy',
    'scipy',
    'scipy.stats',
    'scipy.optimize',
    'scipy.signal',
    'scipy.special',
    'sklearn',
    'sklearn.ensemble',
    'sklearn.metrics',
    'openpyxl',
    'openpyxl.styles',
    'matplotlib',
    'matplotlib.pyplot',
    'seaborn',
]

a = Analysis(
    ['launcher.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch',
        'torchvision',
        'torchaudio',
        'sympy',
        'IPython',
        'notebook',
        'jupyter',
        'pytest',
        'test',
        'tests',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='车辆保修成本精算与预测系统',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
