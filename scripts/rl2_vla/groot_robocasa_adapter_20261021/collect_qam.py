"""Existing collector with only temporary object XML writes isolated."""
import runpy
from pathlib import Path
from readonly_assets import install
install()
runpy.run_path(str(Path(__file__).resolve().parents[3]/'scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py'),run_name='__main__')
