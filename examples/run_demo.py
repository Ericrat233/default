from pathlib import Path
from openpod.demo import run_demo
run_demo(Path(__file__).resolve().parents[1]/'data'/'generated')
