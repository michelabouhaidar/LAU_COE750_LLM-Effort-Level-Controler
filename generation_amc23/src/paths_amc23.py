"""Paths of generation_amc23, and the hook into the MATH code it reuses.

AMC 2023 is recorded exactly like MATH (same prompt, cuts, loop rule, Math-Verify grader, export
columns), so the scripts here run generation_math/src/*.py unchanged, with its data folder pointed at
generation_amc23/data/. Import this module before any generation_math module.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                 # .../generation_amc23
DATA = ROOT / "data"
MATH_SRC = ROOT.parent / "generation_math" / "src"
sys.path.insert(0, str(MATH_SRC))

import grade_math  # noqa: E402  (also puts generation_gsm8k/src on the path)

grade_math.DATA = DATA          # every generation_math module imported after this reads and writes here
