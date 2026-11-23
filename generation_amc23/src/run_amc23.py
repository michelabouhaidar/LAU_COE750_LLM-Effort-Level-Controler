#!/usr/bin/env python
"""Record Qwen3.5-2B on AMC 2023: generation_math/src/run_math.py, data in generation_amc23/data/.

One recording of all 40 questions (--set amc23); the train / val / test split by question is applied
at export. Defaults that differ from MATH:
  --samples 10   40 questions x 10 = 400 traces (as GSM8K and MATH)
  --b-max 32768  double context: the MATH cuts + 24,576 and 32,768
The server needs LLM_MAX_LEN=33792.

  ~/.venvs/vllm/bin/python generation_amc23/src/run_amc23.py run --set amc23 --configs t0.6_pp1.5 direct --stop-on-loop --concurrency 8
  ~/.venvs/vllm/bin/python generation_amc23/src/run_amc23.py report --set amc23 --configs t0.6_pp1.5 direct

Output: generation_amc23/data/results/amc23/<config>/ traces.jsonl, answers.jsonl, end_answers.jsonl
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths_amc23  # noqa: E402,F401  (points the MATH code at generation_amc23/data)
import run_math  # noqa: E402

run_math.SAMPLES = 10
run_math.B_MAX = 32768
run_math.CUTS = run_math.CUTS + (24576, 32768)

if __name__ == "__main__":
    run_math.main()
