#!/usr/bin/env python
"""AMC 2023 recording -> controller_data/amc23_{train,val,test}_controller.csv + _traces.csv.

Reads the one recording (data/results/amc23/) and writes each question's traces to its split
(amc23_train / val / test.jsonl: 24 / 8 / 8 questions). Same columns as GSM8K and MATH, with level
empty and subject 'amc'. A recording still running is exported up to its last finished trace.

Answers are graded from their answer box.

Run:  ~/.venvs/vllm/bin/python generation_amc23/src/export_amc23_csv.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths_amc23  # noqa: E402,F401  (points the MATH code at generation_amc23/data)
import export_math_csv as e  # noqa: E402
from common import read_jsonl  # noqa: E402
from grade_math import grade  # noqa: E402

REFS = {q["question_id"]: q["reference"] for q in read_jsonl(e.DATA / "amc23.jsonl")}
recorded_rows = e.jsonl_rows


def graded_rows(path):
    """Rows of a jsonl file, with pred / valid / correct taken from the answer box."""
    for r in recorded_rows(path):
        if "answer_text" in r:
            g = grade(r["answer_text"], REFS[r["question_id"]])
            r.update(pred=g["pred"], valid=g["valid"], correct=g["correct"])
        yield r


e.jsonl_rows = graded_rows


def main(out_dir=e.OUT_DIR):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(e.MODEL)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        e.export(f"amc23_{split}", split, tok, out_dir, results_set="amc23")


if __name__ == "__main__":
    main()
