#!/usr/bin/env python
"""Write generation_amc23/data/: amc23.jsonl (AMC 2023, all 40 questions: the set that is recorded)
and its split BY QUESTION, amc23_train.jsonl (24), amc23_val.jsonl (8), amc23_test.jsonl (8),
from a seed-2026 shuffle (test = first 8, val = next 8, train = the rest).

Source: math-ai/amc23 (downloaded the first time, a few kB). Every row: question_id
("amc23-<id>"), set, question, reference (the answer string), level (empty: AMC has none),
subject ("amc"). Existing files are left unchanged.

Run:  ~/.venvs/vllm/bin/python generation_amc23/src/prepare_amc23.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths_amc23 import DATA  # noqa: E402
from common import append_jsonl  # noqa: E402  (generation_gsm8k/src)


SEED = 2026
TEST_SIZE = VAL_SIZE = 8


def amc23():
    from datasets import load_dataset
    ds = load_dataset("math-ai/amc23")
    ds = ds[list(ds)[0]]
    rows = []
    for i, ex in enumerate(ds):
        q = ex.get("question") or ex.get("problem")
        rows.append(dict(question_id=f"amc23-{ex.get('id', i)}", set="amc23", question=q,
                         reference=str(ex["answer"]), level=None, subject="amc"))
    return rows


def write(name, rows):
    out = DATA / f"{name}.jsonl"
    if out.exists():
        print(f"{out} already exists, left unchanged (delete it to draw it again).")
        return
    append_jsonl(out, rows)
    print(f"Wrote {len(rows)} questions to {out}")


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    if not (DATA / "amc23.jsonl").exists():
        rows = amc23()
        if any(not r["reference"] for r in rows):
            raise SystemExit("amc23: a reference answer is missing.")
        write("amc23", rows)
    rows = [json.loads(l) for l in open(DATA / "amc23.jsonl")]
    if len(rows) != 40:
        raise SystemExit(f"amc23.jsonl has {len(rows)} questions, expected 40.")
    order = list(range(40))
    random.Random(SEED).shuffle(order)
    parts = dict(test=order[:TEST_SIZE], val=order[TEST_SIZE:TEST_SIZE + VAL_SIZE], train=order[TEST_SIZE + VAL_SIZE:])
    for split, idx in parts.items():
        write(f"amc23_{split}", [dict(rows[i], set=f"amc23_{split}") for i in idx])


if __name__ == "__main__":
    main()
