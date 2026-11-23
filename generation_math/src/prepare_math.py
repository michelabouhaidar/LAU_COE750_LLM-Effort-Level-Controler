#!/usr/bin/env python
"""Write the question sets of generation_math into generation_math/data/.

  math_train.jsonl  MATH train portion, stratified by level: --train-per-level per level 1-5
                    (default 80 -> 400), subjects interleaved inside each level.
  math_val.jsonl    the next --val-per-level per level (default 20 -> 100), no overlap.
  math_pilot.jsonl  the next --pilot-per-level per level (default 4 -> 20): a short run to
                    measure trace lengths and accuracy before the long ones.
  math500.jsonl     MATH-500 (the standard 500-question subset of the MATH test portion).
  math_test.jsonl   --test-size of MATH-500 (default 200), stratified by level (seed 2026).

MATH train and MATH-500 come from different portions (train / test), so no question is in two
sets. Every row: question_id, set, question, reference (the answer string), level, subject.

Run:  ~/.venvs/vllm/bin/python generation_math/src/prepare_math.py
MATH train is read from the copy already in ~/.cache/huggingface (EleutherAI/hendrycks_math).
MATH-500 (HuggingFaceH4/MATH-500) is downloaded the first time (small: a few hundred kB).
AMC 2023 has its own folder: generation_amc23/src/prepare_amc23.py. A set whose file exists is left unchanged.
"""
import argparse
import random
from collections import defaultdict

from grade_math import DATA, last_boxed
from common import append_jsonl  # noqa: E402  (generation_gsm8k/src, path set in grade_math)

SEED = 2026
SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus"]


def stratified_math_train():
    """MATH train questions per level, subjects interleaved (round robin over shuffled subjects)."""
    from datasets import load_dataset
    by_level = defaultdict(lambda: defaultdict(list))
    for subj in SUBJECTS:
        ds = load_dataset("EleutherAI/hendrycks_math", subj)["train"]
        for i, ex in enumerate(ds):
            m = ex["level"].replace("Level ", "")
            ref = last_boxed(ex["solution"])
            if not m.isdigit() or not ref:
                continue
            by_level[int(m)][subj].append(dict(
                question_id=f"math-train-{subj}-{i}", question=ex["problem"], reference=ref,
                level=int(m), subject=subj))
    rng = random.Random(SEED)
    out = {}
    for lvl in range(1, 6):
        lists = [by_level[lvl][s] for s in SUBJECTS]
        for l in lists:
            rng.shuffle(l)
        order = []
        while any(lists):
            for l in lists:
                if l:
                    order.append(l.pop())
        out[lvl] = order
    return out


def stratified_take(rows, n, seed=SEED):
    """n rows spread evenly over levels 1-5 (as even as the counts allow)."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r["level"]].append(r)
    for l in by.values():
        rng.shuffle(l)
    out, lv = [], sorted(by)
    while len(out) < n and any(by[l] for l in lv):
        for l in lv:
            if by[l] and len(out) < n:
                out.append(by[l].pop())
    return out


def math500():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500")["test"]
    rows = []
    for ex in ds:
        uid = ex["unique_id"].replace("test/", "").replace(".json", "").replace("/", "-")
        rows.append(dict(question_id=f"math500-{uid}", question=ex["problem"], reference=ex["answer"],
                         level=int(ex["level"]), subject=ex["subject"]))
    return rows


def write(name, rows):
    out = DATA / f"{name}.jsonl"
    if out.exists():
        print(f"{out} already exists, left unchanged (delete it to draw it again).")
        return
    if any(not r["reference"] for r in rows):
        raise SystemExit(f"{name}: a reference answer is missing.")
    append_jsonl(out, [dict(r, set=name) for r in rows])
    print(f"Wrote {len(rows)} questions to {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-per-level", type=int, default=80)
    p.add_argument("--val-per-level", type=int, default=20)
    p.add_argument("--pilot-per-level", type=int, default=4)
    p.add_argument("--test-size", type=int, default=200)
    p.add_argument("--only", nargs="+", choices=["math_train", "math_val", "math_pilot", "math500",
                                                  "math_test"], help="write only these sets")
    a = p.parse_args()
    want = set(a.only or ["math_train", "math_val", "math_pilot", "math500", "math_test"])
    DATA.mkdir(parents=True, exist_ok=True)

    if want & {"math_train", "math_val", "math_pilot"}:
        lv = stratified_math_train()
        t, v, pl = a.train_per_level, a.val_per_level, a.pilot_per_level
        sets = {"math_train": (0, t), "math_val": (t, t + v), "math_pilot": (t + v, t + v + pl)}
        for name, (i, j) in sets.items():
            if name in want:
                write(name, [r for l in range(1, 6) for r in lv[l][i:j]])
    if want & {"math500", "math_test"}:
        rows = math500()
        if "math500" in want:
            write("math500", rows)
        if "math_test" in want:
            write("math_test", stratified_take(rows, a.test_size))


if __name__ == "__main__":
    main()
