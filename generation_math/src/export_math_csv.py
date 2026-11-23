#!/usr/bin/env python
"""MATH / AMC recordings -> the same two CSVs as GSM8K (generation_gsm8k/src/export_controller_csv.py),
one pair per recorded set (= one split), in controller_data/ at the repo root (shared by all datasets):

  <set>_controller.csv  one row per (trace, budget): budget 0 = 'direct', then every cut
  <set>_traces.csv      one row per trace, with its full reasoning text
Both carry trace_id = "<question_id>#<sample>". The columns are the GSM8K ones, in the same order,
plus two at the end: level (MATH difficulty 1-5; empty for AMC) and subject. A cut a trace did not
reach uses its end answer (from_end = 1). Helpers (hedges, tail repetition, trace_id) are imported
from the GSM8K export, so the FULL features are computed the same way.

Default sets (a set not recorded yet is skipped; a set still being recorded is exported as far as
it got: only traces whose end answer is written):
  math_train (split train), math_val (val), math_test (test)
  -> controller_data/math_train_controller.csv, math_train_traces.csv, ... math_test_traces.csv
(AMC 2023 is exported by generation_amc23/src/export_amc23_csv.py, which runs this script.)
Other sets with --sets NAME:SPLIT ..., e.g. --sets math_pilot:train.

Run:  ~/.venvs/vllm/bin/python generation_math/src/export_math_csv.py
(generates nothing on the GPU).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grade_math import DATA  # noqa: E402  (puts generation_gsm8k/src on the path)
from common import MODEL, read_jsonl  # noqa: E402
from export_controller_csv import (CONTROLLER_COLUMNS, DIRECT, HEDGES, NUMBER, THINK,  # noqa: E402
                                   TRACE_COLUMNS, generation_fields, model_info, r4, tail_repetition,
                                   trace_id)

COLS = CONTROLLER_COLUMNS + ["level", "subject"]
TRACE_COLS = TRACE_COLUMNS[:-1] + ["level", "subject", "reasoning_text"]
DEFAULT = [("math_train", "train"), ("math_val", "val"), ("math_test", "test")]
OUT_DIR = DATA.parents[1] / "controller_data"       # <repo>/controller_data, outside every benchmark


def jsonl_rows(path):
    """Rows of a jsonl file; a last line cut off by a run still writing is skipped."""
    if not Path(path).exists():
        return
    with open(path) as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def export(set_name, split, tok, out_dir, results_set=None):
    """results_set: the recording to read when it holds more questions than this set (AMC: one
    recording, three splits); only the traces of this set's questions are exported."""
    results_set = results_set or set_name
    name = "amc23" if set_name.startswith("amc") else "math"   # the 'dataset' column
    d = DATA / "results" / results_set / THINK
    qs = {q["question_id"]: q for q in read_jsonl(DATA / f"{set_name}.jsonl")}
    if not qs or not (d / "traces.jsonl").exists():
        print(f"{set_name}: not recorded yet, skipped")
        return
    out, out_tr = out_dir / f"{set_name}_controller.csv", out_dir / f"{set_name}_traces.csv"
    n_rows = n_tr = 0
    with open(out, "w", newline="") as f, open(out_tr, "w", newline="") as ft:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        wt = csv.DictWriter(ft, fieldnames=TRACE_COLS)
        wt.writeheader()
        forced = {}
        for a in jsonl_rows(d / "answers.jsonl"):
            forced.setdefault((a["question_id"], a["sample"]), {})[a["cut"]] = a
        ends = {}
        for e in jsonl_rows(d / "end_answers.jsonl"):
            e.pop("reasoning_text", None)
            ends[(e["question_id"], e["sample"])] = e
        direct = {(a["question_id"], a["sample"]): a
                  for a in jsonl_rows(DATA / "results" / results_set / DIRECT / "answers.jsonl")}
        direct_tr = {(t["question_id"], t["sample"]): t                 # the settings of each direct answer
                     for t in jsonl_rows(DATA / "results" / results_set / DIRECT / "traces.jsonl")}
        minfo = model_info()
        k_tr = 0
        for t in jsonl_rows(d / "traces.jsonl"):
            key = (t["question_id"], t["sample"])
            if key not in ends or t["question_id"] not in qs:   # end answer not written yet / other split
                continue
            q = qs[t["question_id"]]
            tid = trace_id(*key)
            base = dict(trace_id=tid, dataset=name, split=split, question_id=t["question_id"],
                        question=q["question"], reference=q["reference"], prompt_tokens=t["prompt_tokens"],
                        n_numbers=len(NUMBER.findall(q["question"])), sample=t["sample"],
                        level=q.get("level"), subject=q.get("subject"))
            ids, n = t["reasoning_ids"], t["reasoning_tokens"]
            gen = {**minfo, **generation_fields(t)}
            wt.writerow(dict(gen, trace_id=tid, dataset=name, split=split,
                             question_id=t["question_id"], question=q["question"], reference=q["reference"],
                             sample=t["sample"], end_kind=t["end_kind"], reasoning_tokens=n,
                             loop_at=t["loop_at"], level=q.get("level"), subject=q.get("subject"),
                             reasoning_text=tok.decode(ids, skip_special_tokens=True)))
            k_tr += 1
            a0 = direct.get(key)
            if a0 is not None:
                w.writerow(dict({**minfo, **generation_fields(direct_tr[key]), **base}, config=DIRECT, budget=0, end_kind=None, trace_reasoning_tokens=0,
                                reached_budget=0, finish_reason="stop", reasoning_tokens_used=0,
                                answer_tokens=a0["answer_tokens"], total_tokens=a0["total_tokens"],
                                answer_text=a0["answer_text"], pred=a0["pred"], valid=a0["valid"],
                                correct=a0["correct"], box_tokens=a0.get("box_tokens"),
                                box_confidence=r4(a0.get("box_confidence")),
                                answer_confidence=r4(a0.get("answer_confidence")),
                                loop_at=None, loop_by_budget=0, hedge_count=0, hedge_per100=0.0,
                                tail_repetition=0.0, from_end=0))
                n_rows += 1
            cut_rows, end = forced.get(key, {}), ends[key]
            for c in t["cuts"]:
                a = cut_rows[c] if c < n else dict(end, from_end=1)
                used = min(n, c)
                reached = int(n > c or (n == c and t["end_kind"] == "cap"))
                hedges = len(HEDGES.findall(tok.decode(ids[:used], skip_special_tokens=True)))
                w.writerow(dict({**gen, **base}, config=THINK, budget=c, end_kind=t["end_kind"], trace_reasoning_tokens=n,
                                reached_budget=reached, finish_reason="length" if reached else "stop",
                                reasoning_tokens_used=used, answer_tokens=a["answer_tokens"],
                                total_tokens=used + a["answer_tokens"], answer_text=a["answer_text"],
                                pred=a["pred"], valid=a["valid"], correct=a["correct"],
                                box_tokens=a.get("box_tokens"), box_confidence=r4(a.get("box_confidence")),
                                answer_confidence=r4(a.get("answer_confidence")), loop_at=t["loop_at"],
                                loop_by_budget=int(t["loop_at"] is not None and t["loop_at"] <= used),
                                hedge_count=hedges, hedge_per100=round(100 * hedges / used, 3) if used else 0.0,
                                tail_repetition=round(tail_repetition(ids[:used]), 4),
                                from_end=int(a.get("from_end", 0))))
                n_rows += 1
        n_tr += k_tr
        print(f"{set_name} ({split}): {k_tr} traces of {len(qs)} questions")
    print(f"Wrote {n_rows} rows to {out}\nWrote {n_tr} traces to {out_tr}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sets", nargs="+", help="NAME:SPLIT pairs (instead of the default sets)")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    a = p.parse_args()
    jobs = [tuple(s.split(":")) for s in a.sets] if a.sets else DEFAULT
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    for set_name, split in jobs:
        export(set_name, split, tok, out_dir)


if __name__ == "__main__":
    main()
