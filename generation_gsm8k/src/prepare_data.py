#!/usr/bin/env python
"""Write the question sets of the benchmark into generation_gsm8k/data/.

  data/config_exp.jsonl  100 questions drawn from the GSM8K TRAIN portion (seed 2026).
                   Used to choose the config (the sweep). Never used for the final number.
  data/train.jsonl, val.jsonl, test.jsonl
                   the 1,319 questions of the official GSM8K TEST portion, split BY QUESTION
                   (all traces of a question are in one split) for the controller (the LLM is
                   frozen and never trained on them). Seed-2026 shuffle of the 1,319:
                     test  = the first 200
                     val   = the next 200
                     train = the other 919
                   Each file keeps the order of that shuffle.

Choosing the config on the test questions and then reporting the test score would make the score
look better than it is (the config would be tuned to those exact questions), hence two sets.

Run:  ~/.venvs/vllm/bin/python generation_gsm8k/src/prepare_data.py
Uses the GSM8K copy already in ~/.cache/huggingface/datasets (no new download if it is there).
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, append_jsonl, reference_number  # noqa: E402

CONFIG_EXP_SIZE = 100
SEED = 2026
TEST_SIZE = VAL_SIZE = 200


def split_order():
    """The seed-2026 shuffle of the 1,319 GSM8K test questions -> {split: [index, ...]}."""
    idx = list(range(1319))
    random.Random(SEED).shuffle(idx)
    return {"test": idx[:TEST_SIZE], "val": idx[TEST_SIZE:TEST_SIZE + VAL_SIZE],
            "train": idx[TEST_SIZE + VAL_SIZE:]}


def main():
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")
    DATA.mkdir(parents=True, exist_ok=True)

    idx = list(range(len(ds["train"])))
    random.Random(SEED).shuffle(idx)
    assert len(ds["test"]) == 1319
    sets = {"config_exp": [("train", i) for i in idx[:CONFIG_EXP_SIZE]]}
    for split, ids in split_order().items():
        sets[split] = [("test", i) for i in ids]
    for name, items in sets.items():
        out = DATA / f"{name}.jsonl"
        if out.exists():
            print(f"{out} already exists, left unchanged (delete it to draw it again).")
            continue
        rows = []
        for source, i in items:
            ex = ds[source][i]
            rows.append(dict(question_id=f"gsm8k-{source}-{i}", set=name, question=ex["question"],
                             reference=reference_number(ex["answer"])))
        if any(r["reference"] is None for r in rows):
            raise SystemExit("A reference answer could not be read as a number.")
        append_jsonl(out, rows)
        print(f"Wrote {len(rows)} questions to {out}")


if __name__ == "__main__":
    main()
