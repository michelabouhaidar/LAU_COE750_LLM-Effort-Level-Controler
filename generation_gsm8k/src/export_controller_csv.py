#!/usr/bin/env python
"""CSVs with everything the controller needs, from the recorded GSM8K run: one pair per split.

One row per (question, trace sample, budget):
  * budget = each cut 256 ... 16384 of the thinking config, and budget 0 = the 'direct' config
    (no reasoning) for the same sample number;
  * a cut the trace did not reach uses the trace's end answer (from_end = 1), exactly like the
    report (common.expand_answers): with that budget the model would have stopped there.
The replay environment of rl_methods/ reads 'a run at level l' as the row (question, sample, l).

Splits (controller only; the LLM is not trained), by question: data/train.jsonl (919),
val.jsonl (200), test.jsonl (200), written by prepare_data.py (seed-2026 shuffle of the 1,319
GSM8K test questions; test = the first 200). Each split is recorded in its own folder
data/results/<split>/ and exported to its own files.

Columns
  question:   dataset, split, question_id, question, reference, prompt_tokens, n_numbers
  run:        config, sample, budget, end_kind (natural / loop / cap of the whole trace),
              trace_reasoning_tokens, reached_budget (1 = still thinking at the budget, i.e. cut),
              finish_reason ('length' if cut by the budget, else 'stop': what an API returns),
              reasoning_tokens_used, answer_tokens, total_tokens (= billed output tokens)
  answer:     answer_text, pred, valid, correct, box_tokens, box_confidence, answer_confidence
  FULL-only:  loop_at, loop_by_budget, hedge_count, hedge_per100, tail_repetition
              (from the reasoning tokens up to the budget; not available from a closed API)
  bookkeeping: from_end
  model:      model, model_org, model_family, model_size, model_type, model_architecture, model_dtype,
              model_layers, model_hidden_size, model_intermediate_size, model_attention_heads,
              model_kv_heads, model_head_dim, model_full_attention_interval, model_vocab_size,
              model_max_context, model_open_weights, model_reasoning, served_by
              (from the id and the local config.json of the served model)
  generation: thinking, seed, temperature, top_p, top_k, min_p, presence_penalty, weights_hash,
              vllm_version, server_max_len, b_max, cuts, stop_on_loop, natural_end, eos_in_reasoning,
              loop_detected: the settings of the run behind the row (for budget 0, the direct run's)

Second file, gsm8k_<split>_traces.csv: one row per trace (question, sample) with its full reasoning text
and everything about how it was generated:
  trace_id, dataset, split, question_id, question, reference, sample, end_kind, reasoning_tokens, loop_at,
  the model columns, the generation columns (above), config, prompt_tokens, reasoning_text
(The direct answers have no reasoning; their settings are in data/results/<split>/direct/traces.jsonl.)
Both files carry trace_id = "<question_id>#<sample>": every row of gsm8k_<split>_controller.csv joins to
exactly one row of the traces file of the same split (9 budget rows + 1 direct row per trace;
the direct row (budget 0) shares the sample number but used no reasoning). The reasoning
available at a budget is the first reasoning_tokens_used tokens of that text (re-tokenize with the
Qwen tokenizer to cut it exactly).

Run:  ~/.venvs/vllm/bin/python generation_gsm8k/src/export_controller_csv.py [--splits train val test]
Writes controller_data/gsm8k_{train,val,test}_controller.csv and gsm8k_{train,val,test}_traces.csv
(controller_data/ is at the repo root, shared by all datasets; generates nothing on the GPU).
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "sweep"))
from common import DATA, LOOP_NGRAM, LOOP_WINDOW, MODEL, read_jsonl  # noqa: E402
from configs import CONFIGS  # noqa: E402

SPLITS = ("train", "val", "test")
OUT_DIR = DATA.parents[1] / "controller_data"       # <repo>/controller_data, outside every benchmark
THINK, DIRECT = "t0.6_pp1.5", "direct"
HEDGES = re.compile(r"\b(wait|alternatively|hmm|let me check|actually|but maybe)\b", re.I)
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")

COLUMNS = ["trace_id", "dataset", "split", "question_id", "question", "reference", "prompt_tokens", "n_numbers",
           "config", "sample", "budget", "end_kind", "trace_reasoning_tokens", "reached_budget",
           "finish_reason", "reasoning_tokens_used", "answer_tokens", "total_tokens",
           "answer_text", "pred", "valid", "correct", "box_tokens", "box_confidence", "answer_confidence",
           "loop_at", "loop_by_budget", "hedge_count", "hedge_per100", "tail_repetition", "from_end"]

# the model: its name parts and its architecture, read from the model's own config.json (HF cache)
MODEL_COLUMNS = ["model", "model_org", "model_family", "model_size", "model_type", "model_architecture",
                 "model_dtype", "model_layers", "model_hidden_size", "model_intermediate_size",
                 "model_attention_heads", "model_kv_heads", "model_head_dim", "model_full_attention_interval",
                 "model_vocab_size", "model_max_context", "model_open_weights", "model_reasoning", "served_by"]

# how each trace was generated (copied from its row in traces.jsonl), so the CSV alone can reproduce it
GENERATION_COLUMNS = ["config", "thinking", "seed", "temperature", "top_p", "top_k", "min_p", "presence_penalty",
                      "model", "weights_hash", "vllm_version", "server_max_len", "b_max", "cuts",
                      "stop_on_loop", "prompt_tokens", "natural_end", "eos_in_reasoning", "loop_detected"]

RUN_COLUMNS = list(dict.fromkeys(MODEL_COLUMNS + GENERATION_COLUMNS))   # model + generation, no repeats
CONTROLLER_COLUMNS = COLUMNS + [c for c in RUN_COLUMNS if c not in COLUMNS]
TRACE_COLUMNS = (["trace_id", "dataset", "split", "question_id", "question", "reference", "sample",
                  "end_kind", "reasoning_tokens", "loop_at"] + RUN_COLUMNS + ["reasoning_text"])


def model_info(model=MODEL):
    """MODEL_COLUMNS for one model: org / family / size from its id ('Qwen/Qwen3.5-2B'), the rest from
    the config.json of the local copy that vLLM served (no download)."""
    from huggingface_hub import hf_hub_download
    cfg = json.load(open(hf_hub_download(model, "config.json", local_files_only=True)))
    tc = cfg.get("text_config", cfg)                  # multimodal configs keep the LLM part here
    org, name = model.split("/")
    family, size = name.rsplit("-", 1)
    return dict(model=model, model_org=org, model_family=family, model_size=size,
                model_type=cfg.get("model_type"), model_architecture=cfg["architectures"][0],
                model_dtype=tc.get("dtype") or tc.get("torch_dtype"), model_layers=tc.get("num_hidden_layers"),
                model_hidden_size=tc.get("hidden_size"), model_intermediate_size=tc.get("intermediate_size"),
                model_attention_heads=tc.get("num_attention_heads"), model_kv_heads=tc.get("num_key_value_heads"),
                model_head_dim=tc.get("head_dim"), model_full_attention_interval=tc.get("full_attention_interval"),
                model_vocab_size=tc.get("vocab_size"), model_max_context=tc.get("max_position_embeddings"),
                model_open_weights=1, model_reasoning=1, served_by="vllm")


def generation_fields(t):
    """The generation settings of one trace row of traces.jsonl, flattened for the CSV."""
    g = {k: t.get(k) for k in GENERATION_COLUMNS}
    g.update({k: t["sampling"].get(k) for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty")})
    g["cuts"] = " ".join(map(str, t.get("cuts", [])))
    g["thinking"] = int(CONFIGS[t["config"]]["thinking"])
    return g


def trace_id(question_id, sample):
    """The key that links a row of <set>_controller.csv to its trace in <set>_traces.csv."""
    return f"{question_id}#{sample}"


def tail_repetition(ids):
    """Share of repeated 20-token pieces in the last 2,048 tokens (the loop rule's measure)."""
    w = ids[-LOOP_WINDOW:]
    if len(w) < LOOP_NGRAM + 1:
        return 0.0
    pieces = [tuple(w[i:i + LOOP_NGRAM]) for i in range(len(w) - LOOP_NGRAM + 1)]
    return 1 - len(set(pieces)) / len(pieces)


def r4(x):
    return None if x is None else round(x, 4)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    args = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    for split in args.splits:
        export(split, tok)


def export(split, tok):
    from tqdm import tqdm
    qs = {q["question_id"]: q for q in read_jsonl(DATA / f"{split}.jsonl")}
    d = DATA / "results" / split / THINK
    if not qs or not (d / "traces.jsonl").exists():
        print(f"{split}: not recorded yet, skipped")
        return
    forced = {}                                       # (qid, sample) -> {cut: answer} for cuts the trace reached
    for a in read_jsonl(d / "answers.jsonl"):
        forced.setdefault((a["question_id"], a["sample"]), {})[a["cut"]] = a
    ends = {}                                         # the answer at each trace's end (without the reasoning text)
    with open(d / "end_answers.jsonl") as f:
        for line in f:
            e = json.loads(line)
            e.pop("reasoning_text", None)              # the text is written from the token ids below
            ends[(e["question_id"], e["sample"])] = e
    direct = {(a["question_id"], a["sample"]): a
              for a in read_jsonl(DATA / "results" / split / DIRECT / "answers.jsonl")}
    direct_tr = {(t["question_id"], t["sample"]): t                 # the settings of each direct answer
                 for t in read_jsonl(DATA / "results" / split / DIRECT / "traces.jsonl")}
    minfo = model_info()

    out = OUT_DIR / f"gsm8k_{split}_controller.csv"
    out_tr = OUT_DIR / f"gsm8k_{split}_traces.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with open(out, "w", newline="") as f, open(out_tr, "w", newline="") as ft:
        w = csv.DictWriter(f, fieldnames=CONTROLLER_COLUMNS)
        w.writeheader()
        wt = csv.DictWriter(ft, fieldnames=TRACE_COLUMNS)
        wt.writeheader()
        for line in tqdm(open(d / "traces.jsonl"), total=len(ends), desc=split):
            t = json.loads(line)
            key = (t["question_id"], t["sample"])
            q = qs[t["question_id"]]
            base = dict(trace_id=trace_id(t["question_id"], t["sample"]), dataset="gsm8k", split=split, question_id=t["question_id"],
                        question=q["question"], reference=q["reference"], prompt_tokens=t["prompt_tokens"],
                        n_numbers=len(NUMBER.findall(q["question"])), sample=t["sample"])
            ids, n = t["reasoning_ids"], t["reasoning_tokens"]
            gen = {**minfo, **generation_fields(t)}
            wt.writerow(dict(gen, trace_id=base["trace_id"], dataset="gsm8k", split=base["split"],
                             question_id=t["question_id"], question=q["question"], reference=q["reference"],
                             sample=t["sample"], end_kind=t["end_kind"], reasoning_tokens=n, loop_at=t["loop_at"],
                             reasoning_text=tok.decode(ids, skip_special_tokens=True)))
            a0 = direct.get(key)
            if a0 is not None:                       # budget 0: no reasoning at all
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
                a = cut_rows[c] if c < n else dict(end, from_end=1)   # as common.expand_answers
                used = min(n, c)
                reached = int(n > c or (n == c and t["end_kind"] == "cap"))
                text = tok.decode(ids[:used], skip_special_tokens=True)
                hedges = len(HEDGES.findall(text))
                w.writerow(dict({**gen, **base}, config=THINK, budget=c, end_kind=t["end_kind"], trace_reasoning_tokens=n,
                                reached_budget=reached, finish_reason="length" if reached else "stop",
                                reasoning_tokens_used=used, answer_tokens=a["answer_tokens"],
                                total_tokens=used + a["answer_tokens"], answer_text=a["answer_text"],
                                pred=a["pred"], valid=a["valid"], correct=a["correct"],
                                box_tokens=a.get("box_tokens"), box_confidence=r4(a.get("box_confidence")),
                                answer_confidence=r4(a.get("answer_confidence")),
                                loop_at=t["loop_at"],
                                loop_by_budget=int(t["loop_at"] is not None and t["loop_at"] <= used),
                                hedge_count=hedges, hedge_per100=round(100 * hedges / used, 3) if used else 0.0,
                                tail_repetition=round(tail_repetition(ids[:used]), 4),
                                from_end=int(a.get("from_end", 0))))
                n_rows += 1
    print(f"Wrote {n_rows} rows to {out}")
    print(f"Wrote {len(ends)} traces to {out_tr}")


if __name__ == "__main__":
    main()
