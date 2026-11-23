#!/usr/bin/env python
"""Record Qwen3.5-2B on MATH / MATH-500 with the SAME recording as GSM8K.
(AMC 2023 runs this same script from generation_amc23/src/run_amc23.py, with its own data folder.)

Identical to generation_gsm8k/src/sweep/run_sweep.py (it imports the same server client, prompt,
forced-answer cue, loop rule and configs), so the data of every dataset has one format:
  * one reasoning trace per (question, sample), streamed; it ends at its own '</think>'
    (natural), at the loop flag with --stop-on-loop (loop), or at --b-max (cap);
  * a forced answer ("</think> The final answer is \\boxed{") at every cut the trace REACHED
    (cut < its end), cuts 256 512 1024 2048 4096 6144 8192 12288 16384 by default, and ONE
    answer at its end; nothing is generated or stored after the end (expand_answers fills the
    higher cuts in from the end answer when the data is read).
Differences from GSM8K, and why:
  * grading: answers are expressions, graded with Math-Verify (grade_math.py); pred is the
    normalised box content, so equal answers can be grouped;
  * the forced answer may use FORCED_ANSWER_MAX = 32 tokens (GSM8K: 16), because answers like
    (-\\infty,-2]\\cup[3,\\infty) need more; confidence is still measured on the box only;
  * a question whose prompt does not fit the server context with --b-max is skipped and listed
    (MATH prompts are longer than GSM8K's).

Output:  generation_math/data/results/<set>/<config>/ traces.jsonl, answers.jsonl, end_answers.jsonl
Stop (Ctrl-C) and start again at any time: finished traces are skipped.

Commands
  run      generate and grade
  report   accuracy per cut and per level, how traces ended, and the L/M/H levels suggested by
           the rule (H = top cut; L, M = first cut reaching 85% / 97% of the accuracy at H).
           Generates nothing.
"""
import argparse
import asyncio
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grade_math import DATA, GRADER, grade, last_boxed  # noqa: E402  (also puts generation_gsm8k/src on the path)
from common import (BASE_URL, FORCE_CUE, FORCE_SUFFIX, MODEL, PROMPT, append_jsonl,  # noqa: E402
                    complete, complete_trace, expand_answers, load_answers, loop_point,
                    provenance, read_jsonl, seed_for)
from configs import CONFIGS  # noqa: E402  (generation_gsm8k/src/sweep/configs.py)

B_MAX = 16384
CUTS = (256, 512, 1024, 2048, 4096, 6144, 8192, 12288, 16384)   # the same as the GSM8K recordings
SAMPLES = 6
FORCED_ANSWER_MAX = 32


def sets_available():
    return sorted(p.stem for p in DATA.glob("*.jsonl"))


def out_dir(set_name, config):
    return DATA / "results" / set_name / config


class Ctx:
    def __init__(self, concurrency, stop_on_loop):
        from openai import AsyncOpenAI
        from transformers import AutoTokenizer
        self.client = AsyncOpenAI(base_url=BASE_URL, api_key="EMPTY", timeout=7200)
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.think_end = self.tok.convert_tokens_to_ids("</think>")
        self.suffix_ids = self.tok.encode(FORCE_SUFFIX, add_special_tokens=False)
        self.direct_ids = self.tok.encode("The final answer is " + FORCE_CUE, add_special_tokens=False)
        self.stop_on_loop = stop_on_loop
        self.sem = asyncio.Semaphore(concurrency)
        self.sem_short = asyncio.Semaphore(concurrency)
        self.lock = asyncio.Lock()
        self.prov = provenance()
        self.score = defaultdict(lambda: [0, 0])
        self.bar = None

    def prompt_ids(self, question, thinking):
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(q=question)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        return self.tok.encode(text, add_special_tokens=False)

    def show(self, rows):
        for r in rows:
            s = self.score[(r["config"], r["cut"])]
            s[0] += 1
            s[1] += r["correct"]
        if self.bar is not None:
            top = {}
            for (cfg, cut), (n, c) in self.score.items():
                if cfg not in top or (cut or 0) >= top[cfg][0]:
                    top[cfg] = (cut or 0, 100 * c / n)
            self.bar.set_postfix_str("  ".join(f"{k}: {v[1]:.0f}%" for k, v in top.items()))


def box_end(ctx, ids):
    """Answer tokens up to and including the '}' closing the forced \\boxed{ (nested braces counted)."""
    for i in range(len(ids)):
        depth = 1
        for ch in ctx.tok.decode(ids[:i + 1], skip_special_tokens=True):
            depth += (ch == "{") - (ch == "}")
            if depth == 0:
                return i + 1
    return len(ids)


def answer_fields(ctx, a, reference):
    text = FORCE_CUE + ctx.tok.decode(a["ids"], skip_special_tokens=True)
    lp = a["logprobs"] or []
    nb = box_end(ctx, a["ids"])
    return dict(answer_text=text, answer_tokens=len(a["ids"]),
                answer_confidence=math.exp(sum(lp) / len(lp)) if lp else None,
                box_tokens=nb, box_closed=int(last_boxed(text) is not None),
                box_confidence=math.exp(sum(lp[:nb]) / nb) if lp and nb else None,
                answer_logprobs=[round(x, 4) for x in lp],
                **grade(text, reference))


async def thinking_sample(ctx, q, cfg_name, cfg, k, cuts, b_max, d):
    sampling = cfg["sampling"]
    P = ctx.prompt_ids(q["question"], thinking=True)
    qid = q["question_id"]
    seed = seed_for(qid, k)
    async with ctx.sem:
        try:
            r = await complete_trace(ctx.client, P, b_max, seed, sampling, [ctx.think_end], ctx.stop_on_loop)
        except Exception as e:
            print(f"\n[trace failed] {cfg_name} {qid} #{k}: {e}")
            return
    ids = r["ids"]
    if ids and ids[-1] == ctx.think_end:
        ids = ids[:-1]
    n = len(ids)
    natural = r["finish"] == "stop" and r["stop_reason"] is not None
    end_kind = "loop" if r["finish"] == "loop" else "natural" if natural else "cap"
    loop_at = r["loop_at"] if end_kind == "loop" else loop_point(ids)
    trace = dict(question_id=qid, config=cfg_name, sample=k, seed=seed, level=q.get("level"),
                 subject=q.get("subject"), prompt_tokens=len(P), reasoning_tokens=n,
                 natural_end=int(natural), end_kind=end_kind, stop_on_loop=int(ctx.stop_on_loop),
                 cuts=list(cuts), eos_in_reasoning=int(r["finish"] == "stop" and r["stop_reason"] is None),
                 loop_detected=int(loop_at is not None), loop_at=loop_at,
                 reasoning_ids=ids, b_max=b_max, sampling=sampling, **ctx.prov)
    try:
        rows = []
        for c in cuts:
            if c >= n:
                break
            async with ctx.sem_short:
                a = await complete(ctx.client, P + ids[:c] + ctx.suffix_ids, FORCED_ANSWER_MAX,
                                   seed_for(qid, k, c), sampling, logprobs=1)
            f = answer_fields(ctx, a, q["reference"])
            rows.append(dict(question_id=qid, config=cfg_name, sample=k, cut=c,
                             reasoning_tokens_used=c, finished_by_cut=0, stopped_by_loop=0,
                             loop_by_cut=int(loop_at is not None and loop_at <= c),
                             total_tokens=c + f["answer_tokens"], reasoning_text=None, **f))
        async with ctx.sem_short:
            a = await complete(ctx.client, P + ids + ctx.suffix_ids, FORCED_ANSWER_MAX,
                               seed_for(qid, k, "end"), sampling, logprobs=1)
        f = answer_fields(ctx, a, q["reference"])
        end = dict(question_id=qid, config=cfg_name, sample=k, end_kind=end_kind, reasoning_tokens=n,
                   total_tokens=n + f["answer_tokens"],
                   reasoning_text=ctx.tok.decode(ids, skip_special_tokens=True), **f)
    except Exception as e:
        print(f"\n[answer failed] {cfg_name} {qid} #{k}: {e}")
        return
    async with ctx.lock:
        append_jsonl(d / "traces.jsonl", [trace])
        append_jsonl(d / "answers.jsonl", rows)
        append_jsonl(d / "end_answers.jsonl", [end])
        ctx.show(expand_answers(rows, [trace], [end]))


async def direct_sample(ctx, q, cfg_name, cfg, k, d):
    """No reasoning at all: thinking off, the answer starts with 'The final answer is \\boxed{'."""
    if not cfg.get("direct"):
        raise SystemExit(f"{cfg_name}: only thinking configs and 'direct' are supported here.")
    P = ctx.prompt_ids(q["question"], thinking=False) + ctx.direct_ids
    seed = seed_for(q["question_id"], k)
    async with ctx.sem_short:
        try:
            r = await complete(ctx.client, P, FORCED_ANSWER_MAX, seed, cfg["sampling"], logprobs=1)
        except Exception as e:
            print(f"\n[answer failed] {cfg_name} {q['question_id']} #{k}: {e}")
            return
    f = answer_fields(ctx, r, q["reference"])
    trace = dict(question_id=q["question_id"], config=cfg_name, sample=k, seed=seed, level=q.get("level"),
                 subject=q.get("subject"), prompt_tokens=len(P), reasoning_tokens=0,
                 natural_end=int(r["finish"] == "stop"), end_kind=None, eos_in_reasoning=0,
                 loop_detected=0, loop_at=None, reasoning_ids=[], b_max=0, sampling=cfg["sampling"], **ctx.prov)
    row = dict(question_id=q["question_id"], config=cfg_name, sample=k, cut=None, reasoning_tokens_used=0,
               finished_by_cut=1, loop_by_cut=0, total_tokens=len(r["ids"]), reasoning_text=None, **f)
    async with ctx.lock:
        append_jsonl(d / "traces.jsonl", [trace])
        append_jsonl(d / "answers.jsonl", [row])
        ctx.show([row])


async def cmd_run(args):
    qs = read_jsonl(DATA / f"{args.set}.jsonl")
    if not qs:
        raise SystemExit(f"No questions in {DATA / (args.set + '.jsonl')}. Run src/prepare_math.py first.")
    qs = qs[:args.limit] if args.limit else qs
    cuts = tuple(c for c in sorted(set(args.cuts) | {args.b_max}) if c <= args.b_max)
    ctx = Ctx(args.concurrency, args.stop_on_loop)
    srv = ctx.prov["server_max_len"]
    if srv is None:
        raise SystemExit("The server does not answer at " + BASE_URL + ". Start it first (VLLM-DOC.md).")
    room = srv - args.b_max - len(ctx.suffix_ids) - FORCED_ANSWER_MAX
    if room < 200:
        raise SystemExit(f"Server context {srv} is too small for --b-max {args.b_max}: restart it with "
                         f"LLM_MAX_LEN of at least {args.b_max + 1024} (VLLM-DOC.md).")
    fits = [q for q in qs if len(ctx.prompt_ids(q["question"], True)) <= room]
    if len(fits) < len(qs):
        skipped = [q["question_id"] for q in qs if q not in fits]
        print(f"{len(skipped)} questions skipped: prompt longer than {room} tokens "
              f"(server context {srv}, --b-max {args.b_max}): {skipped[:10]}{' ...' if len(skipped) > 10 else ''}")
    qs = fits

    tasks = []
    for name in args.configs:
        cfg = CONFIGS[name]
        d = out_dir(args.set, name)
        old = read_jsonl(d / "traces.jsonl")
        if cfg["thinking"] and old and {t.get("stop_on_loop", 0) for t in old} != {int(args.stop_on_loop)}:
            raise SystemExit(f"{d} has traces with a different --stop-on-loop setting; "
                             "move them aside or use the same setting.")
        done = {(t["question_id"], t["sample"]) for t in old}
        for q in qs:
            for k in range(args.samples):
                if (q["question_id"], k) in done:
                    continue
                tasks.append(thinking_sample(ctx, q, name, cfg, k, cuts, args.b_max, d) if cfg["thinking"]
                             else direct_sample(ctx, q, name, cfg, k, d))
    print(f"set={args.set}  questions={len(qs)}  samples={args.samples}  configs={args.configs}  grader={GRADER}")
    print(f"b_max={args.b_max}  cuts={cuts}  stop_on_loop={args.stop_on_loop}  -> {len(tasks)} samples to generate")

    from tqdm import tqdm
    ctx.bar = tqdm(total=len(tasks))
    t0 = time.time()

    async def wrapped(t):
        try:
            await t
        finally:
            ctx.bar.update(1)
    await asyncio.gather(*(wrapped(t) for t in tasks))
    ctx.bar.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    report(args.set, args.configs)


# ----------------------------------------------------------------------------- report

def suggest_levels(acc):
    """L/M/H by the rule in per-dataset levels: acc = {cut: accuracy}."""
    cuts = sorted(c for c in acc if c)
    if not cuts:
        return {}
    H = cuts[-1]
    if acc[H] <= 0:
        return {}
    first = lambda share: next(c for c in cuts if acc[c] >= share * acc[H])  # noqa: E731
    L, M = first(0.85), first(0.97)
    return dict(L=L, M=M, H=H)


def report(set_name, configs):
    import csv
    qs = {q["question_id"]: q for q in read_jsonl(DATA / f"{set_name}.jsonl")}
    lines = []
    for name in configs:
        d = out_dir(set_name, name)
        ans, traces = load_answers(d), read_jsonl(d / "traces.jsonl")
        if not ans:
            continue
        by_cut, by_lvl = defaultdict(list), defaultdict(lambda: defaultdict(list))
        for r in ans:
            by_cut[r["cut"]].append(r["correct"])
            by_lvl[qs.get(r["question_id"], {}).get("level")][r["cut"]].append(r["correct"])
        acc = {c: 100 * statistics.mean(v) for c, v in by_cut.items()}
        top = max(by_cut, key=lambda c: c or 0)
        think = [t for t in traces if t.get("end_kind")]
        line = dict(config=name, questions=len({t["question_id"] for t in traces}), samples=len(traces),
                    **{f"acc@{c or 'direct'}": round(acc[c], 1) for c in sorted(acc, key=lambda c: c or 0)})
        if think:
            lens = [t["reasoning_tokens"] for t in think]
            line.update(natural_pct=round(100 * statistics.mean(t["end_kind"] == "natural" for t in think), 1),
                        loop_stopped_pct=round(100 * statistics.mean(t["end_kind"] == "loop" for t in think), 1),
                        cap_pct=round(100 * statistics.mean(t["end_kind"] == "cap" for t in think), 1),
                        median_reasoning=round(statistics.median(lens)),
                        p95_reasoning=round(sorted(lens)[int(0.95 * (len(lens) - 1))]),
                        mean_tokens_at_top=round(statistics.mean(
                            r["total_tokens"] for r in ans if r["cut"] == top)),
                        **{f"levels_{k}": v for k, v in suggest_levels(acc).items()})
        lines.append(line)
        lv = [l for l in by_lvl if l is not None]
        if lv and think:
            print(f"\n{name}: accuracy (%) per MATH level and cut")
            cs = sorted(c for c in by_cut if c)
            print("level " + "".join(f"{c:>7}" for c in cs))
            for l in sorted(lv):
                print(f"{l:>5} " + "".join(f"{100 * statistics.mean(by_lvl[l][c]):7.1f}" if by_lvl[l][c] else
                                           f"{'':>7}" for c in cs))
    if not lines:
        print("Nothing recorded yet.")
        return
    keys = list(dict.fromkeys(k for l in lines for k in l))
    print()
    print("  ".join(f"{k:>14}" for k in keys))
    for l in lines:
        print("  ".join(f"{str(l.get(k, '')):>14}" for k in keys))
    out = DATA / "results" / set_name / "summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(lines)
    print("\nacc@N = accuracy (%) with thinking limited to N tokens (one run); acc@direct = no reasoning.")
    print("levels_* = L/M/H by the rule: H = top cut, L / M = first cut at 85% / 97% of acc@H.")
    print(f"Saved: {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "report"):
        s = sub.add_parser(name)
        s.add_argument("--set", required=True, help=f"a file in {DATA}: {', '.join(sets_available()) or '(none yet)'}")
        s.add_argument("--configs", nargs="+", default=["t0.6_pp1.5"], choices=list(CONFIGS))
        if name == "run":
            s.add_argument("--limit", type=int, help="only the first N questions (trial run)")
            s.add_argument("--samples", type=int, default=SAMPLES, help=f"traces per question (default {SAMPLES})")
            s.add_argument("--b-max", type=int, default=B_MAX, help=f"longest trace (default {B_MAX})")
            s.add_argument("--cuts", type=int, nargs="+", default=list(CUTS))
            s.add_argument("--stop-on-loop", action="store_true",
                           help="stop a trace when its loop flag fires and answer there (end_kind 'loop')")
            s.add_argument("--concurrency", type=int, default=16, help="requests at the same time")
    args = p.parse_args()
    if args.cmd == "run":
        asyncio.run(cmd_run(args))
    else:
        report(args.set, args.configs)


if __name__ == "__main__":
    main()
