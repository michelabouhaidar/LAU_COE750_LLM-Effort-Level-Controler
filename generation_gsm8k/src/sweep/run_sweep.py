#!/usr/bin/env python
"""Run Qwen3.5-2B on GSM8K under several configs and grade it at several reasoning budgets.

For every question, config and sample k (thinking configs):
  1. ONE reasoning trace is generated, stopped when the model closes it ('</think>') or at
     --b-max tokens.
  2. For every cut c in --cuts, the trace is cut after c tokens, the text
     "</think> The final answer is \\boxed{" is added and the model fills in the box (16 tokens).
     If the trace had already ended by itself before c, its own end is used instead.
  So one trace gives the answer at every budget: "what if the model had only c tokens to think?".
  The cut equal to --b-max is the full-budget answer.
For a no-thinking config: one answer of up to max_tokens (configs.py), graded on its last box.
For a direct config (direct=True): thinking off and the answer starts with
"The final answer is \\boxed{", so the model only writes the number: no reasoning at all.

Output, one folder per set and config:  generation_gsm8k/data/results/<set>/<config>/
  traces.jsonl   one line per trace: token ids, length, ended by itself or not, loop flag
  answers.jsonl  one line per (trace, cut): reasoning used, answer, graded
Everything can be stopped (Ctrl-C) and started again; finished traces are skipped.

Loop answer: if the loop flag fired at token L before the trace ended, one more forced answer is
asked at L (loop_answers.jsonl). That is exactly what "stop on loop" would have answered, so the
report can replay the loop guard without a second run.

Free answer (--free): after the trace, the model writes its answer on its own, with no cue,
up to FREE_MAX tokens, graded on its last box (free_answers.jsonl). A trace that ended by itself
continues right after its own '</think>'; a capped one gets '</think>' added. This is how the
model answers when left alone, next to the forced answer.

Commands
  run      generate and grade
  answers  add cuts (and loop answers) to traces already recorded: no new traces, only the
           16-token forced answers, so it is fast
  report   print the result table from what is recorded (generates nothing)

The server must be running, with a context that fits --b-max (see VLLM-DOC.md).
"""
import argparse
import asyncio
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (DATA, FORCE_CUE, FORCE_SUFFIX, FORCED_ANSWER_MAX, MODEL, BASE_URL, PROMPT,  # noqa: E402
                    append_jsonl, complete, complete_trace, expand_answers, grade, load_answers,
                    loop_point, provenance, read_jsonl, seed_for)
from configs import CONFIGS  # noqa: E402

B_MAX = 16384                                       # longest reasoning trace
CUTS = (512, 1024, 2048, 4096, 8192, 16384)         # budgets at which an answer is graded
SAMPLES = 4                                         # traces per question and config
FREE_MAX = 2048                                     # longest free answer after thinking
FREE_SUFFIX = "\n</think>\n\n"                       # closes a capped trace, no answer cue


def out_dir(set_name, config):
    return DATA / "results" / set_name / config


# --------------------------------------------------------------------------- generation

class Ctx:
    def __init__(self, concurrency):
        from openai import AsyncOpenAI
        from transformers import AutoTokenizer
        self.client = AsyncOpenAI(base_url=BASE_URL, api_key="EMPTY", timeout=7200)
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.think_end = self.tok.convert_tokens_to_ids("</think>")
        self.suffix_ids = self.tok.encode(FORCE_SUFFIX, add_special_tokens=False)
        self.direct_ids = self.tok.encode("The final answer is " + FORCE_CUE, add_special_tokens=False)
        self.free_suffix_ids = self.tok.encode(FREE_SUFFIX, add_special_tokens=False)
        self.free = False                                  # also record free answers (--free)
        self.stop_on_loop = False                          # cut a trace when its loop flag fires
        self.sem = asyncio.Semaphore(concurrency)          # long trace requests
        self.sem_short = asyncio.Semaphore(concurrency)    # short answer requests, own queue
        self.lock = asyncio.Lock()
        self.prov = provenance()
        self.score = defaultdict(lambda: [0, 0])           # (config, cut) -> [answers, correct]
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
            for (cfg, cut), (n, c) in self.score.items():          # best-budget accuracy per config
                if cfg not in top or (cut or 0) >= top[cfg][0]:
                    top[cfg] = (cut or 0, 100 * c / n)
            self.bar.set_postfix_str("  ".join(f"{k}: {v[1]:.0f}%" for k, v in top.items()))


async def thinking_sample(ctx, q, cfg_name, cfg, k, cuts, b_max, d):
    """One trace and its answers. The trace ends at its own '</think>' (natural), at the loop
    flag if --stop-on-loop (loop), or at b_max (cap). A forced answer is asked at every cut it
    reached (cut < end) and ONE at its end; nothing is generated or stored for the cuts after
    its end (the report fills them in from the end answer, see common.expand_answers)."""
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
    natural = r["finish"] == "stop" and r["stop_reason"] is not None   # closed '</think>' itself
    end_kind = "loop" if r["finish"] == "loop" else "natural" if natural else "cap"
    loop_at = r["loop_at"] if end_kind == "loop" else loop_point(ids)  # flag only, if not stopped
    trace = dict(question_id=qid, config=cfg_name, sample=k, seed=seed,
                 prompt_tokens=len(P), reasoning_tokens=n, natural_end=int(natural), end_kind=end_kind,
                 stop_on_loop=int(ctx.stop_on_loop), cuts=list(cuts),
                 eos_in_reasoning=int(r["finish"] == "stop" and r["stop_reason"] is None),
                 loop_detected=int(loop_at is not None), loop_at=loop_at,
                 reasoning_ids=ids, b_max=b_max, sampling=sampling, **ctx.prov)
    try:
        rows = []
        for c in cuts:                         # increasing: the shorter prefix stays in the cache
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
        async with ctx.sem_short:              # the answer at the end of the trace
            a = await complete(ctx.client, P + ids + ctx.suffix_ids, FORCED_ANSWER_MAX,
                               seed_for(qid, k, "end"), sampling, logprobs=1)
        f = answer_fields(ctx, a, q["reference"])
        end = dict(question_id=qid, config=cfg_name, sample=k, end_kind=end_kind, reasoning_tokens=n,
                   total_tokens=n + f["answer_tokens"],
                   reasoning_text=ctx.tok.decode(ids, skip_special_tokens=True), **f)
        free_rows = await free_answer(ctx, q, cfg_name, trace, P, ids) if ctx.free else []
    except Exception as e:
        print(f"\n[answer failed] {cfg_name} {qid} #{k}: {e}")
        return
    async with ctx.lock:                       # trace and its answers are written together
        append_jsonl(d / "traces.jsonl", [trace])
        append_jsonl(d / "answers.jsonl", rows)
        append_jsonl(d / "end_answers.jsonl", [end])
        append_jsonl(d / "free_answers.jsonl", free_rows)
        ctx.show(expand_answers(rows, [trace], [end]))


def box_end(ctx, ids):
    """Number of answer tokens up to and including the '}' that closes the forced \\boxed{
    (braces inside, e.g. \\frac{3}{4}, are counted). len(ids) if the box never closes."""
    for i in range(len(ids)):
        depth = 1
        for ch in ctx.tok.decode(ids[:i + 1], skip_special_tokens=True):
            depth += (ch == "{") - (ch == "}")
            if depth == 0:
                return i + 1
    return len(ids)


def answer_fields(ctx, a, reference):
    """The forced answer, graded. The model often goes on after the box ('}. Here is the
    step-by-step...'), so the confidence is measured on the box only (box_confidence, box_tokens).
    answer_confidence covers all 16 tokens."""
    text = FORCE_CUE + ctx.tok.decode(a["ids"], skip_special_tokens=True)
    lp = a["logprobs"] or []
    nb = box_end(ctx, a["ids"])
    return dict(answer_text=text, answer_tokens=len(a["ids"]),
                answer_confidence=math.exp(sum(lp) / len(lp)) if lp else None,
                box_tokens=nb,
                box_confidence=math.exp(sum(lp[:nb]) / nb) if lp and nb else None,
                answer_logprobs=[round(x, 4) for x in lp],
                **grade(text, reference))


async def cut_answers(ctx, q, cfg_name, trace, P, ids, cuts, own_end=None, top_cut=None):
    """Forced answers of one trace at each cut, in increasing order (the shorter prefix is still
    in the server cache for the next one). own_end: an answer row already recorded after the
    trace's own end, reused for every cut the trace finished before (same as in the run)."""
    qid, k, sampling = q["question_id"], trace["sample"], trace["sampling"]
    n, natural, loop_at = trace["reasoning_tokens"], trace["natural_end"], trace["loop_at"]
    top_cut = top_cut or cuts[-1]
    rows, own_end_answer = [], None
    for c in sorted(cuts):
        ended = bool(natural) and n <= c
        used = min(n, c)
        if ended and own_end is not None:
            fields = {f: own_end.get(f) for f in ("answer_text", "answer_tokens", "answer_confidence",
                                                 "box_tokens", "box_confidence", "answer_logprobs",
                                                 "pred", "valid", "correct")}
        else:
            async with ctx.sem_short:
                if ended:                     # the model had finished: one answer for all higher cuts
                    if own_end_answer is None:
                        own_end_answer = await complete(ctx.client, P + ids + ctx.suffix_ids,
                                                        FORCED_ANSWER_MAX, seed_for(qid, k, "end"),
                                                        sampling, logprobs=1)
                    a = own_end_answer
                else:
                    a = await complete(ctx.client, P + ids[:c] + ctx.suffix_ids, FORCED_ANSWER_MAX,
                                       seed_for(qid, k, c), sampling, logprobs=1)
            fields = answer_fields(ctx, a, q["reference"])
        rows.append(dict(
            question_id=qid, config=cfg_name, sample=k, cut=c,
            reasoning_tokens_used=used, finished_by_cut=int(ended),
            loop_by_cut=int(loop_at is not None and loop_at <= used),
            total_tokens=used + fields["answer_tokens"],
            reasoning_text=ctx.tok.decode(ids[:used], skip_special_tokens=True) if c == top_cut else None,
            **fields))
    return rows


async def loop_answer(ctx, q, cfg_name, trace, P, ids):
    """The forced answer at the loop point, if the loop flag fired before the trace ended:
    what a 'stop on loop' guard would have answered. [] otherwise."""
    L = trace["loop_at"]
    if L is None or L >= trace["reasoning_tokens"]:
        return []
    qid, k = q["question_id"], trace["sample"]
    async with ctx.sem_short:
        a = await complete(ctx.client, P + ids[:L] + ctx.suffix_ids, FORCED_ANSWER_MAX,
                           seed_for(qid, k, "loop"), trace["sampling"], logprobs=1)
    return [dict(question_id=qid, config=cfg_name, sample=k, loop_at=L,
                 natural_end=trace["natural_end"], total_tokens=L + len(a["ids"]),
                 **answer_fields(ctx, a, q["reference"]))]


async def free_answer(ctx, q, cfg_name, trace, P, ids):
    """The answer the model writes on its own after its thinking (no cue). Limited to FREE_MAX
    tokens, or less if the server context is shorter (answer_cap is recorded)."""
    qid, k, natural = q["question_id"], trace["sample"], trace["natural_end"]
    prefix = P + ids + ([ctx.think_end] if natural else ctx.free_suffix_ids)
    cap = min(FREE_MAX, ctx.prov["server_max_len"] - len(prefix))
    if cap < 1:
        return []
    async with ctx.sem_short:
        a = await complete(ctx.client, prefix, cap, seed_for(qid, k, "free"), trace["sampling"])
    text = ctx.tok.decode(a["ids"], skip_special_tokens=True)
    return [dict(question_id=qid, config=cfg_name, sample=k, natural_end=natural,
                 reasoning_tokens=trace["reasoning_tokens"], answer_cap=cap,
                 answer_finished=int(a["finish"] == "stop"), answer_text=text,
                 answer_tokens=len(a["ids"]), total_tokens=trace["reasoning_tokens"] + len(a["ids"]),
                 **grade(text, q["reference"]))]


async def cmd_answers(args):
    """Add cuts and loop answers to traces already recorded (no new traces)."""
    qs = {q["question_id"]: q for q in read_jsonl(DATA / f"{args.set}.jsonl")}
    ctx = Ctx(args.concurrency)
    if ctx.prov["server_max_len"] is None:
        raise SystemExit("The server does not answer at " + BASE_URL + ". Start it first (VLLM-DOC.md).")
    jobs = []
    for name in args.configs:
        if not CONFIGS[name]["thinking"]:
            print(f"{name}: no thinking, no cuts; skipped")
            continue
        d = out_dir(args.set, name)
        traces = read_jsonl(d / "traces.jsonl")
        answers = read_jsonl(d / "answers.jsonl")
        have = {(a["question_id"], a["sample"], a["cut"]) for a in answers}
        own_end = {}
        for a in answers:
            if a["finished_by_cut"]:
                own_end.setdefault((a["question_id"], a["sample"]), a)
        have_loop = {(a["question_id"], a["sample"]) for a in read_jsonl(d / "loop_answers.jsonl")}
        have_free = {(a["question_id"], a["sample"]) for a in read_jsonl(d / "free_answers.jsonl")}
        top = max((a["cut"] for a in answers), default=None)
        n_cuts = n_loop = n_free = 0
        for t in traces:
            key = (t["question_id"], t["sample"])
            new = [c for c in args.cuts if c <= t["b_max"] and key + (c,) not in have]
            loop = args.loop and key not in have_loop and t["loop_at"] is not None \
                and t["loop_at"] < t["reasoning_tokens"]
            free = args.free and key not in have_free
            if new or loop or free:
                jobs.append((name, d, t, new, loop, free, own_end.get(key), top))
                n_cuts += len(new)
                n_loop += loop
                n_free += free
        print(f"{name}: {len(traces)} traces -> {n_cuts} new cut answers, {n_loop} loop answers, "
              f"{n_free} free answers")

    async def one(name, d, t, new, loop, free, own, top):
        q = qs[t["question_id"]]
        P = ctx.prompt_ids(q["question"], thinking=True)
        if len(P) != t["prompt_tokens"]:
            raise SystemExit(f"prompt of {t['question_id']} changed since the run; stopping")
        ids = t["reasoning_ids"]
        try:
            rows = await cut_answers(ctx, q, name, t, P, ids, new, own_end=own, top_cut=top) if new else []
            loop_rows = await loop_answer(ctx, q, name, t, P, ids) if loop else []
            free_rows = await free_answer(ctx, q, name, t, P, ids) if free else []
        except Exception as e:
            print(f"\n[answer failed] {name} {t['question_id']} #{t['sample']}: {e}")
            return
        async with ctx.lock:
            append_jsonl(d / "answers.jsonl", rows)
            append_jsonl(d / "loop_answers.jsonl", loop_rows)
            append_jsonl(d / "free_answers.jsonl", free_rows)
        ctx.bar.update(1)

    from tqdm import tqdm
    ctx.bar = tqdm(total=len(jobs))
    t0 = time.time()
    await asyncio.gather(*(one(*j) for j in jobs))
    ctx.bar.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    report(args.set, args.configs)


async def nothink_sample(ctx, q, cfg_name, cfg, k, d):
    sampling = cfg["sampling"]
    direct = cfg.get("direct", False)
    P = ctx.prompt_ids(q["question"], thinking=False)
    if direct:                                 # the template already closed an empty think block
        P = P + ctx.direct_ids
    seed = seed_for(q["question_id"], k)
    async with ctx.sem_short if direct else ctx.sem:
        try:
            r = await complete(ctx.client, P, cfg["max_tokens"], seed, sampling, logprobs=1)
        except Exception as e:
            print(f"\n[answer failed] {cfg_name} {q['question_id']} #{k}: {e}")
            return
    if direct:                                 # same fields as a forced answer after a cut
        fields = answer_fields(ctx, r, q["reference"])
    else:
        text = ctx.tok.decode(r["ids"], skip_special_tokens=True)
        fields = dict(answer_text=text, answer_tokens=len(r["ids"]), answer_confidence=None,
                      **grade(text, q["reference"]))
    trace = dict(question_id=q["question_id"], config=cfg_name, sample=k, seed=seed,
                 prompt_tokens=len(P), reasoning_tokens=0, natural_end=int(r["finish"] == "stop"),
                 eos_in_reasoning=0, loop_detected=int(loop_point(r["ids"]) is not None), loop_at=None,
                 reasoning_ids=[], b_max=0, sampling=sampling, **ctx.prov)
    row = dict(question_id=q["question_id"], config=cfg_name, sample=k, cut=None,
               reasoning_tokens_used=0, finished_by_cut=int(r["finish"] == "stop"), loop_by_cut=0,
               total_tokens=len(r["ids"]), reasoning_text=None, **fields)
    async with ctx.lock:
        append_jsonl(d / "traces.jsonl", [trace])
        append_jsonl(d / "answers.jsonl", [row])
        ctx.show([row])


async def cmd_run(args):
    qs = read_jsonl(DATA / f"{args.set}.jsonl")
    if not qs:
        raise SystemExit(f"No questions in {DATA / (args.set + '.jsonl')}. Run src/prepare_data.py first.")
    qs = qs[:args.limit] if args.limit else qs
    cuts = tuple(sorted(set(args.cuts) | {args.b_max}))
    cuts = tuple(c for c in cuts if c <= args.b_max)
    ctx = Ctx(args.concurrency)
    ctx.free = args.free
    ctx.stop_on_loop = args.stop_on_loop
    srv = ctx.prov["server_max_len"]
    need = max(len(ctx.prompt_ids(q["question"], True)) for q in qs) + args.b_max + len(ctx.suffix_ids) + FORCED_ANSWER_MAX
    if srv is None:
        raise SystemExit("The server does not answer at " + BASE_URL + ". Start it first (VLLM-DOC.md).")
    if srv < need:
        raise SystemExit(f"Server context {srv} is too small for --b-max {args.b_max} (needs {need}). See VLLM-DOC.md.")

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
                             else nothink_sample(ctx, q, name, cfg, k, d))
    print(f"set={args.set}  questions={len(qs)}  samples={args.samples}  configs={args.configs}")
    print(f"b_max={args.b_max}  cuts={cuts}  -> {len(tasks)} samples to generate")

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

def guard_replay(full, traces, loop_rows):
    """Exact replay of 'stop when the loop flag fires', from the recorded loop answers.
    A trace whose flag fired before it ended is answered at the loop point and costs only the
    tokens up to it; every other trace keeps its full-budget answer. Empty if the loop answers
    are missing (run 'answers' first)."""
    looping = [t for t in traces if t["loop_at"] is not None and t["loop_at"] < t["reasoning_tokens"]]
    la = {(a["question_id"], a["sample"]): a for a in loop_rows}
    if not traces or any((t["question_id"], t["sample"]) not in la for t in looping):
        return {}
    acc, tok, broken, fixed, nat_broken = [], [], 0, 0, 0
    nat = {(t["question_id"], t["sample"]): t["natural_end"] for t in traces}
    for r in full:
        key = (r["question_id"], r["sample"])
        g = la.get(key, r)
        acc.append(g["correct"])
        tok.append(g["total_tokens"])
        broken += r["correct"] and not g["correct"]
        fixed += g["correct"] and not r["correct"]
        nat_broken += nat.get(key, 0) and r["correct"] and not g["correct"]
    raw_tok = statistics.mean(r["total_tokens"] for r in full)
    return dict(guard_acc=round(100 * statistics.mean(acc), 1),
                guard_tokens=round(statistics.mean(tok)),
                tokens_saved_pct=round(100 * (1 - statistics.mean(tok) / raw_tok), 1),
                stopped_pct=round(100 * len(looping) / len(traces), 1),
                guard_broken=broken, guard_fixed=fixed, false_stop_broken=nat_broken)


def free_stats(full, free_rows):
    """Free answer after thinking vs the forced one at the full budget, on the same traces."""
    if not free_rows:
        return {}
    forced = {(r["question_id"], r["sample"]): r for r in full}
    pairs = [(f, forced[(f["question_id"], f["sample"])]) for f in free_rows
             if (f["question_id"], f["sample"]) in forced]
    return dict(free_n=len(free_rows),
                free_acc=round(100 * statistics.mean(f["correct"] for f in free_rows), 1),
                free_valid_pct=round(100 * statistics.mean(f["valid"] for f in free_rows), 1),
                free_answer_tokens=round(statistics.mean(f["answer_tokens"] for f in free_rows)),
                free_same_as_forced_pct=round(100 * statistics.mean(
                    f["valid"] and g["valid"] and abs(f["pred"] - g["pred"]) < 1e-6 for f, g in pairs), 1)
                if pairs else None)


def report(set_name, configs):
    """Accuracy per config and cut, plus behaviour at the full budget. Also saved as CSV."""
    import csv
    lines = []
    for name in configs:
        d = out_dir(set_name, name)
        ans, traces = load_answers(d), read_jsonl(d / "traces.jsonl")
        if not ans:
            continue
        by_cut = defaultdict(list)
        for r in ans:
            by_cut[r["cut"]].append(r)
        top = max(by_cut, key=lambda c: c or 0)
        full = by_cut[top]
        # majority vote and pass@k over the samples of each question, at the full budget
        per_q = defaultdict(list)
        for r in full:
            per_q[r["question_id"]].append(r)
        maj = passk = 0
        for rs in per_q.values():
            votes = Counter(round(r["pred"], 6) for r in rs if r["valid"])
            if votes:
                best = votes.most_common(1)[0][0]
                maj += any(r["correct"] for r in rs if r["valid"] and round(r["pred"], 6) == best)
            passk += any(r["correct"] for r in rs)
        lens = [t["reasoning_tokens"] for t in traces]
        guard = guard_replay(full, traces, read_jsonl(d / "loop_answers.jsonl"))
        guard.update(free_stats(full, read_jsonl(d / "free_answers.jsonl")))
        line = dict(config=name, questions=len(per_q), samples=len(traces),
                    **{f"acc@{c or 'nothink'}": round(100 * statistics.mean(r["correct"] for r in by_cut[c]), 1)
                       for c in sorted(by_cut, key=lambda c: c or 0)},
                    valid_pct=round(100 * statistics.mean(r["valid"] for r in full), 1),
                    maj_vote_pct=round(100 * maj / len(per_q), 1),
                    pass_at_k_pct=round(100 * passk / len(per_q), 1),
                    finished_pct=round(100 * statistics.mean(t["natural_end"] for t in traces), 1),
                    loop_stopped_pct=round(100 * statistics.mean(t.get("end_kind") == "loop" for t in traces), 1),
                    loop_pct=round(100 * statistics.mean(t["loop_detected"] for t in traces), 1),
                    mean_tokens=round(statistics.mean(r["total_tokens"] for r in full)),
                    median_reasoning=round(statistics.median(lens)) if lens else 0, **guard)
        lines.append(line)
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
    print(f"\nacc@N = accuracy (%) when thinking is limited to N tokens; acc@nothink = configs without thinking (nothink, direct).")
    print("maj_vote / pass_at_k use all samples of a question at the full budget.")
    print("guard_* = exact replay of 'stop on loop' (loop_answers.jsonl): accuracy and tokens with the guard,")
    print("  stopped_pct = traces stopped, guard_broken / guard_fixed = traces right->wrong / wrong->right,")
    print("  false_stop_broken = traces that would have ENDED BY THEMSELVES, were right, and the guard broke.")
    print("free_* = the model's own answer after thinking (no cue, --free): accuracy, valid, answer length,")
    print("  and how often it gives the same number as the forced answer at the full budget.")
    print(f"Saved: {out}")


# --------------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "answers", "report"):
        s = sub.add_parser(name)
        s.add_argument("--set", default="config_exp", choices=["config_exp", "train", "val", "test"])
        s.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
        if name == "run":
            s.add_argument("--limit", type=int, help="only the first N questions (trial run)")
            s.add_argument("--samples", type=int, default=SAMPLES, help=f"traces per question (default {SAMPLES})")
            s.add_argument("--b-max", type=int, default=B_MAX, help=f"longest trace (default {B_MAX})")
            s.add_argument("--cuts", type=int, nargs="+", default=list(CUTS))
            s.add_argument("--free", action="store_true",
                           help="also let the model answer on its own after thinking (free_answers.jsonl)")
            s.add_argument("--stop-on-loop", action="store_true",
                           help="stop a trace when its loop flag fires and answer there (end_kind 'loop')")
            s.add_argument("--concurrency", type=int, default=16, help="requests at the same time")
        if name == "answers":
            s.add_argument("--cuts", type=int, nargs="+", default=[],
                           help="cuts to add; cuts already recorded are skipped")
            s.add_argument("--free", action="store_true",
                           help="add the free answer after thinking (free_answers.jsonl)")
            s.add_argument("--no-loop", dest="loop", action="store_false",
                           help="do not add the answer at the loop point")
            s.add_argument("--concurrency", type=int, default=16, help="requests at the same time")
    args = p.parse_args()
    if args.cmd == "run":
        asyncio.run(cmd_run(args))
    elif args.cmd == "answers":
        asyncio.run(cmd_answers(args))
    else:
        report(args.set, args.configs)


if __name__ == "__main__":
    main()
