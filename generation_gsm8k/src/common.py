"""Shared pieces of the GSM8K benchmark: paths, file helpers, grading, the server client.

Nothing here talks to the GPU by itself; run_sweep.py and plot_sweep.py import it.
"""
import hashlib
import json
import math
import re
from pathlib import Path

MODEL = "Qwen/Qwen3.5-2B"
BASE_URL = "http://127.0.0.1:8000/v1"

ROOT = Path(__file__).resolve().parents[1]          # .../generation_gsm8k
DATA = ROOT / "data"                                # questions and results (ignored by git)
# figures are written to <repo>/figures/ by src/plot/plot_sweep.py

# The prompt is the one Qwen recommends for math benchmarks; it is the same for every config,
# so that only the sampling settings change between configs.
PROMPT = "{q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

# When reasoning is cut (or has ended), this text closes it and opens the answer box, so the
# model can only write the number.
FORCE_CUE = "\\boxed{"
FORCE_SUFFIX = "\n</think>\n\nThe final answer is " + FORCE_CUE
FORCED_ANSWER_MAX = 16                              # tokens for the content of the box

# Loop flag: in a 2,048-token window, checked every 256 tokens,
# more than 10% of the 20-token pieces already appeared earlier in the window. Flag only.
LOOP_WINDOW, LOOP_STEP, LOOP_NGRAM, LOOP_DUP_SHARE = 2048, 256, 20, 0.10


# ------------------------------------------------------------------------------ files

def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def seed_for(*parts):
    """A fixed seed from the question id and sample number. The SAME seed is used for every
    config, so differences between configs come from the settings, not from luck."""
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def loop_point(ids):
    """Token count at which word-for-word repetition was first found, or None."""
    for n in range(LOOP_WINDOW, len(ids) + 1, LOOP_STEP):
        w = ids[n - LOOP_WINDOW:n]
        pieces = [tuple(w[i:i + LOOP_NGRAM]) for i in range(len(w) - LOOP_NGRAM + 1)]
        if 1 - len(set(pieces)) / len(pieces) > LOOP_DUP_SHARE:
            return n
    return None


# ---------------------------------------------------------------------------- grading

def reference_number(answer_field):
    """GSM8K stores the final number after '####'."""
    return to_number(answer_field.split("####")[-1])


def to_number(text):
    """The number in a piece of text; None if there is none or there are two different ones.

    Handles '1,234', '$18', '18.00', '30\\%', '155 \\text{ minutes}', '\\frac{3}{4}', '7/2',
    '16 \\frac{2}{3}' and '3 \\times 4 = 12' (the part after the last '=' is used).
    """
    if text is None:
        return None
    t = text.split("=")[-1].replace("−", "-").replace("\\,", "")  # Unicode minus, 10\,000
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)                    # 1,234 -> 1234
    t = t.replace("$", "").replace("\\!", "").replace("\\%", "").replace("%", "")
    num = r"-?\d+\.?\d*"
    m = re.fullmatch(rf"\s*(-?\d+)\s*\\d?frac\{{(\d+)\}}\{{(\d+)\}}\s*", t)   # 16 \frac{2}{3}
    if m and float(m.group(3)) != 0:
        whole = float(m.group(1))
        return whole + math.copysign(float(m.group(2)) / float(m.group(3)), whole if whole else 1)
    m = re.search(rf"\\d?frac\{{({num})\}}\{{({num})\}}", t) or re.search(rf"({num})\s*/\s*({num})", t)
    if m and float(m.group(2)) != 0:
        rest = t[:m.start()] + t[m.end():]
        return None if re.search(r"\d", rest) else float(m.group(1)) / float(m.group(2))
    values = {float(x) for x in re.findall(num, t)}
    return values.pop() if len(values) == 1 else None


def last_boxed(text):
    """Content of the last complete \\boxed{...} (nested braces allowed); None if there is none."""
    start = text.rfind("\\boxed{")
    while start != -1:
        depth, i = 1, start + len("\\boxed{")
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth == 0:
            return text[start + len("\\boxed{"):i - 1]
        start = text.rfind("\\boxed{", 0, start)     # cut off: try the box before
    return None


def first_boxed(text):
    """Content of the first \\boxed{...} if it closes (nested braces allowed); None otherwise."""
    start = text.find("\\boxed{")
    if start == -1:
        return None
    depth, i = 1, start + len("\\boxed{")
    while i < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[start + len("\\boxed{"):i - 1] if depth == 0 else None


def answer_box(text):
    """The model's answer. A forced answer starts at the cue, so its answer is that first box:
    anything written after it is not the answer, and box_confidence measures this box too.
    Any other text uses its last complete box."""
    return first_boxed(text) if text.startswith(FORCE_CUE) else last_boxed(text)


def grade(answer_text, reference):
    """valid = a number could be read from the answer box; correct = it equals the reference."""
    value = to_number(answer_box(answer_text))
    valid = value is not None
    correct = bool(valid and reference is not None and abs(value - reference) < 1e-6)
    return dict(pred=value, valid=int(valid), correct=int(correct))


# ----------------------------------------------------------------------------- server

def provenance():
    """Exact weights version and vLLM version, stored with every result."""
    ref = Path.home() / ".cache/huggingface/hub" / ("models--" + MODEL.replace("/", "--")) / "refs/main"
    out = dict(model=MODEL, weights_hash=ref.read_text().strip() if ref.exists() else None,
               vllm_version=None, server_max_len=None)
    try:
        import urllib.request
        base = BASE_URL.rsplit("/v1", 1)[0]
        with urllib.request.urlopen(base + "/version", timeout=5) as r:
            out["vllm_version"] = json.loads(r.read())["version"]
        with urllib.request.urlopen(BASE_URL + "/models", timeout=5) as r:
            out["server_max_len"] = json.loads(r.read())["data"][0]["max_model_len"]
    except Exception:
        pass
    return out


def _extra(obj, name):
    v = getattr(obj, name, None)
    if v is None and getattr(obj, "model_extra", None):
        v = obj.model_extra.get(name)
    return v


async def complete(client, prompt_ids, max_tokens, seed, sampling, stop_ids=None, logprobs=None):
    """One request to vLLM's raw completions endpoint (the prompt is a list of token ids)."""
    s = dict(sampling)
    r = await client.completions.create(
        model=MODEL, prompt=prompt_ids, max_tokens=max_tokens, seed=seed, logprobs=logprobs,
        temperature=s["temperature"], top_p=s["top_p"], presence_penalty=s["presence_penalty"],
        extra_body=dict(
            top_k=s["top_k"], min_p=s.get("min_p", 0.0),
            stop_token_ids=stop_ids or [],
            skip_special_tokens=False, add_special_tokens=False,
            return_token_ids=True, return_tokens_as_token_ids=True,
        ),
    )
    c = r.choices[0]
    ids = _extra(c, "token_ids")
    if not ids and c.logprobs is not None:
        ids = [int(t.split(":")[1]) for t in c.logprobs.tokens]
    lps = [x if x is not None else 0.0 for x in c.logprobs.token_logprobs] if c.logprobs else []
    return dict(ids=list(ids or []), logprobs=lps, finish=c.finish_reason,
                stop_reason=_extra(c, "stop_reason"))


def loop_at_end(ids):
    """Loop check on the last full window only (same rule as loop_point, for streaming)."""
    n = len(ids)
    if n < LOOP_WINDOW or n % LOOP_STEP:
        return False
    w = ids[n - LOOP_WINDOW:n]
    pieces = [tuple(w[i:i + LOOP_NGRAM]) for i in range(len(w) - LOOP_NGRAM + 1)]
    return 1 - len(set(pieces)) / len(pieces) > LOOP_DUP_SHARE


async def complete_trace(client, prompt_ids, max_tokens, seed, sampling, stop_ids, stop_on_loop):
    """A reasoning trace, streamed. With stop_on_loop the request is cancelled as soon as the
    loop flag fires (checked every LOOP_STEP tokens) and the trace is cut at that point.
    Returns dict(ids, finish, stop_reason, loop_at) ; finish = 'loop' if stopped on a loop."""
    s = dict(sampling)
    stream = await client.completions.create(
        model=MODEL, prompt=prompt_ids, max_tokens=max_tokens, seed=seed, stream=True,
        temperature=s["temperature"], top_p=s["top_p"], presence_penalty=s["presence_penalty"],
        extra_body=dict(
            top_k=s["top_k"], min_p=s.get("min_p", 0.0), stop_token_ids=stop_ids or [],
            skip_special_tokens=False, add_special_tokens=False, return_token_ids=True,
        ),
    )
    ids, finish, stop_reason, next_check = [], None, None, LOOP_WINDOW
    try:
        async for chunk in stream:
            c = chunk.choices[0]
            ids.extend(_extra(c, "token_ids") or [])
            finish = c.finish_reason or finish
            stop_reason = _extra(c, "stop_reason") if c.finish_reason else stop_reason
            while stop_on_loop and len(ids) >= next_check:
                if loop_at_end(ids[:next_check]):
                    return dict(ids=ids[:next_check], finish="loop", stop_reason=None, loop_at=next_check)
                next_check += LOOP_STEP
    finally:
        await stream.close()                     # cancels the request on the server if still running
    return dict(ids=ids, finish=finish, stop_reason=stop_reason, loop_at=None)


def expand_answers(answers, traces, end_rows):
    """Answers of every (trace, cut), as the report and plots need them.

    Since 2026-11-20 a trace stores a forced answer only at the cuts it reached (cut < its end)
    plus ONE answer at its end (end_answers.jsonl): the natural end, the loop stop, or the cap.
    With a budget at or above its end, the model would have stopped there and given that answer,
    so those cuts are filled in here from the end answer (marked from_end=1). Nothing is
    generated for them. Older data (answers at every cut, no end_answers.jsonl) is returned as is."""
    tr = {(t["question_id"], t["sample"]): t for t in traces}
    out = list(answers)
    for e in end_rows:
        t = tr.get((e["question_id"], e["sample"]))
        if t is None:
            continue
        cuts = t["cuts"]
        n = e["reasoning_tokens"]
        for c in cuts:
            if c < n:
                continue
            row = {k: v for k, v in e.items() if k not in ("end_kind", "reasoning_tokens", "reasoning_text")}
            row.update(cut=c, reasoning_tokens_used=n, finished_by_cut=int(e["end_kind"] == "natural"),
                       loop_by_cut=int(e["end_kind"] == "loop"), stopped_by_loop=int(e["end_kind"] == "loop"),
                       reasoning_text=e.get("reasoning_text") if c == cuts[-1] else None, from_end=1)
            out.append(row)
    return out


def load_answers(d):
    """answers.jsonl of a result folder, with the cuts after each trace's end filled in."""
    d = Path(d)
    return expand_answers(read_jsonl(d / "answers.jsonl"), read_jsonl(d / "traces.jsonl"),
                          read_jsonl(d / "end_answers.jsonl"))
