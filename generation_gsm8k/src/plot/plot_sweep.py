#!/usr/bin/env python
"""Draw every figure of one recorded set, GSM8K, MATH or AMC (same result files).
Reads only, generates nothing. Input: generation_<dataset>/data/results/<set>/.
Output (+ analysis.csv), in figures/ at the repo root, one folder per kind of generation:
  figures/gsm8k_config_exp/                 the GSM8K config sweep (--dataset gsm8k --set config_exp)
  figures/math_pilot/                       the 20-question MATH pilot (--dataset math --set math_pilot)
  figures/train_val_test/                   the controller data of ALL datasets in the same figures:
                                            GSM8K, MATH and AMC, each train + val + test merged
                                            (--dataset both --set all), to choose the effort levels
  figures/train_val_test/<dataset>/        optional: one dataset merged (--dataset gsm8k --set all)
  figures/amc23/                            AMC 2023 (test only; --dataset amc23 --set amc23)

  01_accuracy_vs_budget     accuracy at each thinking budget (cut), one line per config
  02_accuracy_vs_tokens     the same against the tokens actually spent (the cost)
  03_still_thinking         share of traces still reasoning after N tokens
  04_length_histograms      reasoning length per config: ended by itself vs hit the cap
  05_accuracy_by_ending     full-budget accuracy of traces that ended / hit the cap with or without a loop
  06_loop_onset             share of traces whose word-for-word loop has started by N tokens
  07_answer_flips           per budget step: answers fixed (wrong->right) and broken (right->wrong)
  08_budget_needed          per question, the smallest budget at which it is solved
  09_calibration            answer confidence against accuracy (pooled over all cuts)
  10_confidence_auroc       how well answer confidence separates right from wrong, per budget
  11_agreement              majority vote: size of the largest agreeing group vs its accuracy
  12_full_budget            accuracy, majority vote, pass@k, valid answers, per config
  13_loop_guard_estimate    stopping on loops (exact with loop_answers.jsonl, else rough): accuracy and tokens with/without
  14_reasoning_levels       no reasoning (direct) / reasoning in the answer (nothink) / thinking
  15_effort_levels          merged sets only: accuracy per budget (merged and per split) with the
                            controller levels L / M / H chosen by the rule

Run:  ~/.venvs/vllm/bin/python generation_gsm8k/src/plot/plot_sweep.py --dataset gsm8k --set config_exp
      ~/.venvs/vllm/bin/python generation_gsm8k/src/plot/plot_sweep.py --dataset both --set all
      --all draws config_exp, math_pilot, amc23 (if recorded) and the all-datasets figures; --out FOLDER writes one set somewhere else
"""
import argparse
import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sweep"))
from common import DATA, ROOT, load_answers, read_jsonl  # noqa: E402
from configs import CONFIGS  # noqa: E402

# Colour follows the config (order of configs.py), never its rank. Validated reference palette.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLOR = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(CONFIGS)}
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
INK, MUTED, GRID, NEUTRAL = "#1f1f1e", "#6b6a63", "#e6e5e0", "#b9b8b0"
plt = None

REPO = ROOT.parent
DATA_DIRS = {"gsm8k": DATA, "math": REPO / "generation_math" / "data", "amc23": REPO / "generation_amc23" / "data"}
FIGURES = REPO / "figures"
MERGED = {"gsm8k": ["train", "val", "test"], "math": ["math_train", "math_val", "math_test"]}


# GSM8K, MATH and AMC in the same figures: colour follows (dataset, config), fixed order
BOTH_COLOR = {("gsm8k", "t0.6_pp1.5"): PALETTE[0], ("math", "t0.6_pp1.5"): PALETTE[1],
              ("amc23", "t0.6_pp1.5"): PALETTE[2], ("gsm8k", "direct"): PALETTE[6],
              ("math", "direct"): PALETTE[3], ("amc23", "direct"): PALETTE[7]}


def figure_dir(dataset, set_name):
    """figures/<kind of generation>/...: the sweep, the pilot, or the train/val/test splits."""
    if dataset == "both":
        return FIGURES / "train_val_test"
    if set_name == "all":
        return FIGURES / "train_val_test" / dataset
    if set_name == "config_exp":
        return FIGURES / f"{dataset}_config_exp"
    if set_name.endswith("pilot"):
        return FIGURES / set_name
    if dataset == "amc23":
        return FIGURES / "amc23"
    name = set_name if set_name.startswith("math_") else f"{dataset}_{set_name}"
    return FIGURES / "train_val_test" / name


def vote_key(pred):
    """Answers that count as the same vote: numbers rounded (GSM8K), strings as they are (MATH)."""
    return round(pred, 6) if isinstance(pred, (int, float)) else pred


# ------------------------------------------------------------------------------ helpers

def style(ax, grid="both"):
    ax.grid(True, axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def log2_axis(ax, ticks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t // 1024}K" if t >= 1024 else str(t) for t in ticks])
    ax.minorticks_off()


def title(ax, text, sub=None):
    ax.set_title(text, color=INK, loc="left", fontsize=11, pad=6 + 11 * (sub.count("\n") + 1) if sub else 6)
    if sub:
        ax.text(0, 1.01, sub, transform=ax.transAxes, color=MUTED, fontsize=8, va="bottom")


def save(fig, out, name, top=1.0, bottom=0.0):
    """top / bottom < 1 / > 0 keep room for a figure title or legend outside the axes."""
    fig.tight_layout(rect=(0, bottom, 1, top))
    fig.savefig(out / f"{name}.png", dpi=150)
    plt.close(fig)


def ktick(x, _pos=None):
    """Token counts as 512, 1K, 16K."""
    return f"{x / 1024:g}K" if x >= 1024 else f"{x:g}"


def ramp(color, n):
    """n shades from light to the given colour (one per budget)."""
    from matplotlib.colors import LinearSegmentedColormap, to_hex
    cmap = LinearSegmentedColormap.from_list("r", ["#f1efe8", color])
    return [to_hex(cmap(0.3 + 0.7 * i / max(n - 1, 1))) for i in range(n)]


def grid_axes(n, w=3.6, h=3.0, cols=None):
    cols = cols or (2 if n == 4 else min(3, n))
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(w * cols, h * rows), squeeze=False)
    flat = list(axes.flat)
    for ax in flat[n:]:
        ax.axis("off")
    return fig, flat[:n]


def blues(n):
    """n steps of the blue ramp, light to dark (interpolated when there are more than 7 cuts)."""
    from matplotlib.colors import LinearSegmentedColormap, to_hex
    cmap = LinearSegmentedColormap.from_list("blues", BLUES[1:])
    return [to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]


def legend(ax, **kw):
    ax.legend(frameon=False, fontsize=8, **kw)


def conf(a):
    """Answer confidence: on the box only when recorded (box_confidence), else over all 16
    answer tokens (older data)."""
    c = a.get("box_confidence")
    return c if c is not None else a.get("answer_confidence")


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def auroc(scores, labels):
    """P(score of a random right answer > score of a random wrong one); ties count 1/2."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    ranked = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg = (i + j + 1) / 2                         # average 1-based rank of the tie group
        rank_sum += avg * sum(1 for k in range(i, j) if ranked[k][1])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


# --------------------------------------------------------------------------------- data

class Run:
    """All results of one config: traces, answers per (question, sample), cuts."""

    def __init__(self, name, answers, traces, loop_answers=()):
        self.name, self.config, self.color = name, name, COLOR[name]
        self.thinking = CONFIGS[name]["thinking"]
        self.traces = traces
        self.answers = answers
        self.cuts = sorted({a["cut"] for a in answers if a["cut"] is not None})
        self.top = self.cuts[-1] if self.cuts else None
        self.by_cut = defaultdict(list)
        for a in answers:
            self.by_cut[a["cut"]].append(a)
        self.full = self.by_cut[self.top]                   # full-budget answers (or nothink answers)
        self.ans = {(a["question_id"], a["sample"], a["cut"]): a for a in answers}
        self.tr = {(t["question_id"], t["sample"]): t for t in traces}
        self.b_max = max((t["b_max"] for t in traces), default=0)
        self.loop_ans = {(a["question_id"], a["sample"]): a for a in loop_answers}

    def acc(self, cut):
        return 100 * mean(a["correct"] for a in self.by_cut[cut])


def load_both(set_name="all"):
    """GSM8K, MATH and AMC runs together; each run is named '<dataset> <config>' and coloured by both."""
    runs = []
    for dataset in ("gsm8k", "math", "amc23"):
        for r in load(dataset, set_name):
            r.name = f"{dataset} {r.config}"
            r.color = BOTH_COLOR.get((dataset, r.config), r.color)
            r.dataset = dataset
            runs.append(r)
    return runs


def load(dataset, set_name):
    """The runs of one set, or with set_name 'all' the train, val and test sets merged (their
    questions are disjoint); run.split_of then gives each question's split."""
    sets = MERGED.get(dataset, [dataset]) if set_name == "all" else [set_name]
    # AMC is one recording; its split is given by the question files
    split_files = [f"amc23_{s}" for s in ("train", "val", "test")] if dataset == "amc23" and set_name == "all" else sets
    split_of = {}
    for s in split_files:
        split_of.update({q["question_id"]: s.split("_")[-1] for q in read_jsonl(DATA_DIRS[dataset] / f"{s}.jsonl")})
    runs = []
    for name in CONFIGS:
        answers, traces, loops = [], [], []
        for s in sets:
            d = DATA_DIRS[dataset] / "results" / s / name
            answers += load_answers(d)                    # cuts after a trace's end filled in
            for t in read_jsonl(d / "traces.jsonl"):
                t.pop("reasoning_ids", None)              # not needed here, saves memory
                traces.append(t)
            loops += read_jsonl(d / "loop_answers.jsonl")
        if answers:
            runs.append(Run(name, answers, traces, loops))
            runs[-1].split_of = split_of if set_name == "all" else None
    return runs


def effort_levels(acc):
    """L / M / H by the rule (same as generation_math/src/run_math.py): acc = {cut: accuracy}.
    H = top cut; L / M = first cut at 85% / 97% of acc@H."""
    cuts = sorted(acc)
    H = cuts[-1]
    first = lambda share: next(c for c in cuts if acc[c] >= share * acc[H])  # noqa: E731
    L, M = first(0.85), first(0.97)
    return dict(L=L, M=M, H=H)


def split_acc(r, split):
    return {c: 100 * mean(a["correct"] for a in r.by_cut[c] if r.split_of[a["question_id"]] == split)
            for c in r.cuts}


# ------------------------------------------------------------------------------ figures

def f01_accuracy_vs_budget(runs, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    cuts = sorted({c for r in runs for c in r.cuts})
    for r in runs:
        if r.thinking:
            ax.plot(r.cuts, [r.acc(c) for c in r.cuts], "-o", color=r.color, lw=2, ms=7,
                    mec="white", mew=2, label=r.name)
        else:
            ax.axhline(r.acc(None), color=r.color, lw=2, ls="--", label=f"{r.name} (no budget)")
    log2_axis(ax, cuts)
    ax.set_xlabel("thinking budget (tokens before the answer is forced)", color=INK)
    ax.set_ylabel("accuracy (%)", color=INK)
    title(ax, "Accuracy at each thinking budget")
    style(ax)
    legend(ax, loc="center right")
    save(fig, out, "01_accuracy_vs_budget")


def f02_accuracy_vs_tokens(runs, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in runs:
        cuts = r.cuts if r.thinking else [None]
        x = [mean(a["total_tokens"] for a in r.by_cut[c]) for c in cuts]
        y = [r.acc(c) for c in cuts]
        ax.plot(x, y, "-o" if r.thinking else "D", color=r.color, lw=2, ms=7 if r.thinking else 9,
                mec="white", mew=2, label=r.name)
    lo = min(min(l.get_xdata()) for l in ax.get_lines())
    log2_axis(ax, [t for t in (16, 64, 256, 1024, 4096, 16384) if t >= lo / 2])
    ax.set_xlabel("mean tokens spent per answer (reasoning + answer), log scale", color=INK)
    ax.set_ylabel("accuracy (%)", color=INK)
    title(ax, "Accuracy against cost", "a trace that ends early spends less than its budget; up and left is better")
    style(ax)
    legend(ax, loc="center right")
    save(fig, out, "02_accuracy_vs_tokens")


def f03_still_thinking(runs, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    think = [r for r in runs if r.thinking]
    b_max = max(r.b_max for r in think)
    xs = [2 ** i for i in range(8, b_max.bit_length()) if 2 ** i <= b_max]
    for r in think:
        ends = [t["reasoning_tokens"] if t["natural_end"] else float("inf") for t in r.traces]
        ys = [100 * sum(e > x for e in ends) / len(ends) for x in xs]
        ax.plot(xs, ys, "-o", color=r.color, lw=2, ms=6, mec="white", mew=1.5, label=r.name)
    log2_axis(ax, xs)
    ax.set_ylim(0, 100)
    ax.set_xlabel("reasoning tokens", color=INK)
    ax.set_ylabel("traces still reasoning (%)", color=INK)
    title(ax, "How long the model thinks", f"last point = traces that hit the {b_max:,}-token cap")
    style(ax)
    legend(ax)
    save(fig, out, "03_still_thinking")


def f04_length_histograms(runs, out):
    think = [r for r in runs if r.thinking]
    fig, axes = grid_axes(len(think), 5.2, 3.4)
    b_max = max(r.b_max for r in think)
    bins = list(range(0, b_max + 1024, 1024))
    for ax, r in zip(axes, think):
        ended = [t["reasoning_tokens"] for t in r.traces if t["natural_end"]]
        capped = [t["reasoning_tokens"] for t in r.traces if not t["natural_end"]]
        ax.hist([ended, capped], bins=bins, stacked=True, color=[r.color, NEUTRAL],
                edgecolor="white", linewidth=1)
        ax.set_title(f"{r.name}\nmedian {statistics.median(t['reasoning_tokens'] for t in r.traces):,.0f} tokens;"
                     f" did not end by itself: {len(capped):,} of {len(r.traces):,}",
                     color=INK, fontsize=9, loc="left")
        ax.xaxis.set_major_formatter(ktick)
        ax.set_xticks([0, 2048, 4096, 8192, 12288, 16384])
        ax.set_xlabel("reasoning tokens", color=MUTED, fontsize=8)
        ax.set_ylabel("traces", color=MUTED, fontsize=8)
        style(ax, "y")
    from matplotlib.patches import Patch
    fig.suptitle("Reasoning length per trace (bins of 1,024 tokens)", color=INK, x=0.01, y=0.98, ha="left")
    fig.legend(handles=[Patch(color=MUTED, label="ended by itself (in the run's colour)"),
                        Patch(color=NEUTRAL, label="did not end by itself (cap, or stopped at a loop)")],
               frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.005, 0.93), ncol=2)
    save(fig, out, "04_length_histograms", top=0.95)


def ending(t):
    if t["natural_end"]:
        return "ended by itself"
    return "loop (cap or stopped)" if t["loop_detected"] else "hit cap, no loop"


def f05_accuracy_by_ending(runs, out):
    think = [r for r in runs if r.thinking]
    groups = ["ended by itself", "hit cap, no loop", "loop (cap or stopped)"]
    shades = [BLUES[4], BLUES[2], NEUTRAL]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    w = 0.26
    for gi, g in enumerate(groups):
        for ri, r in enumerate(think):
            rows = [a for a in r.full if ending(r.tr[(a["question_id"], a["sample"])]) == g]
            if not rows:
                continue
            x = ri + (gi - 1) * w
            v = 100 * mean(a["correct"] for a in rows)
            ax.bar(x, v, w * 0.92, color=shades[gi], edgecolor="white", linewidth=2,
                   alpha=1.0 if len(rows) >= 20 else 0.35, label=g if ri == 0 else None)
            ax.text(x, v + 1, f"{v:.0f}\nn={len(rows)}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(think)))
    ax.set_xticklabels([r.name for r in think], fontsize=8, color=INK)
    ax.set_ylim(0, 118)
    ax.set_ylabel("accuracy at full budget (%)", color=INK)
    title(ax, "Does the way a trace ends predict a right answer?", "pale = fewer than 20 traces")
    style(ax, "y")
    legend(ax, loc="upper right", ncol=3)
    save(fig, out, "05_accuracy_by_ending")


def f06_loop_onset(runs, out):
    think = [r for r in runs if r.thinking]
    fig, ax = plt.subplots(figsize=(8, 5))
    b_max = max(r.b_max for r in think)
    xs = [x for x in range(2048, b_max + 1, 512)]
    for r in think:
        at = [t["loop_at"] for t in r.traces if t["loop_at"] is not None]
        ys = [100 * sum(a <= x for a in at) / len(r.traces) for x in xs]
        ax.plot(xs, ys, "-", color=r.color, lw=2, label=f"{r.name} ({100 * len(at) / len(r.traces):.0f}% in total)")
    log2_axis(ax, [2048, 4096, 8192, 16384])
    ax.set_xlabel("reasoning tokens", color=INK)
    ax.set_ylabel("traces with a loop started (%)", color=INK)
    title(ax, "When word-for-word loops start", "the detector can fire from 2,048 tokens on")
    style(ax)
    legend(ax)
    save(fig, out, "06_loop_onset")


def flips(r):
    """Per step between consecutive cuts: % of traces fixed (0->1) and broken (1->0)."""
    keys = list(r.tr)
    fixed, broken = [], []
    for c0, c1 in zip(r.cuts, r.cuts[1:]):
        f = b = 0
        for q, s in keys:
            a0, a1 = r.ans.get((q, s, c0)), r.ans.get((q, s, c1))
            if a0 is None or a1 is None:
                continue
            f += (not a0["correct"]) and a1["correct"]
            b += a0["correct"] and (not a1["correct"])
        fixed.append(100 * f / len(keys))
        broken.append(100 * b / len(keys))
    return fixed, broken


def f07_answer_flips(runs, out):
    think = [r for r in runs if r.thinking]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for r in think:
        fixed, broken = flips(r)
        x = r.cuts[1:]
        a1.plot(x, fixed, "-o", color=r.color, lw=2, ms=6, mec="white", mew=1.5, label=r.name)
        a2.plot(x, broken, "-o", color=r.color, lw=2, ms=6, mec="white", mew=1.5, label=r.name)
    for ax, t in ((a1, "Fixed: wrong at the previous budget, right now"),
                  (a2, "Broken: right at the previous budget, wrong now")):
        log2_axis(ax, think[0].cuts[1:])
        ax.set_xlabel("budget (compared with the one before)", color=INK)
        title(ax, t)
        style(ax)
    a1.set_ylabel("traces (%)", color=INK)
    legend(a2)
    save(fig, out, "07_answer_flips")


def f08_budget_needed(runs, out):
    """Smallest cut at which >= half of a question's samples are right ('unsolved' if none)."""
    think = [r for r in runs if r.thinking]
    cuts = think[0].cuts
    cats = [str(c) for c in cuts] + ["unsolved"]
    colors = blues(len(cuts)) + [NEUTRAL]
    fig, ax = plt.subplots(figsize=(10, 0.7 * len(think) + 1.6))
    for ri, r in enumerate(think):
        per_q = defaultdict(lambda: defaultdict(list))
        for a in r.answers:
            per_q[a["question_id"]][a["cut"]].append(a["correct"])
        counts = Counter()
        for q, by in per_q.items():
            need = next((str(c) for c in cuts if mean(by[c]) >= 0.5), "unsolved")
            counts[need] += 1
        left = 0
        n = len(per_q)
        for ci, cat in enumerate(cats):
            v = 100 * counts[cat] / n
            ax.barh(ri, v, left=left, color=colors[ci], edgecolor="white", linewidth=2,
                    label=(f"{int(cat) // 1024}K" if cat != "unsolved" and int(cat) >= 1024 else cat) if ri == 0 else None)
            if v >= 6:
                ax.text(left + v / 2, ri, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if ci >= len(cuts) * 0.45 and cat != "unsolved" else INK)
            left += v
    ax.set_yticks(range(len(think)))
    ax.set_yticklabels([r.name for r in think], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("questions (%)", color=INK)
    title(ax, "Smallest budget that solves each question (at least half of its samples right)")
    style(ax, "x")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=8, ncol=len(cats),
               loc="lower center", bbox_to_anchor=(0.5, 0.0))
    save(fig, out, "08_budget_needed", bottom=0.1)


CONF_BINS = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001]


def f09_calibration(runs, out):
    think = [r for r in runs if r.thinking]
    fig, axes = grid_axes(len(think), 3.4, 3.2)
    for ax, r in zip(axes, think):
        rows = [a for a in r.answers if conf(a) is not None]
        xs, ys, ns = [], [], []
        for lo, hi in zip(CONF_BINS, CONF_BINS[1:]):
            b = [a for a in rows if lo <= conf(a) < hi]
            if len(b) >= 15:
                xs.append(mean(conf(a) for a in b))
                ys.append(mean(a["correct"] for a in b))
                ns.append(len(b))
        ax.plot([0, 1], [0, 1], ls=":", color=MUTED, lw=1)
        ax.plot(xs, ys, "-o", color=r.color, lw=2, ms=6, mec="white", mew=1.5)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_title(r.name, color=INK, fontsize=9, loc="left")
        ax.set_xlabel("answer confidence", color=MUTED, fontsize=8)
        ax.set_ylabel("share right", color=MUTED, fontsize=8)
        style(ax)
    fig.suptitle("Calibration: is a confident answer a right answer? (all budgets pooled; dotted = perfect)",
                 color=INK, x=0.01, ha="left", fontsize=10)
    save(fig, out, "09_calibration")


def f10_confidence_auroc(runs, out):
    think = [r for r in runs if r.thinking]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for r in think:
        ys = []
        for c in r.cuts:
            rows = [a for a in r.by_cut[c] if conf(a) is not None]
            ys.append(auroc([conf(a) for a in rows], [a["correct"] for a in rows]))
        ax.plot(r.cuts, ys, "-o", color=r.color, lw=2, ms=6, mec="white", mew=1.5, label=r.name)
    ax.axhline(0.5, color=MUTED, ls=":", lw=1)
    log2_axis(ax, think[0].cuts)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("thinking budget", color=INK)
    ax.set_ylabel("AUROC", color=INK)
    title(ax, "How well answer confidence tells right from wrong", "0.5 = no information, 1.0 = perfect")
    style(ax)
    legend(ax, loc="lower left")
    save(fig, out, "10_confidence_auroc")


def f11_agreement(runs, out):
    fig, axes = grid_axes(len(runs), 5.0, 3.0)
    for ax, r in zip(axes, runs):
        per_q = defaultdict(list)
        for a in r.full:
            per_q[a["question_id"]].append(a)
        k = max(len(v) for v in per_q.values())
        stats = defaultdict(list)
        for rs in per_q.values():
            votes = Counter(vote_key(a["pred"]) for a in rs if a["valid"])
            if not votes:
                stats[0].append(0)
                continue
            best, size = votes.most_common(1)[0]
            stats[size].append(any(a["correct"] for a in rs if a["valid"] and vote_key(a["pred"]) == best))
        xs = list(range(1, k + 1))
        vals = [100 * mean(stats[x]) if stats[x] else 0 for x in xs]
        bars = ax.bar(xs, vals, 0.7, color=r.color, edgecolor="white", linewidth=2)
        for b, x in zip(bars, xs):
            if stats[x]:
                b.set_alpha(1.0 if len(stats[x]) >= 10 else 0.4)
                ax.text(x, b.get_height() + 2, f"{len(stats[x])}", ha="center", fontsize=7, color=INK)
        ax.set_ylim(0, 115)
        ax.set_xticks(xs)
        ax.set_title(r.name, color=INK, fontsize=9, loc="left")
        ax.set_xlabel(f"samples agreeing on the top answer (of {k})", color=MUTED, fontsize=8)
        ax.set_ylabel("majority right (%)", color=MUTED, fontsize=8)
        style(ax, "y")
    fig.suptitle("Agreement: when samples agree, is the shared answer right? (full budget)\n"
                 "number above a bar = questions; pale = fewer than 10 questions",
                 color=INK, x=0.01, y=0.99, ha="left", fontsize=10)
    save(fig, out, "11_agreement")


def full_stats(r):
    per_q = defaultdict(list)
    for a in r.full:
        per_q[a["question_id"]].append(a)
    maj = passk = 0
    for rs in per_q.values():
        votes = Counter(vote_key(a["pred"]) for a in rs if a["valid"])
        if votes:
            best = votes.most_common(1)[0][0]
            maj += any(a["correct"] for a in rs if a["valid"] and vote_key(a["pred"]) == best)
        passk += any(a["correct"] for a in rs)
    return {"accuracy": 100 * mean(a["correct"] for a in r.full),
            "majority vote": 100 * maj / len(per_q),
            "pass@k": 100 * passk / len(per_q),
            "valid answer": 100 * mean(a["valid"] for a in r.full)}


def f12_full_budget(runs, out):
    stats = {r.name: full_stats(r) for r in runs}
    metrics = list(next(iter(stats.values())))
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 0.45 * len(runs) + 1.8))
    for ax, m in zip(axes, metrics):
        vals = [stats[r.name][m] for r in runs]
        bars = ax.barh(range(len(runs)), vals, color=[r.color for r in runs], height=0.7,
                       edgecolor="white", linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center",
                    fontsize=8, color=INK)
        ax.set_yticks(range(len(runs)))
        ax.set_yticklabels([r.name for r in runs] if ax is axes[0] else [], fontsize=8, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, 115)
        ax.set_title(m + " (%)", color=INK, fontsize=10, loc="left")
        style(ax, "x")
    fig.suptitle("At the full budget (k = samples per question)", color=INK, x=0.01, ha="left")
    save(fig, out, "12_full_budget")


def guard_exact(r):
    """True if every trace stopped by the guard has its answer AT the loop point recorded."""
    return all((q, s) in r.loop_ans for (q, s), t in r.tr.items()
               if t["loop_at"] is not None and t["loop_at"] < t["reasoning_tokens"])


def guard_estimate(r):
    """Replay of 'stop when a loop is detected': a looping trace costs only the tokens up to
    its loop point. EXACT when the answer at the loop point is recorded (loop_answers.jsonl);
    otherwise ROUGH: the answer at the largest recorded cut <= the loop point is used."""
    acc, tok, stopped = [], [], 0
    exact = guard_exact(r)
    for (q, s), t in r.tr.items():
        L = t["loop_at"]
        if L is not None and L < t["reasoning_tokens"]:
            if exact:
                a = r.loop_ans[(q, s)]
                stopped += 1
                acc.append(a["correct"])
                tok.append(a["total_tokens"])
                continue
            below = [c for c in r.cuts if c <= L]
            a = r.ans.get((q, s, below[-1])) if below else None
            if a is None:
                continue
            stopped += 1
            acc.append(a["correct"])
            tok.append(L + a["answer_tokens"])
        else:
            a = r.ans.get((q, s, r.top))
            if a is None:
                continue
            acc.append(a["correct"])
            tok.append(a["total_tokens"])
    return 100 * mean(acc), mean(tok), 100 * stopped / max(len(r.tr), 1)


def f13_loop_guard(runs, out):
    think = [r for r in runs if r.thinking]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 0.5 * len(think) + 2))
    y = list(range(len(think)))
    raw_acc = [r.acc(r.top) for r in think]
    raw_tok = [mean(a["total_tokens"] for a in r.full) for r in think]
    est = [guard_estimate(r) for r in think]
    h = 0.38
    for i, r in enumerate(think):
        # accuracy as dots (no zero baseline needed, so the small differences stay readable)
        a1.plot([raw_acc[i], est[i][0]], [i, i], color=NEUTRAL, lw=1.5, zorder=1)
        a1.plot(raw_acc[i], i, "o", color=r.color, ms=9, mec="white", mew=2, zorder=2,
                label="no guard" if i == 0 else None)
        a1.plot(est[i][0], i, "o", color="white", ms=9, mec=r.color, mew=2, zorder=3,
                label="with guard" if i == 0 else None)
        a1.text(max(raw_acc[i], est[i][0]) + 0.4, i, f"{raw_acc[i]:.1f} → {est[i][0]:.1f}",
                va="center", fontsize=7, color=INK)
        a2.barh(i - h / 2, raw_tok[i], h, color=r.color, edgecolor="white", linewidth=1.5)
        a2.barh(i + h / 2, est[i][1], h, color=r.color, alpha=0.45, hatch="///", edgecolor="white", linewidth=1.5)
        a2.text(raw_tok[i] * 1.01, i - h / 2, f"{raw_tok[i]:,.0f}", va="center", fontsize=7, color=INK)
        a2.text(est[i][1] * 1.01, i + h / 2,
                f"{est[i][1]:,.0f}  (-{100 * (1 - est[i][1] / raw_tok[i]):.0f}%, {est[i][2]:.0f}% stopped)",
                va="center", fontsize=7, color=INK)
    for ax in (a1, a2):
        ax.set_yticks(y)
        ax.set_yticklabels([r.name for r in think] if ax is a1 else [], fontsize=8, color=INK)
        ax.invert_yaxis()
        style(ax, "x")
    lo = min(min(raw_acc), min(e[0] for e in est))
    a1.set_xlim(max(0, lo - 3), 101)
    a1.set_ylim(len(think) - 0.4, -0.6)
    a2.set_xlim(0, max(raw_tok) * 1.55)
    title(a1, "Accuracy (%)")
    title(a2, "Mean tokens per answer")
    a1.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    a2.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=NEUTRAL, label="no guard"),
                       plt.Rectangle((0, 0), 1, 1, facecolor=NEUTRAL, alpha=0.45, hatch="///",
                                     label="with guard")],
              frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    kinds = {guard_exact(r) for r in think}
    fig.suptitle("Stop on loop: EXACT replay (answer forced at the loop point)" if kinds == {True} else
                 "Stop on loop: ROUGH estimate where the loop answer is missing (answer at the last cut "
                 "before the loop); exact: " + (", ".join(r.name for r in think if guard_exact(r)) or "none"),
                 color=INK, x=0.01, ha="left", fontsize=10)
    save(fig, out, "13_loop_guard_estimate")


def f14_reasoning_levels(runs, out):
    """Accuracy and cost of each kind of reasoning: none, written in the answer, thinking
    (the best thinking config, at each cut)."""
    rows, cols = [], []                                 # (label, run, cut), bar colour
    for ds in dict.fromkeys(getattr(r, "dataset", None) for r in runs):   # one block per dataset
        group = [r for r in runs if getattr(r, "dataset", None) == ds]
        think = [r for r in group if r.thinking]
        best = max(think, key=lambda r: r.acc(r.top))
        for r in sorted((r for r in group if not r.thinking), key=lambda r: not CONFIGS[r.config].get("direct")):
            kind = "no reasoning" if CONFIGS[r.config].get("direct") else "reasoning in the answer"
            rows.append((f"{r.name}  ({kind})", r, None))
            cols.append(r.color)
        shades = blues(len(best.cuts)) if ds is None else ramp(best.color, len(best.cuts))
        for c, col in zip(best.cuts, shades):
            rows.append((f"{best.name}  thinking, budget {ktick(c)}", best, c))
            cols.append(col)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 0.36 * len(rows) + 1.6), sharey=True,
                                 gridspec_kw=dict(width_ratios=[3, 2]))
    y = range(len(rows))
    for ax, vals, fmt in ((a1, [r.acc(c) for _, r, c in rows], "{:.1f}"),
                          (a2, [mean(a["total_tokens"] for a in r.by_cut[c]) for _, r, c in rows], "{:,.0f}")):
        bars = ax.barh(y, vals, color=cols, height=0.7, edgecolor="white", linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_width() * (1.04 if ax is a2 else 1) + (0 if ax is a2 else 1),
                    b.get_y() + b.get_height() / 2, fmt.format(v), va="center", fontsize=8, color=INK)
        style(ax, "x")
    a1.set_yticks(list(y))
    a1.set_yticklabels([l for l, _, _ in rows], fontsize=8, color=INK)
    a1.invert_yaxis()
    a1.set_xlim(0, 110)
    a2.set_xscale("log")
    a2.set_xlim(8, max(mean(a["total_tokens"] for a in r.by_cut[c]) for _, r, c in rows) * 3)
    a2.minorticks_off()
    a2.set_xticks([16, 128, 1024, 8192])
    a2.set_xticklabels(["16", "128", "1K", "8K"])
    title(a1, "Accuracy (%)")
    title(a2, "Mean tokens per answer (log scale)")
    fig.suptitle("How much reasoning buys: none, written in the answer, or thinking at each budget",
                 color=INK, x=0.01, y=0.99, ha="left", fontsize=10)
    save(fig, out, "14_reasoning_levels")


def f15_effort_levels(runs, out):
    fig, ax = plt.subplots(figsize=(9.5, 6))
    subs, sizes = [], {}
    think = [r for r in runs if r.thinking and getattr(r, "split_of", None)]
    for k, r in enumerate(think):
        acc = {c: r.acc(c) for c in r.cuts}
        lv, lv_train = effort_levels(acc), effort_levels(split_acc(r, "train"))
        for split, marker in (("train", "s"), ("val", "^"), ("test", "v")):
            a = split_acc(r, split)
            n = len({x["question_id"] for x in r.full if r.split_of[x["question_id"]] == split})
            ax.plot(r.cuts, [a[c] for c in r.cuts], marker=marker, color=r.color, alpha=0.35, lw=1, ms=4)
            sizes[split] = sizes.get(split, []) + [f"{n}"]
        ax.plot(r.cuts, [acc[c] for c in r.cuts], "-o", color=r.color, lw=2.5, ms=7, mec="white", mew=2,
                label=f"{r.name}, merged ({len({x['question_id'] for x in r.full})} questions)")
        for share in (0.85, 0.97):
            ax.plot([r.cuts[0], r.cuts[-1]], [share * acc[lv["H"]]] * 2, color=r.color, lw=0.8, ls=":", alpha=0.7)
        dy = 16 if k % 2 == 0 else -24                  # one dataset's labels above its curve, the next below
        for name, c in lv.items():
            ax.annotate(f"{name} {c // 1024}K" if c >= 1024 else f"{name} {c}", (c, acc[c]), xytext=(0, dy),
                        textcoords="offset points", ha="center", color=INK, fontsize=9, fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=r.color, lw=1))
        fmt = lambda d: ", ".join(f"{x} {v // 1024}K" if v >= 1024 else f"{x} {v}" for x, v in d.items())  # noqa: E731
        subs.append(f"{r.name}: {fmt(lv)} (train only: {fmt(lv_train)})")
    for r in runs:
        if not r.thinking:
            ax.axhline(r.acc(None), color=r.color, lw=1.5, ls="--", label=f"{r.name} (no reasoning)")
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    for split, marker in (("train", "s"), ("val", "^"), ("test", "v")):   # faint lines: one entry per split
        handles.append(Line2D([], [], color=NEUTRAL, marker=marker, lw=1, ms=5))
        labels.append(f"{split} only ({' / '.join(sizes[split])} questions)")
    ax.legend(handles, labels, frameon=True, facecolor="white", edgecolor="none", framealpha=0.95,
              fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.02))
    cuts = sorted({c for r in think for c in r.cuts})
    log2_axis(ax, cuts)
    ax.set_xlabel("thinking budget (tokens before the answer is forced)", color=INK)
    ax.set_ylabel("accuracy (%)", color=INK)
    title(ax, "Effort levels by the rule (train + val + test merged); dotted = 85 / 97 % of acc@H",
          "\n".join(subs))
    style(ax)
    save(fig, out, "15_effort_levels")


def write_csv(runs, out):
    """One line per config with the numbers behind the figures."""
    lines = []
    for r in runs:
        s = full_stats(r)
        line = dict(config=r.name, traces=len(r.traces), **{k: round(v, 1) for k, v in s.items()})
        if r.thinking:
            for c in r.cuts:
                line[f"acc@{c}"] = round(r.acc(c), 1)
            if getattr(r, "split_of", None):
                line.update({f"level_{k}": v for k, v in effort_levels({c: r.acc(c) for c in r.cuts}).items()})
                line.update({f"train_level_{k}": v for k, v in effort_levels(split_acc(r, "train")).items()})
            fixed, broken = flips(r)
            ga, gt, gs = guard_estimate(r)
            line.update(
                ended_by_itself=round(100 * mean(t["natural_end"] for t in r.traces), 1),
                loop_flagged=round(100 * mean(t["loop_detected"] for t in r.traces), 1),
                median_reasoning=statistics.median(t["reasoning_tokens"] for t in r.traces),
                mean_tokens_full=round(mean(a["total_tokens"] for a in r.full)),
                fixed_total=round(sum(fixed), 1), broken_total=round(sum(broken), 1),
                confidence_auroc_full=round(auroc([conf(a) for a in r.full],
                                                  [a["correct"] for a in r.full]), 3),
                guard_acc=round(ga, 1), guard_tokens=round(gt), guard_stopped=round(gs, 1), guard_exact=int(guard_exact(r)))
        else:
            line.update(mean_tokens_full=round(mean(a["total_tokens"] for a in r.full)))
        lines.append(line)
    keys = list(dict.fromkeys(k for l in lines for k in l))
    with open(out / "analysis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(lines)


def main():
    global plt
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="gsm8k", choices=list(DATA_DIRS) + ["both"],
                   help="'both' = GSM8K, MATH and AMC in the same figures (with --set all)")
    p.add_argument("--set", default="config_exp",
                   help="gsm8k: config_exp, train, val, test; math: math_pilot, math_train, math_val, "
                        "math_test; amc23: amc23; 'all' = train + val + test merged")
    p.add_argument("--all", action="store_true",
                   help="config_exp, math_pilot, amc23 (if recorded) and the merged train+val+test of each dataset")
    p.add_argument("--out", type=Path, default=None, help="output folder (default: see above)")
    args = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    plt = _plt
    plt.rcParams["hatch.color"] = "white"
    if args.all:
        for dataset, d in DATA_DIRS.items():
            for res in sorted(p for p in (d / "results").glob("*") if p.is_dir()):   # none yet: skipped
                if res.name not in MERGED.get(dataset, []):   # the splits are drawn merged below
                    plot_set(dataset, res.name, figure_dir(dataset, res.name))
        plot_set("both", "all", figure_dir("both", "all"))  # GSM8K + MATH + AMC, train+val+test merged
    else:
        plot_set(args.dataset, args.set, args.out or figure_dir(args.dataset, args.set))


def plot_set(dataset, set_name, out):
    runs = load_both(set_name) if dataset == "both" else load(dataset, set_name)
    if not runs:
        raise SystemExit(f"No results for {dataset} {set_name}. Record the set first.")
    out.mkdir(parents=True, exist_ok=True)
    figs = [f01_accuracy_vs_budget, f02_accuracy_vs_tokens, f12_full_budget, f11_agreement]
    if any(r.thinking for r in runs):
        figs += [f03_still_thinking, f04_length_histograms, f05_accuracy_by_ending, f06_loop_onset,
                 f07_answer_flips, f08_budget_needed, f09_calibration, f10_confidence_auroc]
        if not all(t.get("stop_on_loop") for r in runs if r.thinking for t in r.traces):
            figs.append(f13_loop_guard)                 # with --stop-on-loop the guard is already applied
        if any(not r.thinking for r in runs):
            figs.append(f14_reasoning_levels)
        if set_name == "all":
            figs.append(f15_effort_levels)
    for f in figs:
        f(runs, out)
    write_csv(runs, out)
    print(f"Wrote {len(figs)} figures and analysis.csv to {out}")


if __name__ == "__main__":
    main()
