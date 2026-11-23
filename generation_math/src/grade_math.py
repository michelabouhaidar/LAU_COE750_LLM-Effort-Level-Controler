"""Grading for MATH / MATH-500 / AMC answers, and the paths of generation_math
(generation_amc23/src/paths_amc23.py points DATA at generation_amc23/data).

GSM8K answers are plain numbers; MATH answers are expressions (\\frac{3}{4}, 2\\sqrt{5},
(-\\infty, 3], x^2+1, \\text{(C)}, ...), so the GSM8K number grader is not enough here.

  pred    the content of the answer box, normalised (used to group equal answers for majority
          vote / agreement): the first box of a forced answer, else the last complete box
  valid   a non-empty box was found
  correct Math-Verify says it equals the reference, or the normalised strings are equal
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                 # .../generation_math
DATA = ROOT / "data"
GSM8K_SRC = ROOT.parent / "generation_gsm8k" / "src"        # shared server client, loop rule, prompt
sys.path.insert(0, str(GSM8K_SRC))
sys.path.insert(0, str(GSM8K_SRC / "sweep"))

from common import answer_box, last_boxed, to_number  # noqa: E402

try:
    from math_verify import parse, verify
    GRADER = "math_verify"
except ImportError:                                        # fallback: string / number match only
    parse = verify = None
    GRADER = "string"


def normalize(s):
    """A canonical string for an answer: spacing, \\left/\\right, \\dfrac, \\text{}, units,
    degrees, '$', trailing '.' and '.0' stripped. Used for grouping and as a fallback match."""
    if s is None:
        return None
    t = s.strip()
    t = re.sub(r"\\text\{\s*([^{}]*?)\s*\}", r"\1", t)
    t = re.sub(r"\\(?:mbox|mathrm|textbf|operatorname)\{([^{}]*)\}", r"\1", t)
    for a, b in (("−", "-"), ("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\ ", ""),
                 ("dfrac", "frac"), ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""),
                 ("\\%", ""), ("%", ""), ("$", ""), ("\\displaystyle", "")):
        t = t.replace(a, b)
    t = re.sub(r"\s+", "", t)
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)          # 1,234 -> 1234
    t = t.rstrip(".")
    t = re.sub(r"^([a-zA-Z]|\\[a-z]+)=", "", t)            # x=5 -> 5
    v = to_number(t) if re.fullmatch(r"-?[\d.]+", t or "") else None
    if v is not None:
        t = str(int(v)) if float(v).is_integer() else repr(round(v, 9))
    return t


def _mv_equal(reference, box):
    try:
        gold = parse("$" + reference + "$")
        pred = parse("\\boxed{" + box + "}")
        return bool(gold and pred and verify(gold, pred))
    except Exception:
        return False


def grade(answer_text, reference):
    """Same keys as the GSM8K grader: pred, valid, correct (+ the grader used)."""
    box = answer_box(answer_text or "")
    pred = normalize(box)
    valid = bool(pred)
    correct = False
    if valid and reference is not None:
        correct = pred == normalize(reference) or (verify is not None and _mv_equal(reference, box))
    return dict(pred=pred if valid else None, valid=int(valid), correct=int(correct), grader=GRADER)
