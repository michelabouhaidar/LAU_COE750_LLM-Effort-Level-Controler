"""The configs compared in the sweep. Edit this file to add or change one.

Fixed for every thinking config (Qwen3.5 model card values): top_p 0.95, top_k 20, min_p 0.
What changes: temperature and presence_penalty.

Background:
  * temperature 1.0 with presence_penalty 1.5 was WORSE than 0.6 (more rambling, -15 points
    at 512 and 2,048 tokens), so that pair is not included;
  * presence_penalty reduces word-for-word loops
    but may push the model to re-check the same thing in new words, so it is the main unknown.

Each config: name -> dict(thinking=..., sampling=..., [max_tokens for no-thinking]).
The order here is the order (and colour) used in the plots.
"""

_THINK = dict(top_p=0.95, top_k=20, min_p=0.0)

CONFIGS = {
    # the reference setting
    "t0.6_pp1.5": dict(thinking=True, sampling=dict(_THINK, temperature=0.6, presence_penalty=1.5)),
    # same temperature, no penalty: does the penalty help or hurt?
    "t0.6_pp0.0": dict(thinking=True, sampling=dict(_THINK, temperature=0.6, presence_penalty=0.0)),
    # halfway
    "t0.6_pp0.75": dict(thinking=True, sampling=dict(_THINK, temperature=0.6, presence_penalty=0.75)),
    # a bit warmer with a mild penalty
    "t0.8_pp0.75": dict(thinking=True, sampling=dict(_THINK, temperature=0.8, presence_penalty=0.75)),
    # the card's general thinking temperature, without penalty (only 1.0 + 1.5 was tried before)
    "t1.0_pp0.0": dict(thinking=True, sampling=dict(_THINK, temperature=1.0, presence_penalty=0.0)),
    # thinking switched off; the model may still write a worked solution in its answer.
    # Sampling = the card's non-thinking setting. No cuts: one answer of up to max_tokens.
    "nothink": dict(thinking=False, max_tokens=2048,
                    sampling=dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5)),
    # NO reasoning at all: thinking off and the answer is started for the model with
    # "The final answer is \\boxed{", so it can only write the number (16 tokens).
    # Same sampling as nothink, so the two differ only in the written-out reasoning.
    "direct": dict(thinking=False, direct=True, max_tokens=16,
                   sampling=dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5)),
}
