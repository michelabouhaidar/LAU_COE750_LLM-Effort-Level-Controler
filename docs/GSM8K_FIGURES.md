# GSM8K analysis figures: what each one shows and how it was computed

The figures are drawn by `src/analysis/plot_gsm8k.py` from the data recorded by
`src/sweep/gsm8k_generate.py`. The script only reads the data; it generates nothing.

```bash
~/.venvs/vllm/bin/python ~/Desktop/RL_experiments/src/analysis/plot_gsm8k.py --split val
```

The same command with `--split train`, `--split test` or `--split all` draws the figures for the
other splits. `all` reads the three splits together (1,450 questions, 8,700 traces) and writes to
`figures/gsm8k_all/`; every figure is computed in the same way as described here.

The figures below are for the **validation split**: 150 questions, each with 6 answers without
reasoning and 6 reasoning traces. Output folder: `figures/gsm8k_val/` (figures and `summary.csv`).

---

## The data behind all figures

Everything comes from three files in `data/gsm8k/val/`.

| File | One line per | Lines | Used for |
| --- | --- | --- | --- |
| `noreason.jsonl` | answer given without reasoning (level 0) | 900 | level 0 in figures 1 and 7 |
| `rows.jsonl` | reasoning trace × effort level (512, 2,048, 8,192) | 2,700 | figures 1, 3, 4, 5, 6, 7 |
| `traces.jsonl` | full reasoning trace | 900 | figure 2 |

Fields used:

| Field | File | Meaning |
| --- | --- | --- |
| `question_id` | all | which question |
| `checkpoint_budget` | rows, noreason | the effort level of the answer: 0, 512, 2,048 or 8,192 |
| `correct` | rows, noreason | 1 if the number in the model's `\boxed{}` equals the GSM8K reference number, otherwise 0 |
| `valid` | rows, noreason | 1 if the answer contains a usable `\boxed{}` |
| `reasoning_tokens_so_far` | rows | reasoning tokens used at that level (the level itself if the trace was cut, less if it had finished) |
| `answer_tokens` | rows, noreason | tokens of the answer |
| `cut` | rows | 1 if the reasoning was interrupted at that level and the answer forced; 0 if the reasoning had ended by itself |
| `answer_mean_logprob` | rows | mean log-probability of the answer tokens |
| `tail_entropy_mean` | rows | mean entropy of the last 64 reasoning tokens before the level |
| `hedge_count` | rows | number of hedging expressions ("wait", "alternatively", "hmm", "let me check", "actually", "on second thought") in the reasoning up to the level |
| `cluster_key` | rows | label shared by answers with the same number |
| `reasoning_ids` | traces | the reasoning as token numbers; its length is the reasoning length |
| `natural_end` | traces | 1 if the model ended its reasoning by itself before 8,192 tokens |

Two quantities are derived in the plotting script:

- **tokens of an answer** = `reasoning_tokens_so_far` + `answer_tokens`
- **answer confidence** = exp(`answer_mean_logprob`), the geometric mean of the probabilities the
  model gave to its answer tokens (between 0 and 1)

---

## Figure 1: accuracy against token cost

![Figure 1](../figures/gsm8k_val/1_accuracy_vs_tokens.png)

**Question it answers.** What does each effort level cost and deliver, and how much could be saved
by choosing the level per question?

**Blue line (same level for every question).** For each level, two averages over all 900 answers
at that level:

- y = mean of `correct`, in percent
- x = mean of (`reasoning_tokens_so_far` + `answer_tokens`)

**Orange line (best level per question).** For each question and level, the accuracy and the
tokens are first averaged over the question's 6 runs. Then, for a price λ, each question is given
the level with the highest value of

> accuracy − λ × tokens ÷ (mean tokens at level 8,192)

and accuracy and tokens are averaged over the 150 questions. This is repeated for
λ = 0.02, 0.05, 0.10, 0.25 and 0.50. The values 0.02, 0.05 and 0.10 lead to the same choice, so
they share one point.

**Caution.** The orange line uses the true outcomes to choose, which no real method can do. It is
an upper bound for choosing one level per question, not a result.

| Level | Accuracy | Mean tokens |
| --- | --- | --- |
| 0 | 9.7% | 8 |
| 512 | 52.9% | 526 |
| 2,048 | 85.2% | 1,833 |
| 8,192 | 89.9% | 5,251 |
| Best per question, λ ≤ 0.10 | 90.6% | 2,451 |

---

## Figure 2: how long the reasoning runs

![Figure 2](../figures/gsm8k_val/2_reasoning_length.png)

**Question it answers.** Where do the tokens go, and how often does the model fail to conclude?

**Computation.** From `traces.jsonl`, the length of `reasoning_ids` for each of the 900 traces.

- Blue: traces with `natural_end` = 1, counted in bins of 512 tokens.
- Orange: traces with `natural_end` = 0, that is, stopped by the 8,192-token limit.
- Dotted lines: the effort levels 512 and 2,048.

**Reading.** 521 traces finished by themselves; 379 (42%) were still reasoning at 8,192.

---

## Figure 3: finished traces against cut traces

![Figure 3](../figures/gsm8k_val/3_finished_vs_cut.png)

**Question it answers.** Is "the model finished its reasoning" a sign that the answer is right?

**Computation.** From `rows.jsonl`. At each level the 900 answers are split by `cut`:

- `cut` = 0 (blue): the reasoning had ended by itself at or before this level
- `cut` = 1 (orange): the reasoning was interrupted at this level and the answer was forced

Each bar is the mean of `correct` in that group; n is the number of answers. A pale bar rests on
fewer than 30 answers (only 3 traces finish within 512 tokens).

---

## Figure 4: answer confidence against accuracy

![Figure 4](../figures/gsm8k_val/4_confidence_calibration.png)

**Question it answers.** When the model is confident in its answer, is the answer right?

**Computation.** From `rows.jsonl`, separately for each level:

1. answer confidence = exp(`answer_mean_logprob`)
2. the answers are sorted into confidence bins: 0–0.5, 0.5–0.7, 0.7–0.8, 0.8–0.9, 0.9–0.95,
   0.95–0.99, 0.99–1
3. each point is one bin: x = mean confidence in the bin, y = mean of `correct` in the bin
4. bins with fewer than 15 answers are not drawn

**Reading.** The dotted diagonal is perfect calibration (a confidence of 0.8 would mean 80%
correct). Points below it mean the model is over-confident there; points above, under-confident.

---

## Figure 5: agreement between the six traces

![Figure 5](../figures/gsm8k_val/5_agreement.png)

**Question it answers.** When several independent runs give the same answer, is that answer right?

**Computation.** From `rows.jsonl`, separately for each level and each question:

1. the 6 answers are grouped by `cluster_key` (answers with the same number share a key; invalid
   answers are left out)
2. the largest group is the majority answer
3. x = the size of that group (1 to 6)
4. the question counts as correct if the majority answer has `correct` = 1

Each bar is the share of questions, among those with that group size, whose majority answer is
correct; n is the number of questions. A pale bar rests on fewer than 10 questions.

**Caution.** This uses all 6 traces at once. The controller in the design would see 3 or 5.

---

## Figure 6: how well each signal predicts a correct answer

![Figure 6](../figures/gsm8k_val/6_signal_auroc.png)

**Question it answers.** Which observable signals separate right answers from wrong ones?

**Computation.** From `rows.jsonl`, separately for each level, over the 900 answers. For each
signal the AUROC is computed: the probability that a randomly chosen correct answer has a higher
signal value than a randomly chosen wrong answer (ties count one half). 0.5 means the signal
carries no information; 1.0 means it separates perfectly.

| Signal | Value used | Higher means |
| --- | --- | --- |
| Answer confidence | exp(`answer_mean_logprob`) | the model generated its answer with high probability |
| Reasoning finished | 1 − `cut` | the reasoning ended by itself |
| Low entropy at the end | −`tail_entropy_mean` | the model was decided in its last 64 reasoning tokens |
| Few hedging words | −`hedge_count` ÷ `reasoning_tokens_so_far` | fewer hedging expressions per token |

**Caution.** Each signal is tested alone. The controller will combine them, together with the
embeddings, which are not yet computed.

---

## Figure 7: the lowest level that solves each question

![Figure 7](../figures/gsm8k_val/7_level_needed.png)

**Question it answers.** How much effort do the questions actually need?

**Computation.** From `noreason.jsonl` and `rows.jsonl`:

1. for each question and level, accuracy = mean of `correct` over its 6 runs
2. a level "solves" the question if that accuracy is at least 50% (3 of 6 runs correct)
3. the question is assigned to the lowest level that solves it, or to "not solved" if none does

**Caution.** The 50% threshold is a choice. A stricter one (for example 5 of 6 runs) would move
questions to higher levels.

---

## Settings under which the data were generated

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3.5-2B`, weights version `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Server | vLLM 0.33.0 |
| Reasoning runs | temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5 |
| No-reasoning runs | temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5 |
| Effort levels | 0, 512, 2,048 and 8,192 reasoning tokens |
| Answer after reasoning | the text `The final answer is \boxed{` is inserted and the model fills the box (at most 16 tokens) |
| Grading | the number in the last `\boxed{}` must equal the GSM8K reference number |
