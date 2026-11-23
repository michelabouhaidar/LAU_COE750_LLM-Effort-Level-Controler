# Closed-loop controller, question + every measurement (FULL), Qwen3.5-2B

**Code:** `rl_training/closed_loop_training.ipynb`. **The LLM is frozen; only the controller is trained.**
**Companion files:** [BANDIT.md](BANDIT.md), [OPEN_LOOP.md](OPEN_LOOP.md), [CLOSED_LOOP_API.md](CLOSED_LOOP_API.md) (question + closed-API measurements). They share the data, levels, action space, replay, reward and question representation. They differ only in the state.

**What this controller answers:** the best a controller can do when it sees **everything** Qwen can
measure: the upper bound. Its gap to the API controller is the **price of closed access**.

---

## 1. Data

| Dataset | train | val | test | Replay files (`controller_data/`) |
|---|---|---|---|---|
| GSM8K | 919 | 200 | 200 | `gsm8k_{train,val,test}_*` |
| MATH | 400 | 99 | 200 | `math_{train,val,test}_*` |
| AMC 2023 | 24 | 8 | 8 | `amc23_{train,val,test}_*` |
| **total** | **1,343** | **307** | **408** | 10 traces per question |

No question is in two splits. `config_exp` (100) is not used (4 traces, no loop stop).

**All datasets are trained together** (shared rules, the same in all four controller files):
- **Sampling:** pick a dataset with probability ∝ √(its train questions): GSM8K 54.9%, MATH 36.2%, AMC 8.9%.
  Then pick a question uniformly inside it.
- **No dataset id in the state.** Levels are seen by name (L/M/H), and tokens are relative to the dataset.
- **Constants:** C_ref per dataset; P95s pooled over all train questions.
- **Answer:** every answer is forced to start at `\boxed{`. The answer is the content of that first box,
  normalised (`pred`); anything written after it is ignored. `valid` = the box holds a readable number
  (GSM8K) or is non-empty (MATH, AMC).
- **Answer clusters:** GSM8K: equal numbers. MATH and AMC: equal `pred`, or Math-Verify finds the two boxes
  equal in either direction; connected answers form one cluster, so `7/2`, `3.5` and `\frac{7}{2}` are one
  answer. The reference and `correct` are never used. On MATH train, clusters over the levels
  contain both a right and a wrong answer.
- **Selection:** validation utility, weighted by the sampling shares above.
- **Reporting:** every result per dataset. AMC (8 test questions) is descriptive only.

**Recorded per question (the same for every dataset; GSM8K command below):** 10 independent traces (`t0.6_pp1.5`, `--stop-on-loop`). Per trace:
token ids, length, how it ended (natural / loop / cap), `loop_at`. Per answer (at every cut before the
end, and once at the end): text, boxed answer (`pred`), valid, correct, `box_tokens`, `box_confidence`, the
answer-token log-probs (16 tokens on GSM8K, 48 on MATH).

```bash
~/.venvs/vllm/bin/python generation_gsm8k/src/sweep/run_sweep.py run --set train --configs t0.6_pp1.5 --samples 10 --cuts 256 512 1024 2048 4096 6144 8192 12288 16384 --stop-on-loop --concurrency 16
```

## 2. Levels

| Level | GSM8K budget | Train, one run (accuracy / tokens) | MATH budget | Train, one run (accuracy / tokens) |
|---|---|---|---|---|
| L | 2,048 | 84.5% / 1,856 | 4,096 | 73.5% / 3,616 |
| M | 4,096 | 88.5% / 3,290 | 12,288 | 81.7% / 7,583 |
| H | 16,384 | 90.3% / 6,872 | 16,384 | 83.0% / 8,740 |

Levels follow the per-dataset rule (GENERATION-DOC section 6).

| AMC level (32K cap) | Budget | Train, one run: accuracy | Tokens |
|---|---|---|---|
| L | 12,288 | 54.2% | 9,089 |
| M | 24,576 | 59.2% | 12,314 |
| H | 32,768 | 59.6% | 13,004 |

A run at level ℓ = one trace read at cut ℓ.

## 3. Action space (the same in all closed-loop files)

| t | Action | Effect |
|---|---|---|
| 0 | RUN_L, RUN_M, RUN_H | 1 run at that level |
| ≥ 1 | STOP | return the final answer |
| ≥ 1 | ADD | 1 more run at the current level |
| ≥ 1 | ESC_M, ESC_H | current answers → history; 1 new run at a higher level |

**Mask:** ESC only upward; ADD only if K < 5; ADD and ESC only if runs < 6; at the 6th decision (t = 5) only STOP.
Every run is a new independent call, so the action space is a legal sequence of API calls. There are 55 legal plans.

## 4. Replay environment

```
reset(q, seed):   unused = {0..9};  level = None;  K = 0;  history = [];  t = 0;  spent = 0
run(ℓ):           k = draw from unused (seeded);  unused −= {k};  return row(q, k, ℓ)
step(a):          apply the action, charge its tokens, build the next state; done if STOP
```

`row(q, k, ℓ)` (from the converter): answer, valid, correct, `finish_reason` (`length` if the trace
reached ℓ without ending, else `stop`), tokens = min(length, ℓ) + answer tokens, and the FULL
measurements of section 5.

**Final answer:** the majority cluster of the valid answers at the current level. Ties go to the
cluster with the higher mean `box_conf`. If none is valid: the latest valid answer in the history,
else wrong.

**Exact values:** the state depends only on *which* traces are in hand. So the expected value of an
action is an average over the unused traces (at most 10), which is small enough to enumerate.

## 5. State and measurements

Constants (P95s, C_ref) are computed on train only. Everything after t = 0 is 0 at t = 0. `correct` is
never in the state.

| Block | Feature | Computation | Size |
|---|---|---|---|
| question | `q_emb` | MiniLM question embedding (`all-MiniLM-L6-v2`, 384 dims) reduced by PCA fitted on the 1,343 train questions only (`controller_data/embeddings/<set>_q_64.npy`, 65.1% of variance); frozen (same input for all controllers) | 64 |
| | `prompt_len_norm` | min(prompt tokens / P95, 1) | 1 |
| | `n_numbers_norm` | min(numbers in the question / P95, 1) | 1 |
| position | `level_onehot` | L, M, H | 3 |
| | `step_norm`, `k_norm`, `runs_norm` | t / 5, K / 5, runs used / 6 | 3 |
| | `cost_norm` | min(tokens spent / C_ref, 3) | 1 |
| response | `valid_rate` | share of the K answers with a valid boxed answer | 1 |
| | `cut_rate` | share with `finish_reason = length` (stopped by the budget) | 1 |
| | `len_norm` | mean of tokens / level budget | 1 |
| agreement | `majority_ratio` | largest valid cluster / K | 1 |
| | `cluster_entropy` | −Σ (n_c/K) log(n_c/K) / log K; 0 if K = 1 | 1 |
| history | `has_prev` | an answer exists at a lower level | 1 |
| | `agrees_prev` | current majority = majority of the previous level | 1 |
| | `distinct_norm` | distinct valid clusters over all runs / runs used | 1 |
| **confidence** (FULL only) | `conf_mean`, `conf_min` | mean / min over the K answers of `box_conf` = exp(mean log-prob of the answer tokens up to the closing `}`) | 2 |
| **text** (FULL only) | `hedge_rate` | per 100 reasoning tokens up to the cut: "wait", "alternatively", "hmm", "let me check", "actually", "but maybe"; / P95, capped at 1; mean over K | 1 |
| | `tail_repetition` | share of repeated 20-token pieces in the last 2,048 reasoning tokens before the cut (the loop rule's measure, as a number); mean over K | 1 |
| | `loop_rate` | share of the K traces whose loop flag had fired by the cut | 1 |

**Size:** 64 (PCA) + **22 scalars**. The same as the API controller plus the 5 FULL measurements.

**Measurements and where they come from:**

| Measurement | Source in the recordings | Available on a closed API |
|---|---|---|
| answer, validity, agreement | answer text, `pred`, `valid` | yes |
| ended or cut | `end_kind`, trace length vs level | yes (`finish_reason`) |
| tokens | trace length, answer tokens | yes (usage) |
| answer confidence | `answer_logprobs`, `box_tokens` | **no** |
| hedges, repetition, loop | `reasoning_ids` decoded up to the cut | **no** (reasoning hidden) |

## 6. Reward

```
r_t  = −λ · c_t / C_ref          c_t = tokens of the runs the action launches (0 for STOP)
r_T += 1[final answer correct]   at STOP
R    = 1[correct] − λ · C / C_ref ,    γ = 1
```

C_ref = mean tokens of one H run on the dataset's train split: **GSM8K 6,872, MATH 8,740, AMC 13,004**. λ ∈ {0.02, 0.05, 0.10, 0.25, 0.50}.
Best fixed plan at λ = 0.10 (train): GSM8K M×1 = 0.885 − 0.1·3,290/6,872 = **0.837**; MATH H×1 = 0.830 − 0.1 = **0.730** (tied with M×1); AMC M×1 = 0.592 − 0.1·12,314/13,004 = **0.497**.

## 7. Network

| Part | Layers | Parameters |
|---|---|---|
| Question branch | 64 → 32, ReLU | 2,080 |
| Joined input | 32 + 22 = 54 → 64, ReLU, dropout 0.1 | 3,520 |
| Hidden layer | 64 → 64, ReLU | 4,160 |
| Action head | 64 → 7 | 455 |
| Auxiliary head: P(current answer correct) | 64 → 1 | 65 |
| State-value head | 64 → 1 | 65 |
| **Total** | | **10,345** |

## 8. Learning algorithm

**PPO**, with illegal actions masked: clipped ratio (0.2) + 0.5·value loss − entropy bonus; GAE γ = 1, λ_GAE = 0.95;
learning rate 3e-4 with linear decay; entropy 0.01 → 0.001; rollouts of 2,048 steps; 4 epochs; minibatch 256; gradient clip 0.5;
1M steps. The auxiliary head adds 0.1·BCE.

**Seeds:** 5 per λ (0, 1, 2, 3, 4). A seed sets the starting weights, the sampled actions, the minibatch order, and the
questions and traces drawn in replay. Selection uses validation utility, and the median seed is reported.

## 9. Evaluation

| Item | Value |
|---|---|
| Computation | replay with 200 episodes per question and a fixed evaluation seed |
| Metrics | accuracy, mean tokens, utility at λ, accuracy–cost frontier, chosen plans, learning curves, spread across seeds |
| Baselines | best fixed plan per dataset; rule "RUN_M, ESC_H if cut"; oracle |
| Comparisons | controller vs best fixed plan; controller vs bandit and open loop; **FULL vs API = price of closed access** |
| Statistics | paired bootstrap over questions (10,000 resamples), Holm correction; AMC is descriptive only |

