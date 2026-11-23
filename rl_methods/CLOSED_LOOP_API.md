# Closed-loop controller, question + closed-API measurements, Qwen3.5-2B

**Code:** `rl_training/closed_loop_training.ipynb`. **The LLM is frozen; only the controller is trained.**
**Companion files:** [BANDIT.md](BANDIT.md), [OPEN_LOOP.md](OPEN_LOOP.md), [CLOSED_LOOP_FULL.md](CLOSED_LOOP_FULL.md) (question + every measurement). They share the data, levels, action space, replay, reward and question representation. They differ only in the state.

**What this controller answers:** how well a controller does when it sees **only what a closed
reasoning API returns**. It is trained on Qwen, where everything is cheap to record, but it uses
nothing a closed model cannot give, so it is the one that is applied to a closed model afterwards.

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

**Recorded per question (the same for every dataset; GSM8K command below):** 10 independent traces (`t0.6_pp1.5`, `--stop-on-loop`), each with an
answer at every cut it reached and one at its end. This controller reads only the answer text, how
the run ended, and the token counts.

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

The same as the FULL controller: a pool of the question's 10 trace indices; every run takes an unused
one (so runs are independent, as separate API requests are); `row(q, k, ℓ)` gives the answer, valid,
correct, `finish_reason` and tokens.

**Final answer:** the majority cluster of the valid answers at the current level. Ties are broken at
random with a fixed seed: an API gives no answer probabilities to break them. If none is valid: the
latest valid answer in the history, else wrong.

## 5. State and measurements

**Only what a closed API returns per call:** the visible answer, the `finish_reason`, and the usage
(output tokens, including reasoning tokens). Plus what the controller itself knows: the question, its
own actions and its spend. Constants from train only; response features are 0 at t = 0; `correct`
is never in the state.

| Block | Feature | Computation | From the API | Size |
|---|---|---|---|---|
| question | `q_emb` | MiniLM question embedding (`all-MiniLM-L6-v2`, 384 dims) reduced by PCA fitted on the 1,343 train questions only (`controller_data/embeddings/<set>_q_64.npy`, 65.1% of variance); frozen; a new question uses the stored encoder and PCA | the question we send | 64 |
| | `prompt_len_norm` | min(prompt tokens / P95, 1) | usage: prompt tokens | 1 |
| | `n_numbers_norm` | min(numbers in the question / P95, 1) | the question | 1 |
| position | `level_onehot` | L, M, H | our action | 3 |
| | `step_norm`, `k_norm`, `runs_norm` | t / 5, K / 5, runs / 6 | our actions | 3 |
| | `cost_norm` | min(tokens spent / C_ref, 3) | usage | 1 |
| response | `valid_rate` | share of the K answers with a valid boxed answer | answer text | 1 |
| | `cut_rate` | share stopped by the budget | `finish_reason` | 1 |
| | `len_norm` | mean of tokens / level budget | usage: reasoning + output tokens | 1 |
| agreement | `majority_ratio` | largest valid cluster / K | answer texts | 1 |
| | `cluster_entropy` | −Σ (n_c/K) log(n_c/K) / log K; 0 if K = 1 | answer texts | 1 |
| history | `has_prev` | an answer exists at a lower level | answer texts | 1 |
| | `agrees_prev` | current majority = previous level's majority | answer texts | 1 |
| | `distinct_norm` | distinct valid clusters over all runs / runs used | answer texts | 1 |

**Size:** 64 (PCA) + **17 scalars**.

**Not in this state, unlike FULL** (not available from a closed API): answer confidence
(`conf_mean`, `conf_min`: no token probabilities), `hedge_rate`, `tail_repetition`, `loop_rate`
(the reasoning text is hidden).

## 6. Reward

```
r_t  = −λ · c_t / C_ref          c_t = tokens of the runs the action launches (0 for STOP)
r_T += 1[final answer correct]   at STOP
R    = 1[correct] − λ · C / C_ref ,    γ = 1
```

c_t counts what an API bills: reasoning + output tokens (prompt tokens only with a small weight
ρ = 0.05, if included). C_ref = mean tokens of one H run on the dataset's train split: **GSM8K 6,872, MATH 8,740, AMC 13,004**. λ ∈ {0.02, 0.05, 0.10, 0.25, 0.50}. Best fixed plan at λ = 0.10 (train): GSM8K M×1 = **0.837**,
MATH H×1 = **0.730** (tied with M×1), AMC M×1 = **0.497**.

## 7. Network

| Part | Layers | Parameters |
|---|---|---|
| Question branch | 64 → 32, ReLU | 2,080 |
| Joined input | 32 + 17 = 49 → 64, ReLU, dropout 0.1 | 3,200 |
| Hidden layer | 64 → 64, ReLU | 4,160 |
| Action head | 64 → 7 | 455 |
| Auxiliary head: P(current answer correct) | 64 → 1 | 65 |
| State-value head | 64 → 1 | 65 |
| **Total** | | **10,025** |

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

## 10. Applying it to a closed model

1. **Record a small sweep** on the closed model: on part of validation and test, independent requests
   at each of its effort levels / thinking budgets (enough runs per question to cover an episode: 6).
2. **Map levels** L / M / H onto its settings by measured cost (tokens per run), and recompute
   its constants: level budgets for `len_norm`, its C_ref, its P95s.
3. **Zero-shot:** apply the Qwen-trained controller with those constants.
4. **Recalibrate** with 25 / 50 / 100 of its questions (fine-tune the heads only).
5. **Reference:** a controller trained on the closed model's data alone.

The gap between (3)/(4) and (5) is the transfer result. Two things are expected to differ and must
be reported: the meaning of "cut" (Qwen at a small budget is often cut and still right; a closed
model at its cap is cut rarely) and the base accuracy (a large closed model is right far more often,
so agreement is more common).
