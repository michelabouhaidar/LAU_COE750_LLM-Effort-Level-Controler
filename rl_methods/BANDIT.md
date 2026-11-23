# Contextual bandit controller (question only, one choice), Qwen3.5-2B

**Code:** `rl_training/bandit_training.ipynb`. **The LLM is frozen; only the controller is trained.**
**Companion files:** [OPEN_LOOP.md](OPEN_LOOP.md) (question only, PPO),
[CLOSED_LOOP_FULL.md](CLOSED_LOOP_FULL.md), [CLOSED_LOOP_API.md](CLOSED_LOOP_API.md).

**What this controller answers:** how much can be saved by reading the question and making **one
choice** among fixed plans? It is the simple reference for open-loop PPO: both use the
same information, so if PPO falls well below it, its training is the problem.

---

## 1. Data: what is input and what is training signal

| Dataset | train | val | test | Replay files (`controller_data/`) |
|---|---|---|---|---|
| GSM8K | 919 | 200 | 200 | `gsm8k_{train,val,test}_*` |
| MATH | 400 | 99 | 200 | `math_{train,val,test}_*` |
| AMC 2023 | 24 | 8 | 8 | `amc23_{train,val,test}_*` |
| **total** | **1,343** | **307** | **408** | 10 traces per question |

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

| Data | Columns | Used for | Seen when choosing a plan |
|---|---|---|---|
| question | `question`, `prompt_tokens`, `n_numbers` | input | **yes** |
| answers | `pred`, `valid`, `correct` | majority vote, accuracy target | **no** |
| tokens | `total_tokens` | cost target | **no** |
| indexing | `question_id`, `sample`, `budget` | finding a trace at a level | no |

At run time the controller receives the question only. Answers and tokens from the recordings
teach it what each plan would have achieved.

## 2. Levels (measured on train, one run)

| Level | GSM8K budget | Accuracy | Tokens | MATH budget | Accuracy | Tokens |
|---|---|---|---|---|---|---|
| L | 2,048 | 84.5% | 1,856 | 4,096 | 73.5% | 3,616 |
| M | 4,096 | 88.5% | 3,290 | 12,288 | 81.7% | 7,583 |
| H | 16,384 | 90.3% | 6,872 | 16,384 | 83.0% | 8,740 |

Levels follow the per-dataset rule (GENERATION-DOC section 6).

| AMC level (32K cap) | Budget | Train, one run: accuracy | Tokens |
|---|---|---|---|
| L | 12,288 | 54.2% | 9,089 |
| M | 24,576 | 59.2% | 12,314 |
| H | 32,768 | 59.6% | 13,004 |

A run at level ℓ = one trace read at cut ℓ: its forced answer if it reached ℓ, else its end answer.

## 3. Actions: 9 plans

The bandit makes one choice and then the plan runs to the end.

| Plan | Sequence in the shared action space | Runs |
|---|---|---|
| L×1, L×3, L×5 | RUN_L, (ADD …), STOP | 1, 3, 5 |
| M×1, M×3, M×5 | RUN_M, (ADD …), STOP | 1, 3, 5 |
| H×1, H×3, H×5 | RUN_H, (ADD …), STOP | 1, 3, 5 |

All 9 are legal under the shared limits (K ≤ 5, ≤ 6 runs, ≤ 6 decisions, STOP included).

## 4. Exact plan values

For plan (ℓ, K), average over all C(10, K) sets of traces:

$$
\mathrm{acc}(q,\ell,K)=\frac{1}{\binom{10}{K}}\sum_{D}\mathrm{credit}(D)
$$

$$
\mathrm{credit}(D)=
\begin{cases}
\dfrac{1}{m} & \text{if the correct answer is among the } m \text{ tied largest valid clusters}\\[6pt]
0 & \text{otherwise}
\end{cases}
$$

$$
\mathrm{cost}(q,\ell,K)=K\cdot\frac{1}{10}\sum_{i=1}^{10}\mathrm{tokens}_i(q,\ell)
$$

| Symbol | Meaning |
|---|---|
| $q$ | the question |
| $\ell$ | the level: L, M or H |
| $K$ | the number of runs in the plan |
| $D$ | one set of $K$ traces out of the question's 10 |
| $m$ | the number of clusters tied for largest in $D$ |

No valid answer in the draw = wrong. The credit $1/m$ is the exact expectation of a uniform random
tie-break. If a winning cluster holds both right and wrong answers, it counts for the fraction of its
answers in the draw that are right.

**Example:** at level L, 6 of the 10 answers are right and the 4 wrong ones all differ. For K = 3:

| Draw | Sets | Credit |
|---|---|---|
| at least 2 right | 80 | 1 |
| 1 right, 2 different wrong | 36 | 1/3 |
| 3 wrong | 4 | 0 |

acc = (80 + 12) / 120 = **0.767** (K = 1: 0.600).

## 5. State

| Feature | Size | Computation |
|---|---|---|
| `q_emb` | 64 | MiniLM question embedding (`all-MiniLM-L6-v2`, 384 dims) reduced by PCA fitted on the 1,343 train questions only (`controller_data/embeddings/<set>_q_64.npy`, 65.1% of variance); frozen (same input for all controllers) |
| `prompt_len_norm` | 1 | min(prompt tokens / P95 on train, 1) |
| `n_numbers_norm` | 1 | min(numbers in the question / P95 on train, 1) |

## 6. Reward

**Sampled reward** of one run of a plan $p$ (used by BD-2):

$$
R(q,p)=\mathbb{1}[\text{final answer correct}]-\lambda\,\frac{\mathrm{cost}}{C_{\mathrm{ref}}}
$$

**Exact utility** of the plan (used by BD-1 and for evaluation):

$$
U(q,p;\lambda)=\mathrm{acc}(q,p)-\lambda\,\frac{\mathrm{cost}(q,p)}{C_{\mathrm{ref}}}
$$

C_ref = mean tokens of one H run on the dataset's train split: **GSM8K 6,872, MATH 8,740, AMC 13,004**. λ ∈ {0.02, 0.05, 0.10, 0.25, 0.50}.

**Best fixed plan per dataset (train, exact over the 9 plans):**

| λ | GSM8K | Utility | MATH | Utility | AMC | Utility |
|---|---|---|---|---|---|---|
| 0.02 | H×1 | 0.883 | H×1 | 0.810 | H×1 | 0.576 |
| 0.05 | M×1 | 0.861 | H×1 | 0.780 | H×1 | 0.546 |
| 0.10 | M×1 | 0.837 | H×1 and M×1 (tie) | 0.730 | M×1 | 0.497 |
| 0.25 | L×1 | 0.778 | L×1 | 0.632 | L×1 | 0.367 |
| 0.50 | L×1 | 0.710 | L×1 | 0.528 | L×1 | 0.192 |

## 7. Network

| Step | Layer | Input → output | Parameters |
|---|---|---|---|
| 1 | Question branch: Linear, ReLU | `q_emb` 64 → 32 | 2,080 |
| 2 | Concatenate with the 2 scalars | 32 + 2 → 34 | 0 |
| 3 | Linear, ReLU, Dropout 0.1 | 34 → 64 | 2,240 |
| 4 | Linear, ReLU | 64 → 64 | 4,160 |
| 5a | Accuracy head: Linear, sigmoid | 64 → 9 (accuracy of each plan) | 585 |
| 5b | Cost head: Linear | 64 → 9 (cost / C_ref of each plan) | 585 |
| | **Total** | | **9,650** |

## 8. Learning algorithms

| ID | Method | Loss / update | Settings |
|---|---|---|---|
| BD-1 | **Supervised router** (main) | BCE(acc_pred, acc) + MSE(cost_pred, cost/C_ref), 9 plans, exact targets. Plan = argmax_p [acc_pred − λ·cost_pred], so one model serves every λ | AdamW lr 1e-3, weight decay 1e-4, batch 256, early stopping on validation utility |
| BD-2 | **Policy-gradient bandit** | softmax over 9 plans, sampled plan p, sampled reward R; loss = −(R − b(q))·log π(p\|q) − 0.01·entropy, b(q) = learned value | Adam lr 3e-4, 200K sampled pairs, one model per λ |

5 seeds each. BD-2 shows whether a sampled update reaches the exact-target result.

## 9. Evaluation

| Item | Value |
|---|---|
| Metrics | accuracy, mean tokens, utility at each λ, accuracy–cost frontier, plan frequencies |
| Computation | exact (section 4) per validation / test question |
| Selection | on validation; test once |
| Baselines | every fixed plan; the best fixed plan overall; **the best fixed plan per dataset**; per-question oracle over the 9 plans |
| Comparison | bandit − best fixed plan **per dataset** = **value of reading the question** beyond recognising its dataset |
| Test | paired bootstrap over questions (10,000 resamples) |
| Transfer to a closed model | direct: it uses no LLM output. Only the level mapping and C_ref change |

## 10. Limit of this reference

The 9 plans contain no escalation. Under the shared final-answer rule, an escalating sequence can
fall back on an earlier valid answer when its last level has none, so the bandit is not strictly the
best open-loop policy. The gap should be small: on GSM8K and MATH train, 99.3–99.9% of answers are valid.
