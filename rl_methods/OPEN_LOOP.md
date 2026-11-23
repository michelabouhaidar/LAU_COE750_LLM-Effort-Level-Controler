# Open-loop controller (question only), PPO, Qwen3.5-2B

**Code:** `rl_training/open_loop_training.ipynb`. **The LLM is frozen; only the controller is trained.**
**Companion files:** [BANDIT.md](BANDIT.md) (question only, one choice among 9 plans),
[CLOSED_LOOP_FULL.md](CLOSED_LOOP_FULL.md) (question + every measurement),
[CLOSED_LOOP_API.md](CLOSED_LOOP_API.md) (question + closed-API measurements).

**What this controller answers:** how much can be saved by choosing the effort **from the question
alone**, with the same sequential actions and the same learner as the closed loops? Because only the
state differs, **closed loop − open loop = value of observing answers**.

---

## 1. Data: what is input and what is training signal

| Dataset | train | val | test | Replay files (`controller_data/`) |
|---|---|---|---|---|
| GSM8K | 919 | 200 | 200 | `gsm8k_{train,val,test}_*` |
| MATH | 400 | 99 | 200 | `math_{train,val,test}_*` |
| AMC 2023 | 24 | 8 | 8 | `amc23_{train,val,test}_*` |
| **total** | **1,343** | **307** | **408** | 10 traces per question |

No question is in two splits. `config_exp` is not used. Traces use `t0.6_pp1.5`, `--stop-on-loop`.

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

| Data | Columns | Used for | In the state |
|---|---|---|---|
| question | `question`, `prompt_tokens`, `n_numbers` | input | **yes** |
| answers | `pred`, `valid`, `correct` | final answer, terminal reward | **no** |
| tokens | `total_tokens` | cost reward | **no** |
| indexing | `question_id`, `sample`, `budget` | finding a trace at a level | no |

**Rule:** the replay uses answers and tokens to compute rewards during training. The network never
receives them as input. If it did, it would be a closed-loop controller.

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

## 3. Action space (the same in all closed-loop files)

A **decision** is one action chosen by the controller, STOP included. `RUN_M → ADD → STOP` is 3
decisions and 2 runs.

| t | Action | Effect |
|---|---|---|
| 0 | RUN_L, RUN_M, RUN_H | 1 run at that level |
| ≥ 1 | STOP | return the final answer |
| ≥ 1 | ADD | 1 more run at the current level |
| ≥ 1 | ESC_M, ESC_H | current answers → history; 1 new run at a higher level |

**Mask:** ESC only upward; ADD only if K < 5; ADD and ESC only if runs < 6; at the 6th decision
(t = 5) only STOP. **No CONTINUE.**

These limits allow **55 legal action sequences**. With no observation, a deterministic policy picks
one of them per question.

## 4. Replay environment

```
reset(q, seed):   unused = {0..9};  level = None;  K = 0;  counts = [0,0,0];  t = 0
run(ℓ):           k = draw from unused (seeded);  unused −= {k};  return row(q, k, ℓ)
step(a):          apply the action, charge its tokens, update the position state; done if STOP
```

The answers and token counts of `row` stay inside the environment. `step` returns only the next
position state and the reward.

**Final answer:** the majority cluster of the valid answers at the current level. Ties are broken at
random with a fixed seed. If none is valid: the latest valid answer in the history, else wrong. This
is the closed-API rule, so the comparison with that controller is like for like.

**Exact value of a sequence** (for evaluation), with n_ℓ runs at level ℓ:

```
acc(q, seq)  = mean over all disjoint draws of credit(draw)
credit(draw) = 1/m if the correct answer is among the m tied largest clusters, else 0
cost(q, seq) = Σ_ℓ n_ℓ · mean tokens of the question's 10 traces at ℓ
```

A sequence has at most 25,200 draws.

## 5. State

| Block | Feature | Computation | Size |
|---|---|---|---|
| question | `q_emb` | MiniLM question embedding (`all-MiniLM-L6-v2`, 384 dims) reduced by PCA fitted on the 1,343 train questions only (`controller_data/embeddings/<set>_q_64.npy`, 65.1% of variance); frozen (same input for all controllers) | 64 |
| | `prompt_len_norm` | min(prompt tokens / P95 on train, 1) | 1 |
| | `n_numbers_norm` | min(numbers in the question / P95 on train, 1) | 1 |
| position | `level_onehot` | L, M, H (zeros at t = 0) | 3 |
| | `step_norm`, `k_norm`, `runs_norm` | t / 5, K / 5, runs used / 6 | 3 |
| own history | `count_L`, `count_M`, `count_H` | runs launched at each level / 5 | 3 |

**Size:** 64 (PCA) + **11 scalars**.

`count_*` are the controller's own past actions. They are needed because the fallback makes the value
of a state depend on which lower-level runs exist. `cost_norm` is **not used**: tokens actually spent
are feedback from the LLM.

## 6. Reward

```
r_t  = −λ · c_t / C_ref          c_t = tokens of the runs the action launches (0 for STOP)
r_T += 1[final answer correct]   at STOP
R    = 1[correct] − λ · C / C_ref ,    γ = 1
```

C_ref = mean tokens of one H run on the dataset's train split: **GSM8K 6,872, MATH 8,740, AMC 13,004**. λ ∈ {0.02, 0.05, 0.10, 0.25, 0.50}.

**Example, λ = 0.10, train averages:**

| Sequence | GSM8K accuracy | Cost / C_ref | Utility | MATH accuracy | Cost / C_ref | Utility |
|---|---|---|---|---|---|---|
| RUN_L, STOP | 0.845 | 0.270 | 0.818 | 0.735 | 0.414 | 0.694 |
| RUN_M, STOP | 0.885 | 0.479 | **0.837** | 0.817 | 0.868 | **0.730** |
| RUN_H, STOP | 0.903 | 1.000 | 0.803 | 0.830 | 1.000 | **0.730** |
| RUN_L, ESC_M, STOP | about 0.885 | 0.749 | about 0.810 | about 0.817 | 1.281 | about 0.689 |

The best fixed sequence differs by dataset: M on GSM8K, M and H tied on MATH, M on AMC (0.497). The last row pays for L and still ends on one M answer.
Without feedback, escalation can only gain through the rare fallback.

## 7. Network

```
q_emb (64)  ─ Linear 32 ─ ReLU ─┐
scalars (11) ───────────────────┴─ concat 43 ─ Linear 64 ─ ReLU ─ Dropout 0.1 ─ Linear 64 ─ ReLU ─┬─ Linear 7 → action logits
                                                                                                  ├─ Linear 1 → state value
                                                                                                  └─ Linear 1 ─ sigmoid → P(current answer correct), auxiliary
```

| Layer | Parameters |
|---|---|
| question branch 64→32 | 2,080 |
| 43→64 | 2,816 |
| 64→64 | 4,160 |
| action head 64→7 | 455 |
| auxiliary head | 65 |
| value head | 65 |
| **total** | **9,641** |

The auxiliary target is 1[answer returned by STOP now is correct], for t ≥ 1 only.

## 8. Learning algorithm

The settings match the closed-loop PPO.

| ID | Method | Targets / loss | Settings |
|---|---|---|---|
| **OL-PPO** | **PPO**, masked | ρ = π_θ/π_old; L = −min(ρÂ, clip(ρ, 0.8, 1.2)Â) + 0.5·(V − R̂)² − β_H·entropy + 0.1·BCE(auxiliary); Â by GAE (γ = 1, λ_GAE = 0.95) | lr 3e-4 linear decay, β_H 0.01 → 0.001, rollouts 2,048 steps, 4 epochs, minibatch 256, grad-norm clip 0.5, 1M steps |

One model per λ, 5 seeds. Evaluation uses the greedy valid action. Checkpoints are selected
by exact validation utility, and the median seed goes to test.

**Expected behaviour:** PPO should learn a short fixed sequence per question and approach the bandit.
They have no answers to react to, so a clear win over the bandit is not expected.

## 9. Evaluation

| Item | Value |
|---|---|
| Metrics | accuracy, mean tokens, utility at each λ, accuracy–cost frontier, sequence frequencies, learning curves, spread across seeds |
| Computation | exact (section 4) per validation / test question |
| Selection | on validation; test once |
| Baselines | every fixed sequence; the best fixed sequence overall; **the best fixed sequence per dataset**; [BANDIT.md](BANDIT.md); per-question oracle over the 55 sequences |
| Comparisons | open loop − best fixed sequence **per dataset** = **value of reading the question** beyond recognising its dataset; PPO − bandit = **training check**; closed loop − open loop = **value of observing** |
| Test | paired bootstrap over questions (10,000 resamples), Holm correction |
| Transfer to a closed model | direct: it uses no LLM output. Only the level mapping and C_ref change |

## 10. Expected difficulty

The embedding can reveal which dataset a question comes from, so part of any gain may only be "use the
best plan for this dataset". The per-dataset baseline separates that from real per-question routing.
Inside GSM8K, questions look alike, so the policy may end up "always RUN_M, STOP". That would be a result
in itself, and it makes the closed loops more important. MATH and AMC should carry more signal, but AMC
has only 24 training questions.
