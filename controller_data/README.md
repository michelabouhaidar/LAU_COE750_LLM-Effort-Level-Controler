# controller_data: the controller's replay data (all datasets)

Recorded with Qwen3.5-2B (frozen), config `t0.6_pp1.5`, 10 traces per question, loop stop on,
16K cap (AMC: 32K). Every dataset is split **by question** (all traces of a question are in one split).

| Files | Split | Questions | Traces | Rows (controller) |
|---|---|---|---|---|
| `gsm8k_train_*` | train | 919 | 9,190 | 91,900 |
| `gsm8k_val_*` | val | 200 | 2,000 | 20,000 |
| `gsm8k_test_*` | test | 200 | 2,000 | 20,000 |
| `math_train_*` | train | 400 | 4,000 | 40,000 |
| `math_val_*` | val | 99 | 990 | 9,900 |
| `math_test_*` | test | 200 | 2,000 | 20,000 |
| `amc23_train_*` | train | 24 | 240 | 2,880 |
| `amc23_val_*` | val | 8 | 80 | 960 |
| `amc23_test_*` | test | 8 | 80 | 960 |

Two files per set, joined on `trace_id = "<question_id>#<sample>"`:

- **`<set>_controller.csv`**: one row per (trace, budget):
  - budget 0 is the `direct` answer (no reasoning), followed by the cuts 256 … 16,384;
  - each row holds the answer at that budget (text, pred, correct, confidences, tokens) and the reasoning features up to it;
  - a cut beyond the trace's end uses its end answer (`from_end = 1`);
  - MATH files add `level` and `subject`.
- **Model and run columns, in both files**:
  - the model: model, model_org, model_family (Qwen3.5), model_size (2B), model_type, model_architecture, model_dtype;
  - its config.json: layers, hidden, intermediate, attention/KV heads, head_dim, full_attention_interval, vocab, max context;
  - model_open_weights, model_reasoning, served_by (vllm);
  - the run behind the row: config, thinking, seed, sampling, weights_hash, vllm_version, server_max_len, b_max, cuts, stop_on_loop;
  - for budget 0 these are the direct run's settings.
- **`<set>_traces.csv`**: one row per trace, with the full `reasoning_text`, plus everything about its generation:
  - question, reference;
  - config, seed;
  - temperature, top_p, top_k, min_p, presence_penalty;
  - model, weights_hash, vllm_version, server_max_len;
  - b_max, cuts, stop_on_loop;
  - prompt_tokens, natural_end, eos_in_reasoning, loop_detected.

Column details are in the docstring of `generation_gsm8k/src/export_controller_csv.py`.

Written by (they generate nothing on the GPU):

```bash
~/.venvs/vllm/bin/python generation_gsm8k/src/export_controller_csv.py
```

```bash
~/.venvs/vllm/bin/python generation_math/src/export_math_csv.py
```

```bash
~/.venvs/vllm/bin/python generation_amc23/src/export_amc23_csv.py
```

## Embeddings (`embeddings/`)

Question, reasoning and answer embeddings from `sentence-transformers/all-MiniLM-L6-v2`: 384 dims, mean pooling,
unit length, float32, at most 256 tokens. PCA to 64 dims is fitted on train only. Both versions are kept.

| File | One row per | Content |
|---|---|---|
| `<set>_rows.npz` | CSV row | `row_index`, `trace_id`, `question_id`, `budget`, `q_index` |
| `<set>_q_ids.npy` | unique question | `question_id` of each row of the `q` arrays |
| `<set>_q_384.npy`, `<set>_q_64.npy` | unique question | the question |
| `<set>_r_384.npy`, `<set>_r_64.npy` | CSV row | the reasoning up to `reasoning_tokens_used`, keeping its end; zero vector at budget 0 |
| `<set>_a_384.npy`, `<set>_a_64.npy` | CSV row | `answer_text`, keeping its start |
| `pca/{q,r,a}_pca64.npz` | fitted PCA | `mean`, `components`, `explained_variance`, `explained_variance_ratio`, `n_fit` |

PCA is fitted separately for q, r and a on the train sets of the three datasets: q on the 1,343 train questions, r and
a on the 121,350 train rows of config `t0.6_pp1.5`. Explained variance with 64 dims: q 65.1%, r 64.4%, a 81.0%.
All four controllers take `<set>_q_64.npy` as `q_emb`.

```bash
~/.venvs/vllm/bin/python controller_data/build_embeddings.py
```
