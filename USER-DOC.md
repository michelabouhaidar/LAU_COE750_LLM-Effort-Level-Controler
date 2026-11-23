# USER-DOC: the commands, in order

Run everything from the repo root. Recording commands can be stopped with Ctrl-C and started
again: finished traces are skipped. Server details are in `VLLM-DOC.md`; what each step records is
in `GENERATION-DOC.md`.

```bash
cd ~/Desktop/RL_experiments
```

## Status (2026-11-23)

| Set | Traces | Recorded | CSV exported |
|---|---|---|---|
| GSM8K `config_exp` (sweep) | 400 per config | yes | not needed |
| GSM8K `train` / `val` / `test` | 9,190 / 2,000 / 2,000 | yes | yes |
| MATH `math_pilot` | 200 | yes | not needed |
| MATH `math_train` / `math_val` / `math_test` | 4,000 / 990 / 2,000 | yes | yes |
| **AMC 2023 `amc23`** (one recording, split 24 / 8 / 8 questions at export) | 400 | yes | yes |

## 1. Server

GSM8K and MATH (16K cap):

```bash
LLM_MAX_LEN=17408 LLM_GPU_UTIL=0.90 LLM_MAX_SEQS=64 llm serve
```

AMC 2023 (double context, 32K cap):

```bash
LLM_MAX_LEN=33792 LLM_GPU_UTIL=0.90 LLM_MAX_SEQS=32 llm serve
```

```bash
llm status
```

```bash
llm stop
```

## 2. AMC 2023

Record once: 40 questions x 10 traces = 400 traces, up to 32,768 reasoning tokens:

```bash
~/.venvs/vllm/bin/python generation_amc23/src/run_amc23.py run --set amc23 --configs t0.6_pp1.5 direct --stop-on-loop --concurrency 8
```

Report:

```bash
~/.venvs/vllm/bin/python generation_amc23/src/run_amc23.py report --set amc23 --configs t0.6_pp1.5 direct
```

Export, split by question into `controller_data/amc23_{train,val,test}_*`:

```bash
~/.venvs/vllm/bin/python generation_amc23/src/export_amc23_csv.py
```

## 3. Question files (already written; existing files are never overwritten)

```bash
~/.venvs/vllm/bin/python generation_gsm8k/src/prepare_data.py
```

```bash
~/.venvs/vllm/bin/python generation_math/src/prepare_math.py
```

```bash
~/.venvs/vllm/bin/python generation_amc23/src/prepare_amc23.py
```

## 4. Record GSM8K or MATH again (already done)

GSM8K, with `--set train`, `val` or `test`:

```bash
~/.venvs/vllm/bin/python generation_gsm8k/src/sweep/run_sweep.py run --set train --configs t0.6_pp1.5 direct --samples 10 --cuts 256 512 1024 2048 4096 6144 8192 12288 16384 --stop-on-loop --concurrency 16
```

MATH, with `--set math_train`, `math_val` or `math_test`:

```bash
~/.venvs/vllm/bin/python generation_math/src/run_math.py run --set math_train --configs t0.6_pp1.5 direct --samples 10 --stop-on-loop --concurrency 16
```

## 5. Reports (generate nothing)

```bash
~/.venvs/vllm/bin/python generation_gsm8k/src/sweep/run_sweep.py report --set train --configs t0.6_pp1.5 direct
```

```bash
~/.venvs/vllm/bin/python generation_math/src/run_math.py report --set math_train --configs t0.6_pp1.5 direct
```

## 6. Controller CSVs (generate nothing)

```bash
~/.venvs/vllm/bin/python generation_gsm8k/src/export_controller_csv.py
```

```bash
~/.venvs/vllm/bin/python generation_math/src/export_math_csv.py
```

## 7. Embeddings (generate nothing)

Question, reasoning and answer embeddings from `all-MiniLM-L6-v2` (384 dims), with PCA to 64 dims fitted on train only. Both are kept in `controller_data/embeddings/`. All four controllers take the 64-dim PCA question embedding.

```bash
~/.venvs/vllm/bin/python controller_data/build_embeddings.py
```

## 8. Figures

The final figures for all datasets are drawn once, after every generation is finished.
