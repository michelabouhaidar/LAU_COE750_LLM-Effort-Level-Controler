# RL_experiments

A small RL **controller** that decides how much reasoning effort a frozen LLM should spend on each
question. The controller trains on replay of recorded reasoning traces; the LLM
(Qwen3.5-2B, served locally with vLLM) is never trained.

## Documentation

| File | Content |
|---|---|
| `README.md` | this overview |
| [`USER-DOC.md`](USER-DOC.md) | status and every command to run, in order |
| [`VLLM-DOC.md`](VLLM-DOC.md) | the vLLM server: stack, install, `llm` command, settings, troubleshooting |
| [`GENERATION-DOC.md`](GENERATION-DOC.md) | how the data is recorded, per dataset; splits, CSVs, figures, effort levels |

## Folders

```
RL_experiments/
├── generation_gsm8k/   GSM8K recording; also the shared code (client, prompt, configs, plots)
├── generation_math/    MATH and MATH-500 recording
├── generation_amc23/   AMC 2023 recording (runs the MATH scripts on its own data folder)
├── controller_data/    the controller's CSVs (<set>_controller.csv + <set>_traces.csv) and embeddings/
├── figures/            gsm8k_config_exp/, math_pilot/, train_val_test/ (final figures for all datasets come after all generations)
├── rl_methods/         controller designs: bandit, open loop, closed loop (API and FULL)
├── rl_training/        one training notebook per controller family
├── docs/               GSM8K figure notes
└── requirements.txt    pinned package versions
```

## Data

Every dataset: 10 traces per question, `t0.6_pp1.5` plus a no-reasoning `direct` answer, forced answers
at 256 … 16,384 tokens (AMC: 32,768) plus one at the trace's end. Splits are by question.

| Dataset | train | val | test | acc@16K (merged) | direct | L / M / H |
|---|---|---|---|---|---|---|
| GSM8K | 919 | 200 | 200 | 90.3% | 7.3% | 2K / 4K / 16K |
| MATH | 400 | 99 | 200 | 82.9% | 15.2% | 4K / 12K / 16K |
| AMC 2023 | 24 | 8 | 8 | 63.2% (at 32K) | 4.5% | 12K / 24K / 32K |

Question, reasoning and answer embeddings: `all-MiniLM-L6-v2` (384 dims) and PCA to 64 dims fitted on train only, both kept in `controller_data/embeddings/`. All four controllers take the 64-dim PCA question embedding. Next step: build the replay environment and the controllers.
