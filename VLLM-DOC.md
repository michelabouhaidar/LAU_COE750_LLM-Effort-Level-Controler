# VLLM-DOC: the local vLLM server

The model server that every recording uses. Technical reference only; how the data is generated
is in `GENERATION-DOC.md`, the commands to run in `USER-DOC.md`.

Contents
1. [Stack](#1-stack)
2. [Files and locations](#2-files-and-locations)
3. [Install from scratch](#3-install-from-scratch)
4. [The `llm` command](#4-the-llm-command)
5. [Server settings for the experiments](#5-server-settings-for-the-experiments)
6. [Client access (API)](#6-client-access-api)
7. [Without `llm`](#7-without-llm)
8. [Troubleshooting](#8-troubleshooting)
9. [Uninstall](#9-uninstall)
10. [Full source of `llm` and `chat.py`](#10-full-source-of-llm-and-chatpy)

---

## 1. Stack

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop, 8 GB (Blackwell, sm_120), driver 580.178.04 |
| OS | Ubuntu (Linux), no system CUDA toolkit |
| vLLM | 0.33.0, torch 2.14.0+cu130 (CUDA 13.0 wheels), Python 3.12 |
| Model | `Qwen/Qwen3.5-2B` (bf16, ~4.5 GB; weights hash `15852e8c16360a2fea060d615a32b45270f8a8fc`): hybrid attention (full attention every 4th of 24 layers, linear attention otherwise), max context 262,144, thinking by default (`<think>…</think>`) |
| Endpoint | OpenAI-compatible API at `http://127.0.0.1:8000/v1` |
| Experiment env | `~/.venvs/vllm` (Python 3.12): vllm 0.33.0, torch 2.14.0, transformers 5.20.2, datasets 5.2.0, openai 3.50.0, math-verify 0.9.0, matplotlib 3.11.2, tqdm 4.70.1 |

Models of 4B parameters or more (Qwen3.5-4B, R1-Distill-7B, …) do not fit in 8 GB at bf16.

---

## 2. Files and locations

Nothing machine-wide lives in the repo.

| Item | Location |
|---|---|
| `uv` | `~/.local/bin/uv` |
| vLLM (global tool, used by `llm serve`) | `~/.local/share/uv/tools/vllm/` → `~/.local/bin/vllm` |
| CUDA toolkit used for kernel compiles | `~/.local/share/uv/tools/vllm/lib/python3.12/site-packages/nvidia/cu13` |
| Hugging Face CLI | `~/.local/share/uv/tools/huggingface-hub/` → `~/.local/bin/hf` |
| Open WebUI | `~/.local/share/uv/tools/open-webui/` → `~/.local/bin/open-webui` |
| `llm` command | `~/.local/bin/llm` (section 10) |
| Terminal chat client | `~/.local/share/llm/chat.py` (section 10) |
| Model weights | `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/` (links into the shared `blobs/`) |
| Logs, PIDs, served model name | `~/.local/state/llm/` (`vllm.log`, `vllm.pid`, `ui.log`, `ui.pid`, `model`) |
| Web UI data and secret key | `~/.local/share/llm/open-webui/` (`.webui_secret_key`, mode 600) |
| Experiment env | `~/.venvs/vllm` |
| Pinned versions | `requirements.txt` (repo root) |

---

## 3. Install from scratch

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
```

```bash
uv tool install --python 3.12 "vllm==0.33.0"
uv tool install --python 3.12 huggingface_hub
uv tool install --python 3.12 open-webui
```

```bash
hf download Qwen/Qwen3.5-2B
```

Write `~/.local/bin/llm` and `~/.local/share/llm/chat.py` from section 10, then:

```bash
chmod +x ~/.local/bin/llm
```

Experiment env:

```bash
uv venv --python 3.12 ~/.venvs/vllm
uv pip install --python ~/.venvs/vllm/bin/python -r ~/Desktop/RL_experiments/requirements.txt
uv pip install --python ~/.venvs/vllm/bin/python datasets math-verify matplotlib
```

`~/.local/bin` must be on `PATH` (the uv installer adds `export PATH="$HOME/.local/bin:$PATH"` to
`~/.bashrc`).

---

## 4. The `llm` command

A bash script (`~/.local/bin/llm`) that starts and stops vLLM and Open WebUI in the background
(`setsid nohup`, PID in `~/.local/state/llm/`, stopped by process group).

| Command | Action |
|---|---|
| `llm serve [model]` | start vLLM (default `2b` = `Qwen/Qwen3.5-2B`); waits up to 10 min for `/v1/models`; refuses while the download is incomplete |
| `llm stop` | stop vLLM (SIGTERM to the group, SIGKILL after 15 s) |
| `llm status` | server / UI state and GPU memory |
| `llm logs [ui]` | follow `vllm.log` (or `ui.log`) |
| `llm ui` / `llm ui-stop` | Open WebUI on `http://localhost:8080` (auth off, Ollama off) |
| `llm chat` / `llm ask "…"` | terminal client (`chat.py`) |
| `llm models` | model shortcuts |

**Environment variables** (read by `llm serve`):

| Variable | vLLM flag | Default |
|---|---|---|
| `LLM_MAX_LEN` | `--max-model-len` | 16384 |
| `LLM_GPU_UTIL` | `--gpu-memory-utilization` | 0.85 |
| `LLM_MAX_SEQS` | `--max-num-seqs` | 256 |
| `LLM_KV_DTYPE` | `--kv-cache-dtype` | auto |
| `LLM_PORT` | `--port` | 8000 |
| `LLM_UI_PORT` | Open WebUI port | 8080 |

**Fixed flags:** `--host 127.0.0.1 --enable-prefix-caching --reasoning-parser qwen3
--limit-mm-per-prompt '{"image":0,"video":0}' --seed 0`.

**CUDA environment set by the script** (vLLM compiles some kernels on first use; there is no
system CUDA):

| Setting | Why |
|---|---|
| `CUDA_HOME=…/site-packages/nvidia/cu13` | nvcc from the pip CUDA toolkit (otherwise: `Could not find nvcc`) |
| `PATH` += `$CUDA_HOME/bin`, `~/.local/share/uv/tools/vllm/bin` | nvcc and ninja (otherwise: `No such file or directory: 'ninja'`) |
| `VLLM_USE_FLASHINFER_SAMPLER=0` | the pip toolkit has nvcc 13.5 with 13.0 headers, so FlashInfer's sampler cannot compile (`CUDA compiler and CUDA toolkit headers are incompatible`); vLLM's PyTorch sampler is used instead |

`llm serve` always runs the global vLLM binary, even with a venv active.

---

## 5. Server settings for the experiments

| Use | Command | Result |
|---|---|---|
| All recordings (16K reasoning cap) | `LLM_MAX_LEN=17408 LLM_GPU_UTIL=0.90 LLM_MAX_SEQS=64 llm serve` | KV cache 179,302 tokens, 10.3 requests of 17,408 at once |
| AMC 2023 (32K reasoning cap) | `LLM_MAX_LEN=33792 LLM_GPU_UTIL=0.90 LLM_MAX_SEQS=32 llm serve` | |
| Long-context pilot (65K) | `LLM_MAX_LEN=67584 LLM_GPU_UTIL=0.90 LLM_MAX_SEQS=64 llm serve` | KV cache 154,857 tokens, 2.29 requests at full length |

- **Context = cap + prompt + answer.** 17,408 = 16,384 reasoning + prompt (≤ ~1,000) + forced
  answer cue and 16–32 answer tokens. The recording scripts check the server's `max_model_len`
  and refuse to start if it is too small.
- **`LLM_MAX_SEQS=64`**: with the default 256 the start fails with `max_num_seqs (256) exceeds
  available Mamba cache blocks (236)` (the linear-attention layers keep a state per sequence).
- **`LLM_KV_DTYPE=fp8` does not work**: it needs a FlashInfer kernel compiled on the fly, which
  fails with the header mismatch above. The default (bf16) cache is used.
- **Determinism**: the seed is fixed per request (`seed_for(question_id, sample)`), but vLLM's
  batching changes the numerics, so the same seed can give a different trace in another run.
- Client concurrency 16 (`--concurrency 16`) for 17,408; 8 for 33,792.

---

## 6. Client access (API)

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
r = client.chat.completions.create(
    model="Qwen/Qwen3.5-2B",
    messages=[{"role": "user", "content": "Solve 2x+3=11"}],
    temperature=0.6, top_p=0.95, presence_penalty=1.5,
    extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}},   # False = no thinking
)
print(r.choices[0].message.reasoning_content)   # the <think> part (reasoning parser qwen3)
print(r.choices[0].message.content)             # the answer
```

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "Qwen/Qwen3.5-2B", "messages": [{"role": "user", "content": "What is 17*23?"}]}'
```

The recording scripts use the **completions** endpoint with token ids (prompt built with the chat
template, `</think>` and the answer cue appended as tokens), streaming, `logprobs` on the answer
tokens; see `generation_gsm8k/src/common.py`.

Model-card sampling: thinking `temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5`;
non-thinking `0.7, 0.8, 20, 1.5`. The recordings use `0.6, 0.95, 20, 1.5` (chosen in the GSM8K
config sweep).

---

## 7. Without `llm`

```bash
export CUDA_HOME=~/.local/share/uv/tools/vllm/lib/python3.12/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$HOME/.local/share/uv/tools/vllm/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
vllm serve Qwen/Qwen3.5-2B --host 127.0.0.1 --port 8000 --max-model-len 17408 \
  --gpu-memory-utilization 0.90 --max-num-seqs 64 --enable-prefix-caching --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"image":0,"video":0}' --seed 0
```

Ready when the log shows `Application startup complete`. Web UI by hand:

```bash
DATA_DIR=~/.local/share/llm/open-webui WEBUI_AUTH=False ENABLE_OLLAMA_API=False OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=none open-webui serve --host 127.0.0.1 --port 8080
```

---

## 8. Troubleshooting

| Symptom | Cause / solution |
|---|---|
| `llm: command not found` | `~/.local/bin` not on `PATH`: open a new terminal or `export PATH="$HOME/.local/bin:$PATH"` |
| `nvcc` / `ninja` / headers-incompatible errors | `vllm serve` run without the CUDA environment: use `llm serve` or section 7 |
| `max_num_seqs … exceeds available Mamba cache blocks` | `LLM_MAX_SEQS=64` |
| CUDA out of memory at start | another process holds the GPU (`nvidia-smi`); or lower `LLM_MAX_LEN` / `LLM_GPU_UTIL` |
| `Server context … is too small for --b-max` (recording scripts) | restart with `LLM_MAX_LEN` ≥ b_max + 1,024 |
| Server crashed or hangs | `llm logs`, or `tail -50 ~/.local/state/llm/vllm.log` |
| GPU still used after `llm stop` | `nvidia-smi` → `kill -9 <pid>` |
| "still downloading" | `ls ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/blobs \| grep -c incomplete` must be 0 |
| Web UI shows no model | `llm serve`, then `llm ui-stop && llm ui` |

Do not upgrade vLLM during an experiment: the version is recorded in every trace (`vllm_version`).

---

## 9. Uninstall

```bash
llm stop; llm ui-stop
rm ~/.local/bin/llm
rm -r ~/.local/share/llm ~/.local/state/llm
uv tool uninstall vllm huggingface-hub open-webui
```

---

## 10. Full source of `llm` and `chat.py`

### `~/.local/bin/llm`

```bash
#!/usr/bin/env bash
# llm — run local LLMs with vLLM from anywhere on this computer.
#   llm serve [model]   start the model server (default: 2b) on http://localhost:8000
#   llm stop            stop the model server
#   llm ui              start the web interface on http://localhost:8080 (and open it)
#   llm ui-stop         stop the web interface
#   llm chat            chat in the terminal
#   llm ask "question"  ask one question and print the answer
#   llm status          show what is running
#   llm logs [ui]       follow the server (or UI) log
#   llm models          list model shortcuts
set -euo pipefail

STATE="$HOME/.local/state/llm"
SHARE="$HOME/.local/share/llm"
mkdir -p "$STATE" "$SHARE"
PORT="${LLM_PORT:-8000}"
UI_PORT="${LLM_UI_PORT:-8080}"
VLLM_PY="$HOME/.local/share/uv/tools/vllm/bin/python"
VLLM_BIN="$HOME/.local/share/uv/tools/vllm/bin/vllm"   # always the global vLLM, even if a venv is active
# vLLM compiles some GPU kernels on first use and needs nvcc; use the CUDA toolkit that came with it
export CUDA_HOME="$HOME/.local/share/uv/tools/vllm/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$HOME/.local/share/uv/tools/vllm/bin:$PATH"   # nvcc + ninja
# FlashInfer's sampler must be compiled on first use, and the pip CUDA packages don't match
# (nvcc 13.5 vs 13.0 headers) so that compile fails. Use vLLM's built-in PyTorch sampler instead.
export VLLM_USE_FLASHINFER_SAMPLER=0

resolve_model() {
  case "${1:-2b}" in
    2b) echo "Qwen/Qwen3.5-2B" ;;
    *)  echo "$1" ;;   # any Hugging Face id or local path
  esac
}

running() { [[ -f "$STATE/$1.pid" ]] && kill -0 "$(cat "$STATE/$1.pid")" 2>/dev/null; }

stop_proc() {
  if running "$1"; then
    local pid; pid="$(cat "$STATE/$1.pid")"
    # processes are started with setsid, so pid == process-group id: stop the whole group
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
    for _ in $(seq 1 15); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$pid" 2>/dev/null || true
    rm -f "$STATE/$1.pid"; echo "Stopped $1."
  else
    rm -f "$STATE/$1.pid"; echo "$1 is not running."
  fi
}

case "${1:-help}" in
  serve)
    if running vllm; then echo "Already serving $(cat "$STATE/model") on port $PORT (run 'llm stop' first)."; exit 0; fi
    MODEL="$(resolve_model "${2:-2b}")"
    CACHE_DIR="$HOME/.cache/huggingface/hub/models--${MODEL//\//--}"
    if [[ ! -d "$MODEL" ]]; then
      if [[ ! -d "$CACHE_DIR" ]]; then echo "$MODEL is not downloaded. Run: hf download $MODEL"; exit 1; fi
      if compgen -G "$CACHE_DIR/blobs/*.incomplete" >/dev/null; then
        echo "$MODEL is still downloading ($(du -sh "$CACHE_DIR" | cut -f1) so far). Wait for 'hf download' to finish."; exit 1
      fi
    fi
    echo "$MODEL" > "$STATE/model"
    setsid nohup "$VLLM_BIN" serve "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --max-model-len "${LLM_MAX_LEN:-16384}" \
      --gpu-memory-utilization "${LLM_GPU_UTIL:-0.85}" \
      --kv-cache-dtype "${LLM_KV_DTYPE:-auto}" \
      --max-num-seqs "${LLM_MAX_SEQS:-256}" \
      --enable-prefix-caching \
      --reasoning-parser qwen3 \
      --limit-mm-per-prompt '{"image":0,"video":0}' \
      --seed 0 > "$STATE/vllm.log" 2>&1 &
    echo $! > "$STATE/vllm.pid"
    echo -n "Loading $MODEL "
    for _ in $(seq 1 300); do
      if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then echo; echo "Ready: http://localhost:$PORT/v1"; exit 0; fi
      if ! running vllm; then echo; echo "Server crashed. Last log lines:"; tail -20 "$STATE/vllm.log"; exit 1; fi
      echo -n "."; sleep 2
    done
    echo; echo "Still loading after 10 min — check 'llm logs'." ;;
  stop) stop_proc vllm ;;
  ui)
    command -v open-webui >/dev/null || { echo "Open WebUI isn't installed yet. Install it with: uv tool install --python 3.12 open-webui"; exit 1; }
    if ! running ui; then
      # Open WebUI reads (or creates) .webui_secret_key in the folder it is started from,
      # so start it inside its own data folder, never in whatever folder you are in.
      mkdir -p "$SHARE/open-webui"
      ( cd "$SHARE/open-webui"
        DATA_DIR="$SHARE/open-webui" WEBUI_AUTH=False ENABLE_OLLAMA_API=False \
        OPENAI_API_BASE_URL="http://127.0.0.1:$PORT/v1" OPENAI_API_KEY=none \
        setsid nohup open-webui serve --host 127.0.0.1 --port "$UI_PORT" > "$STATE/ui.log" 2>&1 &
        echo $! > "$STATE/ui.pid" )
      echo -n "Starting web UI "
      for _ in $(seq 1 120); do
        curl -sf "http://127.0.0.1:$UI_PORT/health" >/dev/null && break
        running ui || { echo; echo "UI crashed:"; tail -20 "$STATE/ui.log"; exit 1; }
        echo -n "."; sleep 2
      done; echo
    fi
    echo "Web UI: http://localhost:$UI_PORT"
    running vllm || echo "Note: no model is being served yet — run 'llm serve'."
    xdg-open "http://localhost:$UI_PORT" >/dev/null 2>&1 || true ;;
  ui-stop) stop_proc ui ;;
  chat) exec "$VLLM_PY" "$SHARE/chat.py" --port "$PORT" ;;
  ask)  shift; exec "$VLLM_PY" "$SHARE/chat.py" --port "$PORT" --ask "$*" ;;
  status)
    if running vllm; then echo "Model server: running ($(cat "$STATE/model")) on http://localhost:$PORT/v1"; else echo "Model server: stopped"; fi
    if running ui; then echo "Web UI:       running on http://localhost:$UI_PORT"; else echo "Web UI:       stopped"; fi
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/GPU memory:   /' ;;
  logs) tail -f "$STATE/${2:-vllm}.log" ;;
  models) printf '2b   Qwen/Qwen3.5-2B (default, reasoning)\nor any downloaded Hugging Face id: llm serve <org>/<model>\n' ;;
  *) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
```

### `~/.local/share/llm/chat.py`

```python
"""Terminal chat with the local vLLM server (used by `llm chat` / `llm ask`)."""
import argparse
import sys

from openai import OpenAI

DIM, RESET = "\033[2m", "\033[0m"

# Sampling settings recommended by the model card (README.md of Qwen/Qwen3.5-2B) for text tasks.
# presence_penalty makes the model avoid repeating itself; without it the 2B model easily gets
# stuck in "thinking loops".
SAMPLING_THINK = dict(temperature=1.0, top_p=0.95, presence_penalty=1.5, top_k=20)
SAMPLING_NO_THINK = dict(temperature=0.7, top_p=0.8, presence_penalty=1.5, top_k=20)

# Loop guard: the last TAIL characters of the reasoning are looked up in the reasoning so far.
# If that exact piece of text already appeared REPEATS times, the model is going round in circles.
TAIL, REPEATS, CHECK_EVERY = 300, 3, 200

p = argparse.ArgumentParser()
p.add_argument("--port", default="8000")
p.add_argument("--ask", default=None)
p.add_argument("--no-think", action="store_true")
args = p.parse_args()

client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="none")
try:
    model = client.models.list().data[0].id
except Exception:
    sys.exit("No model server running. Start one with: llm serve")


def is_looping(text):
    """True if the end of the text is an exact copy of something written REPEATS times before."""
    if len(text) < TAIL * (REPEATS + 1):
        return False
    return text.count(text[-TAIL:]) > REPEATS


def stream_reply(messages, think):
    """Stream one reply. Returns (answer, looped). Ctrl+C stops the reply, not the chat."""
    s = SAMPLING_THINK if think else SAMPLING_NO_THINK
    stream = client.chat.completions.create(
        model=model, messages=messages, stream=True,
        temperature=s["temperature"], top_p=s["top_p"], presence_penalty=s["presence_penalty"],
        extra_body={"top_k": s["top_k"], "chat_template_kwargs": {"enable_thinking": think}},
    )
    answer, reasoning, thinking, checked = "", "", False, 0
    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            d = chunk.choices[0].delta
            r = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
            if r:
                if not thinking:
                    print(f"{DIM}[thinking] ", end="")
                    thinking = True
                print(r, end="", flush=True)
                reasoning += r
                if len(reasoning) - checked >= CHECK_EVERY:
                    checked = len(reasoning)
                    if is_looping(reasoning):
                        print(f"{RESET}\n\n[loop detected: the reasoning repeats itself; stopped]")
                        return answer, True
            if d.content:
                if thinking:
                    print(f"{RESET}\n")
                    thinking = False
                print(d.content, end="", flush=True)
                answer += d.content
    except KeyboardInterrupt:
        print(f"{RESET}\n[stopped]")
    finally:
        stream.close()          # closing the connection makes the server stop generating
    print(RESET)
    return answer, False


def reply(messages):
    answer, looped = stream_reply(messages, think=not args.no_think)
    if looped:
        # Ask again without reasoning, so that the user still gets an answer.
        print("[answering again without reasoning]\n")
        answer, _ = stream_reply(messages, think=False)
    return answer


if args.ask:
    reply([{"role": "user", "content": args.ask}])
    sys.exit()

print(f"Chatting with {model}. Type /reset to clear, /exit to quit. Ctrl+C stops a reply.")
history = []
while True:
    try:
        q = input("\n> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if q in ("/exit", "/quit"):
        break
    if q == "/reset":
        history = []
        print("(cleared)")
        continue
    if not q:
        continue
    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": reply(history)})
```
