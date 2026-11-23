#!/usr/bin/env python
"""Question, reasoning and answer embeddings for the controller (all-MiniLM-L6-v2 + PCA 64).

Encoder: sentence-transformers/all-MiniLM-L6-v2 from the local Hugging Face cache (no downloads),
mean pooling, unit length, 384 dims, float32, at most 256 tokens. The GPU is used only if it is
free; otherwise the CPU.

For each set {gsm8k,math,amc23}_{train,val,test}, written to controller_data/embeddings/:

  <set>_rows.npz      row_index, trace_id, question_id, budget, q_index (one per controller-CSV row)
  <set>_q_ids.npy     question_id of each row of the q arrays
  <set>_q_384.npy     one embedding per unique question            (q_index points here)
  <set>_r_384.npy     one per CSV row: the reasoning up to the row's reasoning_tokens_used,
                      keeping the END of the text when truncating. Budget-0 rows: zero vector.
  <set>_a_384.npy     one per CSV row: answer_text, keeping the START when truncating
  <set>_{q,r,a}_64.npy  the PCA projections of the arrays above (zero r vectors stay zero)

The reasoning is cut on the token ids recorded in each run's traces.jsonl and decoded with the
Qwen3.5-2B tokenizer, so nothing after the budget is seen. Every full decoded trace is checked
against reasoning_text in <set>_traces.csv.

PCA (embeddings/pca/{q,r,a}_pca64.npz: mean, components, explained_variance, explained_variance_ratio,
n_fit) is fitted separately for q, r and a on the TRAIN sets of the three datasets only:
q on the train questions, r and a on train rows of config t0.6_pp1.5, without zero vectors.

  ~/.venvs/vllm/bin/python controller_data/build_embeddings.py            # build missing, fit, verify
  ~/.venvs/vllm/bin/python controller_data/build_embeddings.py --force    # rebuild everything
  ~/.venvs/vllm/bin/python controller_data/build_embeddings.py --verify   # only check saved files
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "embeddings"
PCA_DIR = OUT / "pca"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
LLM = "Qwen/Qwen3.5-2B"
CONFIG = "t0.6_pp1.5"
DIM, PCA_DIM, MAX_TOKENS = 384, 64, 256
DATASETS, SPLITS = ("gsm8k", "math", "amc23"), ("train", "val", "test")
SETS = [f"{d}_{s}" for d in DATASETS for s in SPLITS]
KINDS = ("q", "r", "a")


def traces_jsonl(name):
    d, s = name.split("_")
    return {"gsm8k": REPO / f"generation_gsm8k/data/results/{s}",
            "math": REPO / f"generation_math/data/results/math_{s}",
            "amc23": REPO / "generation_amc23/data/results/amc23"}[d] / CONFIG / "traces.jsonl"


def fixed_str(series):
    """A fixed-width unicode array, loadable without pickle."""
    return np.array(series.tolist(), dtype=np.str_)


def save(path, array):
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def vllm_server_running():
    """True if a process was started as `vllm serve` (checks its arguments, not a shell's text)."""
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if b"serve" in argv and any(Path(a.decode(errors="ignore")).name == "vllm" for a in argv):
            return True
    return False


def pick_device(choice):
    if choice != "auto":
        return choice
    if not torch.cuda.is_available():
        return "cpu"
    try:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
        used = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=20).stdout.split()[0])
    except Exception:
        return "cpu"
    return "cuda" if not apps and not vllm_server_running() and used < 500 else "cpu"


class Encoder:
    def __init__(self, device, batch_size):
        self.tok = AutoTokenizer.from_pretrained(ENCODER, local_files_only=True)
        self.model = AutoModel.from_pretrained(ENCODER, local_files_only=True).to(device).eval()
        self.device, self.batch_size = device, batch_size

    @torch.inference_mode()
    def __call__(self, texts, keep):
        """Unit-length embeddings. keep='start' or 'end' chooses which part survives truncation."""
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        if not texts:
            return out
        self.tok.truncation_side = "right" if keep == "start" else "left"
        ids = self.tok(list(texts), truncation=True, max_length=MAX_TOKENS,
                       return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        order = np.argsort([len(x) for x in ids], kind="stable")
        for i in range(0, len(order), self.batch_size):
            b = order[i:i + self.batch_size]
            batch = self.tok.pad({"input_ids": [ids[j] for j in b]}, return_tensors="pt").to(self.device)
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            v = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out[b] = torch.nn.functional.normalize(v, dim=1).float().cpu().numpy()
        return out


def encode_unique(enc, texts, keep, chunk=8192):
    """Embed a list of strings, encoding each distinct string once."""
    inverse, distinct = pd.factorize(pd.Series(texts, dtype=object))
    first = list(distinct)
    vecs = np.zeros((len(first), DIM), dtype=np.float32)
    for i in range(0, len(first), chunk):
        vecs[i:i + chunk] = enc(first[i:i + chunk], keep)
    return vecs[inverse]


def load_recorded_ids(name, trace_ids):
    want, ids = set(trace_ids), {}
    with open(traces_jsonl(name)) as f:
        for line in f:
            r = json.loads(line)
            tid = f"{r['question_id']}#{r['sample']}"
            if tid in want:
                if tid in ids:
                    raise SystemExit(f"{name}: duplicate recorded trace {tid}")
                ids[tid] = np.asarray(r["reasoning_ids"], dtype=np.int32)
    if len(ids) != len(want):
        raise SystemExit(f"{name}: {len(want) - len(ids)} traces have no recorded ids")
    return ids


def check_decoding(name, llm, ids, traces, chunk=512):
    tids, texts = traces.trace_id.tolist(), traces.reasoning_text.tolist()
    for i in range(0, len(tids), chunk):
        dec = llm.decode_batch([ids[t].tolist() for t in tids[i:i + chunk]], skip_special_tokens=True)
        for t, a, b in zip(tids[i:i + chunk], dec, texts[i:i + chunk]):
            if a != b:
                raise SystemExit(f"{name}: decoded reasoning differs from the traces CSV at {t}")


def encode_reasoning(enc, llm, rows, ids, chunk=2048):
    """One vector per CSV row; identical reasoning prefixes are encoded once."""
    out = np.zeros((len(rows), DIM), dtype=np.float32)
    has = (rows.reasoning_tokens_used.values > 0)
    pairs = pd.DataFrame({"t": rows.trace_id.values[has], "n": rows.reasoning_tokens_used.values[has]})
    keys = pairs.drop_duplicates().reset_index(drop=True)
    key_index = pairs.merge(keys.assign(k=np.arange(len(keys))), on=["t", "n"], how="left").k.values
    store = np.zeros((len(keys), DIM), dtype=np.float32)
    to_store, seen, n = np.zeros(len(keys), dtype=np.int64), {}, 0
    for i in range(0, len(keys), chunk):
        part = keys.iloc[i:i + chunk]
        texts = llm.decode_batch([ids[t][:c].tolist() for t, c in zip(part.t, part.n)], skip_special_tokens=True)
        new = []
        for j, x in enumerate(texts):
            h = hashlib.blake2b(x.encode("utf-8", "surrogatepass"), digest_size=16).digest()
            if h not in seen:
                seen[h] = n + len(new)
                new.append(x)
            to_store[i + j] = seen[h]
        store[n:n + len(new)] = enc(new, "end")
        n += len(new)
    out[has] = store[to_store[key_index]]
    return out, len(keys), n


def build_set(name, enc, llm):
    t0 = time.time()
    cols = ["trace_id", "question_id", "question", "config", "budget", "reasoning_tokens_used", "answer_text"]
    rows = pd.read_csv(HERE / f"{name}_controller.csv", usecols=cols, keep_default_na=False, na_filter=False,
                       dtype={"trace_id": str, "question_id": str, "question": str, "config": str, "answer_text": str,
                              "budget": np.int32, "reasoning_tokens_used": np.int32})[cols]
    traces = pd.read_csv(HERE / f"{name}_traces.csv", usecols=["trace_id", "reasoning_text"],
                         keep_default_na=False, na_filter=False, dtype=str)
    if traces.trace_id.duplicated().any():
        raise SystemExit(f"{name}: duplicate trace_id in traces CSV")
    if not set(rows.trace_id) <= set(traces.trace_id):
        raise SystemExit(f"{name}: controller rows without a trace")
    if ((rows.budget == 0) != (rows.config == "direct")).any() or (rows.reasoning_tokens_used[rows.budget == 0] != 0).any():
        raise SystemExit(f"{name}: budget-0 rows are not exactly the direct rows")
    ids = load_recorded_ids(name, traces.trace_id.tolist())
    check_decoding(name, llm, ids, traces)
    del traces

    q = rows.drop_duplicates("question_id")[["question_id", "question"]].reset_index(drop=True)
    if rows.groupby("question_id").question.nunique().max() != 1:
        raise SystemExit(f"{name}: one question_id has several texts")
    save(OUT / f"{name}_q_384.npy", encode_unique(enc, q.question.tolist(), "start"))
    save(OUT / f"{name}_q_ids.npy", fixed_str(q.question_id))
    r, n_keys, n_enc = encode_reasoning(enc, llm, rows, ids)
    save(OUT / f"{name}_r_384.npy", r)
    save(OUT / f"{name}_a_384.npy", encode_unique(enc, rows.answer_text.tolist(), "start"))
    q_index = pd.Series(np.arange(len(q)), index=q.question_id).loc[rows.question_id].values.astype(np.int32)
    tmp = OUT / f"{name}_rows.tmp.npz"
    np.savez(tmp, row_index=np.arange(len(rows), dtype=np.int64), trace_id=fixed_str(rows.trace_id),
             question_id=fixed_str(rows.question_id), budget=rows.budget.values, q_index=q_index)
    os.replace(tmp, OUT / f"{name}_rows.npz")
    print(f"{name}: {len(rows)} rows, {len(q)} questions, {n_keys} reasoning prefixes ({n_enc} distinct), "
          f"{time.time() - t0:.0f}s", flush=True)


def fit_pca(X):
    """Exact PCA from the covariance matrix; each component's largest entry is positive."""
    mean = X.mean(0, dtype=np.float64)
    C = np.zeros((DIM, DIM))
    for i in range(0, len(X), 20000):
        b = X[i:i + 20000].astype(np.float64) - mean
        C += b.T @ b
    C /= len(X) - 1
    w, v = np.linalg.eigh(C)
    w, v = w[::-1], v[:, ::-1].T
    comp = v[:PCA_DIM]
    comp *= np.sign(comp[np.arange(PCA_DIM), np.abs(comp).argmax(1)])[:, None]
    return dict(mean=mean, components=comp, explained_variance=w[:PCA_DIM],
                explained_variance_ratio=w[:PCA_DIM] / w.sum(), n_fit=len(X))


def project(X, p):
    z = np.zeros((len(X), PCA_DIM), dtype=np.float32)
    keep = np.any(X != 0, axis=1)
    z[keep] = ((X[keep].astype(np.float64) - p["mean"]) @ p["components"].T).astype(np.float32)
    return z


def fit_and_project():
    PCA_DIR.mkdir(parents=True, exist_ok=True)
    train = [f"{d}_train" for d in DATASETS]
    for k in KINDS:
        if k == "q":
            X = np.concatenate([np.load(OUT / f"{s}_q_384.npy") for s in train])
        else:
            parts = []
            for s in train:
                cfg = pd.read_csv(HERE / f"{s}_controller.csv", usecols=["config"]).config.values
                v = np.load(OUT / f"{s}_{k}_384.npy")[cfg == CONFIG]
                parts.append(v[np.any(v != 0, axis=1)])
            X = np.concatenate(parts)
        p = fit_pca(X)
        tmp = PCA_DIR / f"{k}_pca64.tmp.npz"
        np.savez(tmp, **p)
        os.replace(tmp, PCA_DIR / f"{k}_pca64.npz")
        print(f"PCA {k}: fit on {len(X)} vectors, explained variance {p['explained_variance_ratio'].sum():.4f}", flush=True)
        for s in SETS:
            save(OUT / f"{s}_{k}_64.npy", project(np.load(OUT / f"{s}_{k}_384.npy"), p))


def verify():
    pcas = {k: dict(np.load(PCA_DIR / f"{k}_pca64.npz")) for k in KINDS}
    total, n_q_train = 0, 0
    for s in SETS:
        csv = pd.read_csv(HERE / f"{s}_controller.csv", usecols=["trace_id", "question_id", "budget"],
                          keep_default_na=False, dtype={"trace_id": str, "question_id": str})
        m = np.load(OUT / f"{s}_rows.npz")
        qids = np.load(OUT / f"{s}_q_ids.npy")
        assert (m["trace_id"] == fixed_str(csv.trace_id)).all(), f"{s}: trace_id order"
        assert (m["budget"] == csv.budget.values).all(), f"{s}: budget order"
        assert (qids[m["q_index"]] == fixed_str(csv.question_id)).all(), f"{s}: q_index"
        assert len(qids) == csv.question_id.nunique(), f"{s}: question count"
        n_q_train += len(qids) if s.endswith("train") else 0
        for k in KINDS:
            x, z = np.load(OUT / f"{s}_{k}_384.npy"), np.load(OUT / f"{s}_{k}_64.npy")
            n = len(qids) if k == "q" else len(csv)
            assert x.shape == (n, DIM) and z.shape == (n, PCA_DIM), f"{s} {k}: shape"
            assert x.dtype == np.float32 and z.dtype == np.float32, f"{s} {k}: dtype"
            assert np.isfinite(x).all() and np.isfinite(z).all(), f"{s} {k}: NaN/inf"
            norm = np.linalg.norm(x, axis=1)
            zero = norm == 0
            assert (np.abs(norm[~zero] - 1) < 1e-4).all(), f"{s} {k}: norm"
            if k == "r":
                assert (zero == (csv.budget.values == 0)).all(), f"{s}: zero r vectors must be the budget-0 rows"
            else:
                assert not zero.any(), f"{s} {k}: zero vector"
            assert np.array_equal(project(x, pcas[k]), z), f"{s} {k}: PCA reload differs"
        total += len(csv)
    assert n_q_train == 1343, f"train questions: {n_q_train}"
    print(f"verified: {len(SETS)} sets, {total} rows, {n_q_train} train questions", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--force", action="store_true", help="re-encode sets that are already complete")
    p.add_argument("--verify", action="store_true", help="only check the saved files")
    a = p.parse_args()
    if a.verify:
        return verify()
    OUT.mkdir(parents=True, exist_ok=True)
    device = pick_device(a.device)
    print("device:", device, flush=True)
    enc = Encoder(device, a.batch_size)
    llm = AutoTokenizer.from_pretrained(LLM, local_files_only=True).backend_tokenizer
    for s in SETS:
        done = all((OUT / f"{s}_{x}").exists() for x in ("q_384.npy", "r_384.npy", "a_384.npy", "q_ids.npy", "rows.npz"))
        if a.force or not done:
            build_set(s, enc, llm)
    fit_and_project()
    verify()


if __name__ == "__main__":
    main()
