#!/usr/bin/env python3
"""Stage 0: LLM-only zero-shot baseline. vLLM, one forward pass per patient, score = log-odds(Yes vs No) at the first token.

  python run_llm_zeroshot.py --lead 12 --model medgemma-27b [--max-eval N] [--tag zs] [--splits dev,test]
Writes results/lead{L}_{tag}_{model}_zeroshot/ via exp_common.save_run (metrics incl. subgroups, predictions CSV)."""
from __future__ import annotations
import argparse, os, time, math
import numpy as np, pandas as pd
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from exp_common import metrics, save_run
from llm_common import model_dir, load_timeline, build_messages, yes_no_token_ids


def subgroups(recs):
    age = np.array([r["age"] for r in recs]); nn = np.array([r["n_notes"] for r in recs], float)
    return {"sex": [r["sex"] for r in recs], "age_band": pd.cut(age, [0, 15, 18, 22, 100], labels=["12-15", "15-18", "18-22", "22+"]).astype(str),
            "note_quartile": pd.qcut(nn, 4, labels=["q1", "q2", "q3", "q4"], duplicates="drop").astype(str) if len(recs) >= 8 else ["q1"] * len(recs)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lead", type=int, default=12); ap.add_argument("--model", default="medgemma-27b"); ap.add_argument("--max-eval", type=int, default=None)
    ap.add_argument("--tag", default="zs"); ap.add_argument("--splits", default="dev,test"); ap.add_argument("--max-model-len", type=int, default=8192); ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--batch", type=int, default=2000); ap.add_argument("--input", default="timeline", choices=["timeline", "xml"], help="xml = pre-L4 representation (exp_common.build_text notes_ehr), same as the ModernBERT baseline"); ap.add_argument("--max-chars", type=int, default=14000); a = ap.parse_args()
    global load_timeline
    if a.input == "xml":
        from exp_common import load_split, build_text
        def load_timeline(lead, split, max_n=None, seed_salt="subset"):
            def slim(r): return {"person_id": r["person_id"], "label": r["label"], "sex": r.get("sex"), "age": r.get("age_at_cutoff"), "n_notes": (r.get("features") or {}).get("n_notes", 0),
                                 "text": build_text(r, "notes_ehr", max_chars=a.max_chars, header=True)}
            if max_n: out = [slim(r) for r in load_split(lead, split, max_n, keep_text=True, seed_salt=seed_salt)]
            else:   # stream the JSONL so that only the packed texts stay in memory (full records are ~10x larger; matters on small-RAM hosts)
                import json as _json
                from exp_common import DATA as _DATA
                out = []
                with open(_DATA / f"lead_{lead}mo" / f"{split}.jsonl") as fh:
                    for line in fh: out.append(slim(_json.loads(line)))
            print(f"[xml] {split}: {len(out)} records, text chars median {int(np.median([len(o['text']) for o in out]))}", flush=True); return out
    from vllm import LLM, SamplingParams
    md = model_dir(a.model); t0 = time.time()
    llm = LLM(model=md, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, seed=0, max_num_seqs=256)
    tok = llm.get_tokenizer(); yes_ids, no_ids = yes_no_token_ids(tok); print(f"[{a.model}] yes ids {yes_ids} no ids {no_ids}; loaded in {time.time()-t0:.0f}s", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    preds = {}; ms = {}; n_trunc = 0
    for split in a.splits.split(","):
        recs = load_timeline(a.lead, split, a.max_eval); scores = []; t1 = time.time()
        for i in range(0, len(recs), a.batch):
            chunk = recs[i:i + a.batch]; prompts = []
            for r in chunk:
                p = tok.apply_chat_template(build_messages(r["text"], a.lead), tokenize=False, add_generation_prompt=True)
                ids = tok.encode(p, add_special_tokens=False)
                if len(ids) > a.max_model_len - 8:   # keep head (instruction) + tail (recent evidence): drop the middle
                    n_trunc += 1; head = ids[:1200]; tail = ids[-(a.max_model_len - 8 - 1200):]; p = tok.decode(head) + "\n...[truncated]...\n" + tok.decode(tail)
                prompts.append(p)
            outs = llm.generate(prompts, sp, use_tqdm=False)
            for o in outs:
                lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
                ly = max([lp[t].logprob for t in yes_ids if t in lp], default=-30.0); ln = max([lp[t].logprob for t in no_ids if t in lp], default=-30.0)
                scores.append(1.0 / (1.0 + math.exp(-(ly - ln))))
            print(f"  {split}: {min(i+a.batch, len(recs))}/{len(recs)} | {(min(i+a.batch, len(recs)))/(time.time()-t1):.1f} patients/s | truncated {n_trunc}", flush=True)
        y = np.array([r["label"] for r in recs]); s = np.array(scores)
        ms[split] = metrics(y, s, subgroups(recs)); preds[split] = pd.DataFrame({"person_id": [r["person_id"] for r in recs], "label": y, "prob": s})
        print(f"lead {a.lead} {a.model} zero-shot {split}: AUROC {ms[split]['auroc']:.4f} AP {ms[split]['auprc']:.4f} sens@sp90 {ms[split]['sens_at_spec90']:.3f} ppv@5 {ms[split]['ppv_top5']:.3f} ece {ms[split]['ece']:.3f} | n={len(y)}", flush=True)
    save_run(f"lead{a.lead}_{a.tag}_{a.model}_zeroshot", {"lead": a.lead, "model": a.model, "model_dir": md, "tag": a.tag, "scoring": "logodds(Yes,No) first token", "max_model_len": a.max_model_len, "n_truncated": n_trunc}, ms, preds)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
