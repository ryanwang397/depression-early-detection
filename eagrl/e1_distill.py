#!/usr/bin/env python3
"""e1: Stage-1 trajectory distillation by rejection sampling (replaces the paper's expert-guided MCTS).

  python e1_distill.py --lead 3 --model llama-3.1-8b --n-pos 1500 --n-neg 1500 --k-unguided 3 --k-guided 3 --keep 2
For a label-balanced train subset, the base LLM samples K_u trajectories from the plain prompt and K_g from a hindsight prompt
(true outcome + expert top concepts as private notes).  Every candidate is scored WITHOUT the hindsight information:
    p_yes = P(Yes | plain prompt + trajectory up to "<Final answer>")   (1-token vLLM pass with logprobs)
    R     = log p(y*) + margin + w_att * Jaccard(concepts, expert top) - w_ground * #ungrounded concepts
and the `keep` best format-valid, label-consistent trajectories per patient form the SFT set (outputs/eagrl/lead{L}_{tag}.jsonl)."""
from __future__ import annotations
import argparse, json, os, time
from collections import Counter, defaultdict
import numpy as np
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from common import (set_expert_variant, set_tl_suffix, set_knowledge_card, null_expert, OUT, model_dir, load_timeline, yes_no_token_ids, build_messages, encode_prompt, parse_trajectory, canonical_text, reward, balanced_subset, load_expert,
                    p_yes_from_logprobs, synthetic_trajectory, metrics)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lead", type=int, default=3); ap.add_argument("--model", default="llama-3.1-8b"); ap.add_argument("--tag", default="sft")
    ap.add_argument("--n-pos", type=int, default=1500); ap.add_argument("--n-neg", type=int, default=1500); ap.add_argument("--k-unguided", type=int, default=3); ap.add_argument("--k-guided", type=int, default=3)
    ap.add_argument("--keep", type=int, default=2); ap.add_argument("--temperature", type=float, default=0.8); ap.add_argument("--top-p", type=float, default=0.95); ap.add_argument("--max-tokens", type=int, default=800); ap.add_argument("--prefill", type=int, default=1, help="prefill the assistant turn with <Subquestion 1> to pin the tag format")
    ap.add_argument("--max-prompt-tokens", type=int, default=3300); ap.add_argument("--max-model-len", type=int, default=4608); ap.add_argument("--gpu-mem", type=float, default=0.9); ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--w-att", type=float, default=0.5); ap.add_argument("--w-ground", type=float, default=0.1); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-expert", action="store_true", help="expert-free: label-only hindsight, no alignment reward (w_att forced to 0)"); ap.add_argument("--expert-variant", default=None, help="suffix of the e0 expert file to use, e.g. _inc (incident-only, prior-evidence gated)"); ap.add_argument("--tl-suffix", default=""); ap.add_argument("--knowledge-card", action="store_true")
    a = ap.parse_args(); t0 = time.time(); set_tl_suffix(a.tl_suffix); set_knowledge_card(a.knowledge_card)
    if a.expert_variant: set_expert_variant(a.expert_variant)
    if a.no_expert: a.w_att = 0.0
    recs = load_timeline(a.lead, "train"); expert = null_expert() if a.no_expert else load_expert(a.lead)
    sub = balanced_subset(recs, a.n_pos, a.n_neg, "eagrl_sft"); print(f"[e1 lead {a.lead}] {len(sub)} patients ({sum(r['label'] for r in sub)} pos) from {len(recs)} train", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_dir(a.model), dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, seed=a.seed, max_num_seqs=a.max_num_seqs)
    tok = llm.get_tokenizer(); yes_ids, no_ids = yes_no_token_ids(tok); print(f"model loaded {time.time()-t0:.0f}s; yes {yes_ids} no {no_ids}", flush=True)
    hind = lambda r: {"label": r["label"], "expert": [(c, dict(expert[r["person_id"]]["signed"]).get(c, 0.0)) for c in expert[r["person_id"]]["top"]]}
    P_un = [encode_prompt(tok, build_messages(r["text"], a.lead), a.max_prompt_tokens) for r in sub]
    P_gd = [encode_prompt(tok, build_messages(r["text"], a.lead, hind(r)), a.max_prompt_tokens) for r in sub]
    print(f"prompt tokens median {int(np.median([len(p) for p in P_un]))} max {max(len(p) for p in P_un)}; encoded in {time.time()-t0:.0f}s", flush=True)
    cands = []; pre = tok.encode("<Subquestion 1>", add_special_tokens=False) if a.prefill else []
    for src, P, K in (("unguided", P_un, a.k_unguided), ("guided", P_gd, a.k_guided)):
        if K <= 0: continue
        t1 = time.time(); outs = llm.generate([{"prompt_token_ids": p + pre} for p in P], SamplingParams(n=K, temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, seed=a.seed), use_tqdm=False)
        for i, o in enumerate(outs):
            for c in o.outputs:
                cands.append({"i": i, "src": src, "text": ("<Subquestion 1>" if a.prefill else "") + c.text, "finish": c.finish_reason, "n_tok": len(c.token_ids)})
        print(f"{src}: {len(outs)*K} samples in {time.time()-t1:.0f}s ({sum(len(c.token_ids) for o in outs for c in o.outputs)/(time.time()-t1):.0f} tok/s)", flush=True)
    if a.smoke:   # make the guided half format-valid so that downstream stages have data to run on
        for c in cands:
            if c["src"] == "guided": c["text"] = synthetic_trajectory(sub[c["i"]]["label"], expert[sub[c["i"]]["person_id"]]["top"], a.lead)
    for c in cands:
        t = c["text"]; cut = t.find("</Final answer>"); t = t[:cut + len("</Final answer>")] if cut >= 0 else t
        c["parsed"] = parse_trajectory(t, sub[c["i"]]["text"])
        if c["parsed"]["valid"]:   # canonical tag format for SFT targets; scoring prefix = everything before the answer token
            c["response"] = canonical_text(c["parsed"], a.lead); c["prefix"] = c["response"][: c["response"].rfind("<Final answer>") + len("<Final answer>")]
    valid = [c for c in cands if c["parsed"]["valid"]]
    t1 = time.time()
    if valid:
        outs = llm.generate([{"prompt_token_ids": (P_un[c["i"]] + tok.encode(c["prefix"], add_special_tokens=False))[: a.max_model_len - 2]} for c in valid],
                            SamplingParams(temperature=0.0, max_tokens=1, logprobs=20), use_tqdm=False)
        for c, o in zip(valid, outs):
            c["p_yes"] = p_yes_from_logprobs(o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}, yes_ids, no_ids)
    print(f"scored {len(valid)} valid candidates in {time.time()-t1:.0f}s", flush=True)
    for c in cands:
        r = sub[c["i"]]; c["rw"] = reward(c["parsed"], c.get("p_yes", 0.5), r["label"], expert[r["person_id"]]["top"], a.w_att, a.w_ground)
    # ---- selection ----
    by = defaultdict(list)
    for c in cands: by[c["i"]].append(c)
    kept = []; n_none = 0
    for i, cs in by.items():
        r = sub[i]; ok = [c for c in cs if c["parsed"]["valid"] and c["parsed"]["answer"] == ("Yes" if r["label"] == 1 else "No")]
        ok.sort(key=lambda c: -c["rw"]["R"])
        if not ok: n_none += 1
        for c in ok[: a.keep]:
            kept.append({"person_id": r["person_id"], "label": r["label"], "text": r["text"], "response": c["response"], "source": c["src"], "R": round(c["rw"]["R"], 4), "r_cls": round(c["rw"]["r_cls"], 4),
                         "r_att": round(c["rw"]["r_att"], 4), "p_yes": round(c["p_yes"], 4), "concepts": c["parsed"]["concepts"], "ungrounded": c["parsed"]["ungrounded"], "expert_top": expert[r["person_id"]]["top"], "n_sub": c["parsed"]["n_sub"]})
    out = OUT / f"lead{a.lead}_{a.tag}.jsonl"
    with open(out, "w") as f:
        for k in kept: f.write(json.dumps(k) + "\n")
    with open(OUT / f"lead{a.lead}_{a.tag}_candidates.jsonl", "w") as f:
        for c in cands:
            f.write(json.dumps({"person_id": sub[c["i"]]["person_id"], "label": sub[c["i"]]["label"], "src": c["src"], "finish": c["finish"], "n_tok": c["n_tok"], "valid": c["parsed"]["valid"], "answer": c["parsed"]["answer"],
                                "n_sub": c["parsed"]["n_sub"], "concepts": c["parsed"]["concepts"], "unknown": c["parsed"]["unknown_features"], "ungrounded": c["parsed"]["ungrounded"], "leak": c["parsed"]["leak"],
                                "p_yes": c.get("p_yes"), **{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in c["rw"].items()}, "text": c["text"]}) + "\n")
    # ---- stats ----
    st = {"lead": a.lead, "model": a.model, "n_patients": len(sub), "n_candidates": len(cands), "n_kept": len(kept), "n_patients_without_valid": n_none, "kept_by_source": dict(Counter(k["source"] for k in kept)),
          "kept_label_balance": dict(Counter(k["label"] for k in kept)), "seconds": round(time.time() - t0)}
    for src in ("unguided", "guided"):
        cs = [c for c in cands if c["src"] == src]
        if not cs: continue
        v = [c for c in cs if c["parsed"]["valid"]]
        st[src] = {"n": len(cs), "valid_rate": len(v) / len(cs), "tagged_rate": np.mean([c["parsed"]["tagged"] for c in cs]), "truncated_rate": np.mean([c["finish"] == "length" for c in cs]), "leak_rate": np.mean([c["parsed"]["leak"] for c in cs]),
                   "answer_correct_rate": np.mean([c["parsed"]["answer"] == ("Yes" if sub[c["i"]]["label"] == 1 else "No") for c in v]) if v else None,
                   "mean_jaccard": float(np.mean([c["rw"]["r_att"] for c in v])) if v else None, "mean_ungrounded": float(np.mean([c["rw"]["n_ungrounded"] for c in v])) if v else None,
                   "mean_n_sub": float(np.mean([c["parsed"]["n_sub"] for c in v])) if v else None, "mean_R": float(np.mean([c["rw"]["R"] for c in v])) if v else None,
                   "unknown_feature_names": Counter(u.lower() for c in cs for u in c["parsed"]["unknown_features"]).most_common(20)}
    un = [c for c in cands if c["src"] == "unguided" and c["parsed"]["valid"]]
    if un:   # zero-shot diagnostic of the base model on this subset: mean p_yes of its own valid trajectories
        pp = defaultdict(list)
        for c in un: pp[c["i"]].append(c["p_yes"])
        ii = sorted(pp); y = [sub[i]["label"] for i in ii]; s = [float(np.mean(pp[i])) for i in ii]
        if len(set(y)) == 2: m = metrics(y, s, n_boot=0); st["unguided_zero_shot"] = {"n": len(y), "auroc": m["auroc"], "auprc": m["auprc"]}
    (OUT / f"lead{a.lead}_{a.tag}_stats.json").write_text(json.dumps(st, indent=2, default=str))
    print(json.dumps(st, indent=1, default=str)); print(f"wrote {len(kept)} SFT examples -> {out}; done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
