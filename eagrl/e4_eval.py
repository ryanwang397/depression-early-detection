#!/usr/bin/env python3
"""e4: evaluate an EAG-RL adapter (or the base model) with greedy structured reasoning; metrics + interpretability report.

  python e4_eval.py --lead 3 --model llama-3.1-8b --adapter results/eagrl/lead3_sft_llama-3.1-8b/adapter_final --tag eagrl_stage1 [--faithfulness 300]
Score = P(Yes) at the token after "<Final answer>" of the greedy trajectory (invalid trajectories are re-scored by appending the tag).
Writes results/lead{L}_{tag}_{model}/ (metrics.json incl. subgroups + registry row, predictions, trajectories_{split}.jsonl,
interpretability.json): format validity, expert agreement (Jaccard, per-concept precision/recall), concept citation rates and
lift among high-risk / true positives, phenotype clusters of cited-feature sets among the top-20% risk, grounding rate, and an
optional deletion-based faithfulness test (does removing the evidence of the top cited concept lower the risk more than removing a random uncited one?)."""
from __future__ import annotations
import argparse, json, os, re, time, random
from collections import Counter, defaultdict
import numpy as np, pandas as pd
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from common import (set_expert_variant, set_tl_suffix, set_knowledge_card, set_align_exclude_v3, resolve_path, RESULTS, CONCEPT_NAMES, CONCEPTS, ALIGN_EXCLUDE, model_dir, load_timeline, yes_no_token_ids, build_messages, encode_prompt, parse_trajectory, token_spans,
                    load_expert, null_expert, load_context, prompt_extras, p_yes_from_logprobs, metrics, save_run, jaccard, grounded)
from run_llm_zeroshot import subgroups

_GR = {c: (re.compile(p, re.I) if p else None) for c, p in CONCEPTS}


def cut_text(t):
    k = t.find("</Final answer>"); return t[:k + len("</Final answer>")] if k >= 0 else t


def delete_concept(text: str, concept: str) -> str:
    """Remove the evidence of one concept from a timeline: matching items of EVIDENCE lines ('; '-separated), and whole matching
    DIAGNOSES / MEDICATIONS / summary / excerpt lines. Header lines are never touched."""
    r = _GR.get(concept)
    if r is None: return text
    out = []
    for line in text.split("\n"):
        if line.startswith(("PATIENT:", "UTILIZATION:")): out.append(line); continue
        if re.match(r"\s+\S+ ago: ", line) and "; " in line or re.match(r"\s+\d+d ago: |\s+\d+mo ago: |\s+[\d.]+y ago: ", line):
            head, _, body = line.partition(": ")
            items = [it for it in body.split("; ") if not r.search(it)]
            if items: out.append(head + ": " + "; ".join(items))
            continue
        if r.search(line) and not line.startswith(("DIAGNOSES", "EVIDENCE", "RECENT NOTE SUMMARIES")): continue
        out.append(line)
    return "\n".join(out)


def report(rows, recs, score_texts, faithfulness=0, do_faith=False, n_clusters=6, seed=0):
    """Interpretability report for one split; rows = trajectory records, recs = timeline records (same order)."""
    # ---------------- interpretability report ----------------
    V = [row for row in rows if row["valid"]]; rep = {"n": len(rows), "valid_rate": len(V) / len(rows), "mean_n_sub": float(np.mean([r["n_sub"] for r in V])) if V else None,
                                                      "mean_jaccard_expert": float(np.mean([r["jaccard"] for r in V])) if V else None,
                                                      "grounding_rate": 1 - sum(len(r["ungrounded"]) for r in V) / max(sum(len(r["concepts"]) for r in V), 1),
                                                      "unknown_feature_names": Counter(u.lower() for r in rows for u in r["unknown"]).most_common(15),
                                                      "spearman_prob_vs_expert": float(pd.Series([r["prob"] for r in rows]).corr(pd.Series([r["expert_prob"] for r in rows]), method="spearman"))}
    if V:
        pv = np.array([r["prob"] for r in V]); yv = np.array([r["label"] for r in V]); hi = pv >= np.quantile(pv, 0.9); lo = pv <= np.quantile(pv, 0.5)
        cite = {c: np.array([c in r["concepts"] for r in V]) for c in CONCEPT_NAMES}; exp_has = {c: np.array([c in r["expert_top"] for r in V]) for c in CONCEPT_NAMES}
        rep["concepts"] = {c: {"cite_rate": float(cite[c].mean()), "cite_rate_top10": float(cite[c][hi].mean()) if hi.any() else None, "cite_rate_bottom50": float(cite[c][lo].mean()) if lo.any() else None,
                               "lift_top10_vs_all": float(cite[c][hi].mean() / cite[c].mean()) if cite[c].mean() > 0 and hi.any() else None,
                               "cite_rate_pos": float(cite[c][yv == 1].mean()) if (yv == 1).any() else None, "cite_rate_neg": float(cite[c][yv == 0].mean()) if (yv == 0).any() else None,
                               "expert_top_rate": float(exp_has[c].mean()), "p_cite_given_expert": float(cite[c][exp_has[c]].mean()) if exp_has[c].any() else None,
                               "p_expert_given_cite": float(exp_has[c][cite[c]].mean()) if cite[c].any() else None, "grounded_rate": float(np.mean([c not in r["ungrounded"] for r in V if c in r["concepts"]])) if cite[c].any() else None}
                           for c in CONCEPT_NAMES if cite[c].any() or exp_has[c].any()}
        # phenotypes among the top-20% predicted risk: k-means on binary cited-concept vectors
        top = [r for r in V if r["prob"] >= np.quantile(pv, 0.8)]; cols = [c for c in CONCEPT_NAMES if c not in ALIGN_EXCLUDE]
        if len(top) >= 5 * n_clusters:
            from sklearn.cluster import KMeans
            X = np.array([[c in r["concepts"] for c in cols] for r in top], dtype=float); km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(X); base_rate = X.mean(0)
            rep["phenotypes_top20"] = []
            for k in range(n_clusters):
                m = km.labels_ == k
                if not m.any(): continue
                fr = X[m].mean(0); order = np.argsort(-fr)[:6]
                rep["phenotypes_top20"].append({"cluster": k, "n": int(m.sum()), "positive_rate": float(np.mean([r["label"] for r, mm in zip(top, m) if mm])), "mean_prob": float(np.mean([r["prob"] for r, mm in zip(top, m) if mm])),
                                                 "top_concepts": [(cols[j], round(float(fr[j]), 2), round(float(fr[j] / max(base_rate[j], 1e-6)), 2)) for j in order if fr[j] > 0]})
        co = Counter()
        for r in V:
            cs = sorted(set(r["concepts"]) - ALIGN_EXCLUDE)
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)): co[(cs[i], cs[j])] += 1
        rep["top_cooccurrence"] = [(f"{x} + {y}", n) for (x, y), n in co.most_common(15)]
    # ---------------- deletion-based faithfulness test ----------------
    if faithfulness and do_faith:
        random.seed(seed); cand = [r for r in V if r["concepts"] and _GR.get(r["concepts"][0]) is not None and r["concepts"][0] not in ALIGN_EXCLUDE]; random.shuffle(cand); cand = cand[:faithfulness]
        byid = {r["person_id"]: r for r in recs}; mod_cited = []; mod_ctrl = []; keep = []
        for r in cand:
            t = byid[r["person_id"]]["text"]; c0 = r["concepts"][0]; t1_ = delete_concept(t, c0)
            others = [c for c in CONCEPT_NAMES if c not in r["concepts"] and c not in ALIGN_EXCLUDE and _GR.get(c) is not None and grounded(c, t)]
            if t1_ == t or not others: continue
            cc = random.choice(others); t2_ = delete_concept(t, cc)
            if t2_ == t: continue
            mod_cited.append(t1_); mod_ctrl.append(t2_); keep.append((r, c0, cc))
        if keep:
            kp = [r["person_id"] for r, _, _ in keep]; p1, _, _ = score_texts(mod_cited, kp); p2, _, _ = score_texts(mod_ctrl, kp)
            d1 = np.array([r["prob"] - q for (r, _, _), q in zip(keep, p1)]); d2 = np.array([r["prob"] - q for (r, _, _), q in zip(keep, p2)])
            rep["faithfulness_deletion"] = {"n": len(keep), "mean_drop_delete_top_cited": float(d1.mean()), "mean_drop_delete_random_uncited": float(d2.mean()), "frac_cited_drop_larger": float((d1 > d2).mean()),
                                            "frac_cited_drop_positive": float((d1 > 0).mean()), "abs_mean_cited": float(np.abs(d1).mean()), "abs_mean_control": float(np.abs(d2).mean()),
                                            "examples": [{"person_id": r["person_id"], "cited": c0, "control": cc, "drop_cited": float(x), "drop_control": float(z)} for (r, c0, cc), x, z in list(zip(keep, d1, d2))[:10]]}
            print(f"  faithfulness (n={len(keep)}): mean prob drop when deleting top cited concept {d1.mean():+.4f} vs random uncited {d2.mean():+.4f}; cited drop larger in {100*(d1 > d2).mean():.0f}%", flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lead", type=int, default=3); ap.add_argument("--model", default="llama-3.1-8b"); ap.add_argument("--adapter", default=None); ap.add_argument("--tag", default="eagrl")
    ap.add_argument("--splits", default="dev,test"); ap.add_argument("--max-eval", type=int, default=None); ap.add_argument("--max-tokens", type=int, default=800); ap.add_argument("--max-prompt-tokens", type=int, default=3300)
    ap.add_argument("--max-model-len", type=int, default=4608); ap.add_argument("--gpu-mem", type=float, default=0.9); ap.add_argument("--max-num-seqs", type=int, default=256); ap.add_argument("--faithfulness", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=1, help="self-consistency: sample N trajectories per patient and average P(Yes) (1 = greedy)"); ap.add_argument("--sample-temp", type=float, default=0.7); ap.add_argument("--sample-top-p", type=float, default=0.95); ap.add_argument("--include-greedy", action="store_true", help="add the greedy trajectory as one more sample"); ap.add_argument("--chunk", type=int, default=512, help="prompts per vLLM generate call (bounds host RAM for logprobs)")
    ap.add_argument("--n-clusters", type=int, default=6); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--no-registry", action="store_true")
    ap.add_argument("--no-expert", action="store_true", help="expert-free evaluation (no expert agreement metrics)"); ap.add_argument("--expert-opinion", action="store_true"); ap.add_argument("--care-context", action="store_true"); ap.add_argument("--expert-variant", default=None, help="suffix of the e0 expert file to use, e.g. _inc (incident-only, prior-evidence gated)"); ap.add_argument("--tl-suffix", default=""); ap.add_argument("--knowledge-card", action="store_true"); ap.add_argument("--align-v3", action="store_true"); a = ap.parse_args(); set_tl_suffix(a.tl_suffix); set_knowledge_card(a.knowledge_card)
    if a.expert_variant: set_expert_variant(a.expert_variant)
    if a.align_v3: set_align_exclude_v3()
    a.adapter = resolve_path(a.adapter); t0 = time.time()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    md = model_dir(a.model); lora_req = None; kw = {}
    if a.adapter:
        r = json.load(open(os.path.join(a.adapter, "adapter_config.json")))["r"]; kw = {"enable_lora": True, "max_lora_rank": max(r, 8), "max_loras": 1}; lora_req = LoRARequest("eval", 1, a.adapter)
    llm = LLM(model=md, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, seed=a.seed, max_num_seqs=a.max_num_seqs, **kw)
    tok = llm.get_tokenizer(); yes_ids, no_ids = yes_no_token_ids(tok); expert = null_expert() if a.no_expert else load_expert(a.lead); context = load_context(a.lead) if a.care_context else None
    if a.no_expert: a.expert_opinion = False
    run = f"lead{a.lead}_{a.tag}_{a.model}"; rdir = RESULTS / run; rdir.mkdir(parents=True, exist_ok=True)
    print(f"[e4] {run} adapter={a.adapter}; loaded in {time.time()-t0:.0f}s", flush=True)

    def score_texts(texts, pids=None):
        """trajectories + P(Yes) for (text, pid) pairs; returns (probs, parsed list, responses). With --n-samples > 1 the prob is the
        mean of P(Yes) over N sampled trajectories (optionally + the greedy one); parsed/response come from the first completion."""
        pids = pids or [None] * len(texts)
        P = [encode_prompt(tok, build_messages(t, a.lead, **prompt_extras(p, expert if a.expert_opinion else None, context)), a.max_prompt_tokens) for t, p in zip(texts, pids)]
        sp_kw = dict(max_tokens=a.max_tokens, logprobs=20, stop=["</Final answer>"], include_stop_str_in_output=True)
        runs = []
        if a.n_samples > 1: runs.append(SamplingParams(n=a.n_samples, temperature=a.sample_temp, top_p=a.sample_top_p, seed=a.seed, **sp_kw))
        if a.n_samples == 1 or a.include_greedy: runs.append(SamplingParams(temperature=0.0, seed=0, **sp_kw))
        # generate in chunks and keep only (prob | redo token ids) per completion: full outputs with logprobs for N samples x all
        # patients do not fit in host RAM
        probs_all = [[] for _ in P]; redo = []; parsed = [None] * len(P); resp = [None] * len(P); CH = a.chunk
        for sp in runs:
            for c0 in range(0, len(P), CH):
                outs = llm.generate([{"prompt_token_ids": p} for p in P[c0:c0 + CH]], sp, lora_request=lora_req, use_tqdm=False)
                for k, o in enumerate(outs):
                    i = c0 + k
                    for c in o.outputs:
                        tids = list(c.token_ids); ans, _ = token_spans(tok, tids); pr = parse_trajectory(cut_text(c.text), texts[i]); j = len(probs_all[i])
                        if parsed[i] is None: parsed[i] = pr; resp[i] = cut_text(c.text)
                        if pr["valid"] and ans is not None and c.logprobs is not None and ans < len(c.logprobs): probs_all[i].append(p_yes_from_logprobs(c.logprobs[ans], yes_ids, no_ids))
                        else: probs_all[i].append(None); redo.append((i, j, tids))
                del outs
        if redo:
            outs2 = llm.generate([{"prompt_token_ids": (P[i] + t[:a.max_tokens - 16] + tok.encode("\n<Final answer>", add_special_tokens=False))[:a.max_model_len - 2]} for i, _, t in redo],
                                 SamplingParams(temperature=0.0, max_tokens=1, logprobs=20), lora_request=lora_req, use_tqdm=False)
            for (i, j, _), o in zip(redo, outs2): probs_all[i][j] = p_yes_from_logprobs(o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}, yes_ids, no_ids)
        probs = [float(np.mean(pp)) for pp in probs_all]
        score_texts.last_samples = probs_all
        return probs, parsed, resp

    ms = {}; preds = {}; interp = {}
    for split in a.splits.split(","):
        recs = load_timeline(a.lead, split, a.max_eval); t1 = time.time(); probs, parsed, resp = score_texts([r["text"] for r in recs], [r["person_id"] for r in recs])
        y = np.array([r["label"] for r in recs]); s = np.array(probs, dtype=float); ms[split] = metrics(y, s, subgroups(recs)); preds[split] = pd.DataFrame({"person_id": [r["person_id"] for r in recs], "label": y, "prob": s})
        print(f"lead {a.lead} {a.tag} {split}: AUROC {ms[split]['auroc']:.4f} AP {ms[split]['auprc']:.4f} sens@sp90 {ms[split]['sens_at_spec90']:.3f} ppv@5 {ms[split]['ppv_top5']:.3f} ece {ms[split]['ece']:.3f} | n={len(y)} {time.time()-t1:.0f}s", flush=True)
        rows = []
        samples = getattr(score_texts, "last_samples", [[p] for p in probs])
        for r, p, pr, t, sp_ in zip(recs, probs, parsed, resp, samples):
            ex = expert[r["person_id"]]["top"]; jac = jaccard([c for c in pr["concepts"] if c not in ALIGN_EXCLUDE], [c for c in ex if c not in ALIGN_EXCLUDE])
            rows.append({"person_id": r["person_id"], "label": r["label"], "prob": p, "valid": pr["valid"], "n_sub": pr["n_sub"], "concepts": pr["concepts"], "ungrounded": pr["ungrounded"], "unknown": pr["unknown_features"],
                         "expert_top": ex, "expert_prob": expert[r["person_id"]]["prob"], "jaccard": jac, "response": t, "sample_probs": sp_})
        with open(rdir / f"trajectories_{split}.jsonl", "w") as f:
            for row in rows: f.write(json.dumps(row) + "\n")
        rep = report(rows, recs, score_texts, a.faithfulness, split == "test", a.n_clusters, a.seed)
        interp[split] = rep
        print(f"  {split}: valid {rep['valid_rate']:.2f} | jaccard vs expert {rep['mean_jaccard_expert']} | grounding {rep['grounding_rate']:.2f} | spearman(prob, expert) {rep['spearman_prob_vs_expert']:.3f}", flush=True)
        if "concepts" in rep:
            top_c = sorted(rep["concepts"].items(), key=lambda kv: -kv[1]["cite_rate"])[:12]
            print("  most cited: " + ", ".join(f"{c} {v['cite_rate']:.2f} (lift@top10 {v['lift_top10_vs_all'] if v['lift_top10_vs_all'] is None else round(v['lift_top10_vs_all'],2)})" for c, v in top_c), flush=True)
        for ph in rep.get("phenotypes_top20", []): print(f"  phenotype {ph['cluster']}: n={ph['n']} pos_rate={ph['positive_rate']:.2f} p={ph['mean_prob']:.2f} " + "; ".join(f"{c} {f} (x{l})" for c, f, l in ph["top_concepts"][:5]), flush=True)
    (rdir / "interpretability.json").write_text(json.dumps(interp, indent=1, default=str))
    if a.no_registry:
        (rdir / "metrics_smoke.json").write_text(json.dumps({"metrics": ms, "interp": interp}, indent=1, default=str)); print("no-registry: metrics written to metrics_smoke.json"); return
    save_run(run, {"lead": a.lead, "model": a.model, "adapter": a.adapter, "tag": a.tag, "scoring": "P(Yes) at the token after <Final answer> of the greedy structured trajectory; invalid -> tag appended", "max_tokens": a.max_tokens,
                   "max_prompt_tokens": a.max_prompt_tokens, "expert_opinion": a.expert_opinion, "care_context": a.care_context, "n_samples": a.n_samples, "sample_temp": a.sample_temp, "include_greedy": a.include_greedy}, ms, preds, extra={"interpretability_summary": {s: {k: v for k, v in interp[s].items() if k in ("valid_rate", "mean_jaccard_expert", "grounding_rate", "spearman_prob_vs_expert", "faithfulness_deletion")} for s in interp}})
    print(f"done in {(time.time()-t0)/60:.1f} min -> {rdir}")


if __name__ == "__main__":
    main()
