#!/usr/bin/env python3
"""e5: direct (no-reasoning) Yes/No log-odds scoring with an adapter, using the classifier prompt (llm_common.build_messages).

  python e5_direct_score.py --lead 3 --adapter <dir> --splits dev,test --tag sft_direct
  python e5_direct_score.py --lead 3 --adapter <lora2 dir> --splits train --pids-file outputs/eagrl/lead3_sft.jsonl --tag lora2_sftpat
Diagnostic 1: does the Stage-1 SFT adapter still rank well when asked directly (is the direct head intact)?
Diagnostic 2 / soft targets: classifier probabilities for the SFT patients (train) to be used as distillation targets."""
from __future__ import annotations
import argparse, json, os, time
import numpy as np, pandas as pd
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from common import set_tl_suffix, set_knowledge_card, set_align_exclude_v3, RESULTS, resolve_path, model_dir, load_timeline, yes_no_token_ids, metrics, p_yes_from_logprobs
from llm_common import build_messages as cls_messages


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lead", type=int, default=3); ap.add_argument("--model", default="llama-3.1-8b"); ap.add_argument("--adapter", default=None); ap.add_argument("--tag", required=True)
    ap.add_argument("--splits", default="dev,test"); ap.add_argument("--pids-file", default=None); ap.add_argument("--max-eval", type=int, default=None); ap.add_argument("--max-model-len", type=int, default=4608)
    ap.add_argument("--gpu-mem", type=float, default=0.9); ap.add_argument("--extra-blocks", default=""); ap.add_argument("--tl-suffix", default=""); ap.add_argument("--knowledge-card", action="store_true"); ap.add_argument("--align-v3", action="store_true"); a = ap.parse_args(); set_tl_suffix(a.tl_suffix); set_knowledge_card(a.knowledge_card)
    if a.align_v3: set_align_exclude_v3()
    a.adapter = resolve_path(a.adapter); a.pids_file = resolve_path(a.pids_file); t0 = time.time()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    kw = {}; lora_req = None
    if a.adapter:
        r = json.load(open(os.path.join(a.adapter, "adapter_config.json")))["r"]; kw = {"enable_lora": True, "max_lora_rank": max(r, 8), "max_loras": 1}; lora_req = LoRARequest("d", 1, a.adapter)
    llm = LLM(model=model_dir(a.model), dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, seed=0, max_num_seqs=256, **kw)
    tok = llm.get_tokenizer(); yes_ids, no_ids = yes_no_token_ids(tok); keep = {json.loads(l)["person_id"] for l in open(a.pids_file)} if a.pids_file else None
    out = []; (RESULTS / "eagrl").mkdir(parents=True, exist_ok=True); lines = {}
    if a.extra_blocks:
        import pandas as _pd
        from common import ROOT as _ROOT
        if "ctx" in a.extra_blocks:
            for r_ in _pd.read_parquet(_ROOT / "outputs" / "eagrl" / f"lead{a.lead}_context.parquet", columns=["person_id", "context_text"]).itertuples(): lines.setdefault(int(r_.person_id), []).append("CARE CONTEXT (how often patients seen in these settings or by this clinician receive a depression code within the prediction window; reflects screening and coding practice rather than the patient): " + r_.context_text)
        if "ses" in a.extra_blocks:
            for r_ in _pd.read_parquet(_ROOT / "outputs" / "eagrl" / f"lead{a.lead}_ses.parquet", columns=["person_id", "ses_text"]).itertuples(): lines.setdefault(int(r_.person_id), []).append(r_.ses_text)
    for split in a.splits.split(","):
        recs = load_timeline(a.lead, split, a.max_eval)
        if keep: recs = [r for r in recs if r["person_id"] in keep]
        prompts = []
        for r in recs:
            txt = r["text"] + ("\n" + "\n".join(lines[r["person_id"]]) if r["person_id"] in lines else "")
            p = tok.apply_chat_template(cls_messages(txt, a.lead), tokenize=False, add_generation_prompt=True); ids = tok.encode(p, add_special_tokens=False)
            if len(ids) > a.max_model_len - 8: ids = ids[:1200] + tok.encode("\n...[truncated]...\n", add_special_tokens=False) + ids[-(a.max_model_len - 8 - 1200 - 12):]
            prompts.append({"prompt_token_ids": ids})
        outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1, logprobs=20), lora_request=lora_req, use_tqdm=False)
        s = np.array([p_yes_from_logprobs(o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}, yes_ids, no_ids) for o in outs]); y = np.array([r["label"] for r in recs])
        if len(set(y)) == 2: m = metrics(y, s, n_boot=0); print(f"[{a.tag}] {split}: n={len(y)} AUROC {m['auroc']:.4f} AP {m['auprc']:.4f} ece {m['ece']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        out += [{"person_id": r["person_id"], "split": split, "label": int(r["label"]), "p_direct": float(v)} for r, v in zip(recs, s)]
    pd.DataFrame(out).to_csv(RESULTS / "eagrl" / f"direct_{a.tag}.csv", index=False); print(f"wrote results/eagrl/direct_{a.tag}.csv ({len(out)} rows); done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
