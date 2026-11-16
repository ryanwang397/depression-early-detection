#!/usr/bin/env python3
"""Stage 1: LoRA-fine-tuned LLM classifier over evidence timelines (LLM-native scoring: log-odds of the answer tokens Yes vs No).

  python run_llm_lora_cls.py --lead 12 --model medgemma-27b [--max-train N --max-eval N] [--tag lora] [--epochs 1]
Training: left-padded chat prompts; loss = class-weighted 2-way cross-entropy over the logits of the "Yes"/"No" tokens at the
next-token position (equivalent to logistic regression on the Yes/No log-odds); LoRA on attention+MLP projections; bf16;
gradient checkpointing. Model selection: dev-subset AUPRC every --eval-every optimizer steps; best adapter is re-scored on full
dev + test and saved via exp_common.save_run (results/lead{L}_{tag}_{model}_lora/)."""
from __future__ import annotations
import argparse, os, time, json, math, random
import numpy as np, pandas as pd, torch
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from exp_common import metrics, save_run, RESULTS
from llm_common import model_dir, load_timeline, build_messages
from run_llm_zeroshot import subgroups


def encode(tok, recs, lead, max_len):
    out = []
    for r in recs:
        p = tok.apply_chat_template(build_messages(r["text"], int(r.get("lead_months") or lead)), tokenize=False, add_generation_prompt=True)   # multi-cutoff rows carry their own horizon
        ids = tok.encode(p, add_special_tokens=False)
        if len(ids) > max_len:
            ids = ids[:1200] + tok.encode("\n...[truncated]...\n", add_special_tokens=False) + ids[-(max_len - 1200 - 8):]
        out.append(ids)
    return out


def batches(ids_list, labels, order, bs, pad_id):
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]; seqs = [ids_list[j] for j in idx]; L = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), L), pad_id, dtype=torch.long); attn = torch.zeros((len(seqs), L), dtype=torch.long)
        for k, s in enumerate(seqs):   # left padding -> last position is the next-token slot for every row
            input_ids[k, L - len(s):] = torch.tensor(s); attn[k, L - len(s):] = 1
        yield input_ids, attn, torch.tensor([labels[j] for j in idx], dtype=torch.float), idx


@torch.no_grad()
def score(model, tok, ids_list, labels, yes_id, no_id, bs, desc=""):
    model.eval(); order = sorted(range(len(ids_list)), key=lambda j: len(ids_list[j])); out = np.zeros(len(ids_list)); t0 = time.time()
    for input_ids, attn, _, idx in batches(ids_list, labels, order, bs, tok.pad_token_id):
        logits = model(input_ids=input_ids.cuda(), attention_mask=attn.cuda(), use_cache=False).logits[:, -1, :]
        d = (logits[:, yes_id] - logits[:, no_id]).float().cpu().numpy(); out[idx] = 1 / (1 + np.exp(-d))
    model.train(); print(f"    scored {len(ids_list)} {desc} in {time.time()-t0:.0f}s", flush=True); return out


def _save_best(model, rdir, step, m, tries=6, wait=120):
    """Atomically replace rdir/adapter_best (write to .tmp, then rename); retry on transient I/O errors such as a full group
    disk quota instead of crashing a multi-hour run. Returns False (previous adapter_best kept) if every attempt fails."""
    import shutil
    tmp, dst, old = rdir / "adapter_best.tmp", rdir / "adapter_best", rdir / "adapter_best.old"
    for i in range(tries):
        try:
            shutil.rmtree(tmp, ignore_errors=True); shutil.rmtree(old, ignore_errors=True)
            model.save_pretrained(tmp)
            if dst.exists(): dst.rename(old)
            tmp.rename(dst); shutil.rmtree(old, ignore_errors=True)
            (rdir / "metrics_provisional.json").write_text(json.dumps({"step": step, "dev_subset": m}, indent=2))
            return True
        except OSError as e:
            if not dst.exists() and old.exists(): old.rename(dst)
            print(f"  [save] attempt {i+1}/{tries} failed at step {step}: {e}; retrying in {wait}s", flush=True); time.sleep(wait)
    print(f"  [save] giving up at step {step}; keeping previous adapter_best", flush=True); return False


def _retry_io(fn, what, tries=6, wait=120):
    for i in range(tries):
        try: return fn()
        except OSError as e: print(f"  [{what}] attempt {i+1}/{tries} failed: {e}; retrying in {wait}s", flush=True); time.sleep(wait)
    raise RuntimeError(f"{what}: all {tries} attempts failed")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lead", type=int, default=12); ap.add_argument("--model", default="medgemma-27b"); ap.add_argument("--tag", default="lora")
    ap.add_argument("--max-train", type=int, default=None); ap.add_argument("--max-eval", type=int, default=None); ap.add_argument("--dev-subset", type=int, default=2000)
    ap.add_argument("--epochs", type=float, default=1.0); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--batch", type=int, default=2); ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=8); ap.add_argument("--max-len", type=int, default=4096); ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--input", default="timeline", choices=["timeline", "xml"], help="timeline = L4 evidence timeline (default); xml = pre-L4 representation: exp_common.build_text(notes_ehr) over the dataset JSONL (same input as the ModernBERT baseline)")
    ap.add_argument("--lora-r", type=int, default=16); ap.add_argument("--lora-alpha", type=int, default=32); ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--pos-weight", type=float, default=None, help="weight of positives in the 2-way CE; default = n_neg/n_pos (balanced)"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn", default="sdpa"); ap.add_argument("--init-adapter", default=None, help="path to a saved LoRA adapter to continue training from (e.g. results/<run>/adapter_best)")
    ap.add_argument("--extra-train-files", default="", help="comma-separated timeline JSONL files appended to the training set (multi-cutoff training: same patients at other leads); persons present in dev/test are dropped"); ap.add_argument("--tl-suffix", default="", help="timeline file suffix (e.g. _x3 for the 3-excerpt timelines from s7 --suffix _x3)"); ap.add_argument("--extra-blocks", default="", help="comma list of extra information lines appended to every timeline: ctx (care-context coding propensity, outputs/eagrl/lead{L}_context.parquet), ses (payer/smoking, outputs/eagrl/lead{L}_ses.parquet)"); a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed); t0 = time.time()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel
    md = model_dir(a.model); run = f"lead{a.lead}_{a.tag}_{a.model}_lora"; rdir = RESULTS / run; rdir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(md); tok.padding_side = "left"
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    yes_id = tok.encode("Yes", add_special_tokens=False); no_id = tok.encode("No", add_special_tokens=False); assert len(yes_id) == 1 and len(no_id) == 1, (yes_id, no_id); yes_id, no_id = yes_id[0], no_id[0]
    load_tl = load_timeline   # local alias: the two overrides below would otherwise make `load_timeline` a local name and shadow the import (UnboundLocalError on the default path)
    if a.tl_suffix:
        import llm_common as _lc; _orig = _lc.load_timeline
        def load_tl(lead, split, max_n=None, seed_salt="subset"): return _orig(lead, f"{split}{a.tl_suffix}", max_n, seed_salt)
    if a.input == "xml":   # pre-L4 ablation: recent-first packed notes + per-encounter EHR lines, no LLM-extracted evidence
        from exp_common import load_split, build_text
        def load_tl(lead, split, max_n=None, seed_salt="subset"):
            out = []
            for r in load_split(lead, split, max_n, keep_text=True, seed_salt=seed_salt):
                out.append({"person_id": r["person_id"], "label": r["label"], "sex": r.get("sex"), "age": r.get("age_at_cutoff"), "age_at_cutoff": r.get("age_at_cutoff"),
                            "n_notes": (r.get("features") or {}).get("n_notes", 0), "features": {"n_notes": (r.get("features") or {}).get("n_notes", 0)},
                            "text": build_text(r, "notes_ehr", max_chars=int(a.max_len * 3.6), header=True)})
            print(f"[xml] {split}: {len(out)} records, text chars median {int(np.median([len(o['text']) for o in out]))}", flush=True); return out
    recs = {"train": load_tl(a.lead, "train", a.max_train), "dev": load_tl(a.lead, "dev", a.max_eval), "test": load_tl(a.lead, "test", a.max_eval)}
    if a.extra_train_files:
        held = {r["person_id"] for s_ in ("dev", "test") for r in recs[s_]}; n0 = len(recs["train"])
        for f in [x for x in a.extra_train_files.split(",") if x]:
            ex = [json.loads(l) for l in open(f)]; ex = [r for r in ex if r["person_id"] not in held]
            recs["train"].extend(ex); print(f"[extra-train] {f}: +{len(ex)} rows (lead {ex[0].get('lead_months') if ex else '?'})", flush=True)
        random.Random(a.seed).shuffle(recs["train"]); print(f"[extra-train] train {n0} -> {len(recs['train'])} rows, pos {sum(r['label'] for r in recs['train'])}", flush=True)
    blocks = [b for b in a.extra_blocks.split(",") if b]
    if blocks:   # v3 information blocks (see eagrl/README_eagrl.md): appended as extra lines at the end of the timeline text
        from exp_common import ROOT as _ROOT
        lines = {}
        if "ctx" in blocks:
            cdf = pd.read_parquet(_ROOT / "outputs" / "eagrl" / f"lead{a.lead}_context.parquet", columns=["person_id", "context_text"])
            for r_ in cdf.itertuples(): lines.setdefault(int(r_.person_id), []).append("CARE CONTEXT (how often patients seen in these settings or by this clinician receive a depression code within the prediction window; reflects screening and coding practice rather than the patient): " + r_.context_text)
        if "ses" in blocks:
            sdf = pd.read_parquet(_ROOT / "outputs" / "eagrl" / f"lead{a.lead}_ses.parquet", columns=["person_id", "ses_text"])
            for r_ in sdf.itertuples(): lines.setdefault(int(r_.person_id), []).append(r_.ses_text)
        n_hit = 0
        for s_ in recs:
            for r in recs[s_]:
                if r["person_id"] in lines: r["text"] = r["text"] + "\n" + "\n".join(lines[r["person_id"]]); n_hit += 1
        print(f"extra blocks {blocks}: appended to {n_hit} records", flush=True)
    ids = {s: encode(tok, recs[s], a.lead, a.max_len) for s in recs}; y = {s: [r["label"] for r in recs[s]] for s in recs}
    dev_sub = load_tl(a.lead, "dev", min(a.dev_subset, len(recs["dev"])))
    if blocks:
        for r in dev_sub:
            if r["person_id"] in lines: r["text"] = r["text"] + "\n" + "\n".join(lines[r["person_id"]])
    ids_sub = encode(tok, dev_sub, a.lead, a.max_len); y_sub = [r["label"] for r in dev_sub]
    lens = [len(x) for x in ids["train"]]; print(f"[{run}] sizes {[len(recs[s]) for s in recs]}; train tokens median {int(np.median(lens))} p95 {int(np.percentile(lens, 95))} max {max(lens)}; prep {time.time()-t0:.0f}s", flush=True)
    model = AutoModelForCausalLM.from_pretrained(md, torch_dtype=torch.bfloat16, attn_implementation=a.attn, device_map={"": 0})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads(); model.config.use_cache = False
    if a.init_adapter:
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True); print(f"continuing from adapter {a.init_adapter}", flush=True)
    else:
        model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    n_pos = sum(y["train"]); pw = a.pos_weight if a.pos_weight else (len(y["train"]) - n_pos) / max(n_pos, 1); cw = torch.tensor([1.0, pw]).cuda()
    params = [p for p in model.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(ids["train"]) / (a.batch * a.grad_accum)); total = int(steps_per_epoch * a.epochs); warm = max(int(0.03 * total), 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(total - warm, 1))))))
    print(f"train {len(ids['train'])} (pos {n_pos}, pos_weight {pw:.2f}); {total} optimizer steps (eff. batch {a.batch*a.grad_accum}); eval every {a.eval_every}", flush=True)
    best = (-1.0, None); step = 0; micro = 0; run_loss = 0.0; t1 = time.time(); model.train(); done = False; ep = 0
    while not done:
        rng = np.random.default_rng(a.seed + ep); order = rng.permutation(len(ids["train"])).tolist()
        # length-bucketed shuffling: sort within chunks of 64 to reduce padding, then shuffle chunk order
        chunks = [sorted(order[i:i + 64], key=lambda j: len(ids["train"][j])) for i in range(0, len(order), 64)]; rng.shuffle(chunks); order = [j for c in chunks for j in c]
        for input_ids, attn, yb, _ in batches(ids["train"], y["train"], order, a.batch, tok.pad_token_id):
            logits = model(input_ids=input_ids.cuda(), attention_mask=attn.cuda(), use_cache=False).logits[:, -1, :]
            two = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1).float()      # class 0 = No, class 1 = Yes
            loss = torch.nn.functional.cross_entropy(two, yb.long().cuda(), weight=cw) / a.grad_accum; loss.backward(); run_loss += loss.item(); micro += 1
            if micro % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 20 == 0:
                    print(f"ep {ep} step {step}/{total} loss {run_loss/20:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t1:.0f}s mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True); run_loss = 0.0
                if step % a.eval_every == 0 or step == total:
                    s_sub = score(model, tok, ids_sub, y_sub, yes_id, no_id, a.eval_batch, "dev-subset"); m = metrics(y_sub, s_sub, n_boot=0)
                    print(f"  [eval step {step}] dev-subset AUROC {m['auroc']:.4f} AP {m['auprc']:.4f}", flush=True)
                    if m["auprc"] > best[0]:
                        if _save_best(model, rdir, step, m): best = (m["auprc"], step)
                if step >= total: done = True; break
        ep += 1
    print(f"training done in {(time.time()-t1)/3600:.2f} h; best dev-subset AP {best[0]:.4f} at step {best[1]}", flush=True)
    if best[1] is not None and best[1] != step:
        model.load_adapter(str(rdir / "adapter_best"), adapter_name="best"); model.set_adapter("best")
    preds = {}; ms = {}
    for s in ["dev", "test"]:
        p = score(model, tok, ids[s], y[s], yes_id, no_id, a.eval_batch, s); ms[s] = metrics(y[s], p, subgroups(recs[s])); preds[s] = pd.DataFrame({"person_id": [r["person_id"] for r in recs[s]], "label": y[s], "prob": p})
        print(f"lead {a.lead} {a.model} LoRA {s}: AUROC {ms[s]['auroc']:.4f} AP {ms[s]['auprc']:.4f} sens@sp90 {ms[s]['sens_at_spec90']:.3f} ppv@5 {ms[s]['ppv_top5']:.3f} ece {ms[s]['ece']:.3f}", flush=True)
    _retry_io(lambda: save_run(run, {"lead": a.lead, "model": a.model, "model_dir": md, "tag": a.tag, "n_train": len(ids["train"]), "epochs": a.epochs, "lr": a.lr, "eff_batch": a.batch * a.grad_accum, "max_len": a.max_len, "lora_r": a.lora_r, "pos_weight": pw, "best_step": best[1], "init_adapter": a.init_adapter, "extra_blocks": a.extra_blocks, "tl_suffix": a.tl_suffix, "scoring": "logodds(Yes,No) next-token; 2-way weighted CE"}, ms, preds), "save_run")
    print(f"done in {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
