#!/usr/bin/env python3
"""e2: Stage-1 trajectory-level SFT (LoRA) with an extra answer-position objective.

  python e2_sft.py --lead 3 --model llama-3.1-8b --init-adapter results/lead3_lora1_llama-3.1-8b_lora/adapter_best --epochs 2
loss = LM cross-entropy on the response tokens (reasoning + tags + answer) + lambda_ans * 2-way CE over the Yes/No logits at the
token after "<Final answer>" (keeps the classifier head trained on the label, as in experiments/run_llm_lora_cls.py).
Starting from the LoRA classifier adapter (Stage 0) is the default warm start; --lora-r > 0 without --init-adapter trains a new adapter.
Writes results/eagrl/lead{L}_{tag}_{model}/adapter_final (+ adapter_ep{k}) and train_log.json."""
from __future__ import annotations
import argparse, json, math, os, random, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from common import set_expert_variant, set_tl_suffix, set_knowledge_card, set_align_exclude_v3, OUT, RESULTS, resolve_path, model_dir, yes_no_token_ids, build_messages, encode_prompt, token_spans, metrics, load_expert, load_context, prompt_extras
from llm_common import build_messages as cls_messages


def build_examples(tok, rows, lead, max_prompt_tokens, expert=None, context=None, soft=None, soft_mix=1.0, direct=False):
    """Reasoning examples (LM loss on the response + answer-token loss) and, with direct=True, one extra direct-head example per
    patient (classifier prompt, answer = first generated token, no LM loss).  q = soft_mix * classifier prob + (1 - soft_mix) * label."""
    out = []; eos = tok.eos_token_id
    for r in rows:
        y = int(r["label"]); q = y if (soft is None or r["person_id"] not in soft) else soft_mix * soft[r["person_id"]] + (1 - soft_mix) * y
        p = encode_prompt(tok, build_messages(r["text"], lead, **prompt_extras(r["person_id"], expert, context)), max_prompt_tokens); resp = tok.encode(r["response"], add_special_tokens=False) + [eos]
        ans, _ = token_spans(tok, resp)
        if ans is None: continue
        out.append({"ids": p + resp, "plen": len(p), "ans": len(p) + ans, "label": y, "q": float(q), "person_id": r["person_id"], "direct": False})
        if direct:
            pd_ = encode_prompt(tok, cls_messages(r["text"], lead), max_prompt_tokens)
            out.append({"ids": pd_, "plen": len(pd_), "ans": len(pd_), "label": y, "q": float(q), "person_id": r["person_id"], "direct": True})
    return out


def collate(exs, pad_id):
    L = max(len(e["ids"]) for e in exs); B = len(exs)
    ids = torch.full((B, L), pad_id, dtype=torch.long); attn = torch.zeros((B, L), dtype=torch.long); lab = torch.full((B, L), -100, dtype=torch.long)
    for k, e in enumerate(exs):
        n = len(e["ids"]); ids[k, :n] = torch.tensor(e["ids"]); attn[k, :n] = 1; lab[k, e["plen"]:n] = torch.tensor(e["ids"][e["plen"]:])
    return ids, attn, lab, torch.tensor([e["ans"] for e in exs]), torch.tensor([e["label"] for e in exs]), torch.tensor([e["q"] for e in exs], dtype=torch.float), torch.tensor([e["direct"] for e in exs])


def two_way(logits_rows, yes_ids, no_ids):
    """[B, V] -> [B, 2] (No, Yes) logits via logsumexp over the token variants."""
    return torch.stack([torch.logsumexp(logits_rows[:, no_ids].float(), 1), torch.logsumexp(logits_rows[:, yes_ids].float(), 1)], 1)


def losses(model, batch, yes_ids, no_ids):
    """lm = token-mean LM loss over response tokens (0 if none in the batch); ce = soft two-way CE at the answer token:
    -(q log p_yes + (1-q) log p_no); direct examples contribute only to ce."""
    ids, attn, lab, ans, y, q, direct = [t.cuda() for t in batch]
    logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits
    tgt = lab[:, 1:].reshape(-1); n_lm = int((tgt != -100).sum())
    lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), tgt, ignore_index=-100) if n_lm > 0 else logits.sum() * 0.0
    two = two_way(logits[torch.arange(len(ans)), ans - 1], yes_ids, no_ids); lp = torch.log_softmax(two, 1)
    ce = -(q * lp[:, 1] + (1 - q) * lp[:, 0]).mean(); p = torch.softmax(two, 1)[:, 1]
    return lm, ce, p.detach()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lead", type=int, default=3); ap.add_argument("--model", default="llama-3.1-8b"); ap.add_argument("--tag", default="sft"); ap.add_argument("--data", default=None)
    ap.add_argument("--init-adapter", default=None); ap.add_argument("--lora-r", type=int, default=32); ap.add_argument("--lora-alpha", type=int, default=64); ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--batch", type=int, default=2); ap.add_argument("--grad-accum", type=int, default=8); ap.add_argument("--max-prompt-tokens", type=int, default=3300)
    ap.add_argument("--lambda-ans", type=float, default=2.0); ap.add_argument("--max-examples", type=int, default=None); ap.add_argument("--holdout", type=float, default=0.05); ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--attn", default="sdpa"); ap.add_argument("--expert-opinion", action="store_true"); ap.add_argument("--care-context", action="store_true")
    ap.add_argument("--soft-targets", default=None, help="CSV with person_id,p_direct (classifier probabilities) used as answer-token targets"); ap.add_argument("--soft-mix", type=float, default=1.0, help="weight of the soft target vs the hard label")
    ap.add_argument("--direct-multitask", action="store_true", help="add one direct-head example (classifier prompt) per patient"); ap.add_argument("--expert-variant", default=None, help="suffix of the e0 expert file to use, e.g. _inc (incident-only, prior-evidence gated)"); ap.add_argument("--tl-suffix", default=""); ap.add_argument("--knowledge-card", action="store_true"); ap.add_argument("--align-v3", action="store_true"); a = ap.parse_args(); set_tl_suffix(a.tl_suffix); set_knowledge_card(a.knowledge_card)
    if a.expert_variant: set_expert_variant(a.expert_variant)
    if a.align_v3: set_align_exclude_v3()
    a.init_adapter = resolve_path(a.init_adapter); a.data = resolve_path(a.data)
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed); t0 = time.time()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel
    md = model_dir(a.model); rdir = RESULTS / "eagrl" / f"lead{a.lead}_{a.tag}_{a.model}"; rdir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(md)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    yes_ids, no_ids = yes_no_token_ids(tok)
    rows = [json.loads(l) for l in open(a.data or OUT / f"lead{a.lead}_sft.jsonl")]
    if a.max_examples: rows = rows[: a.max_examples]
    expert = load_expert(a.lead) if a.expert_opinion else None; context = load_context(a.lead) if a.care_context else None
    soft = None
    if a.soft_targets:
        sp = pd.read_csv(resolve_path(a.soft_targets)); soft = dict(zip(sp.person_id.astype(int), sp.p_direct.astype(float))); print(f"soft targets: {len(soft)} patients, soft_mix {a.soft_mix}", flush=True)
    exs = build_examples(tok, rows, a.lead, a.max_prompt_tokens, expert, context, soft, a.soft_mix, a.direct_multitask); random.shuffle(exs)
    n_hold = max(int(len(exs) * a.holdout), 4) if len(exs) >= 8 else 0; hold, train = exs[:n_hold], exs[n_hold:]
    lens = [len(e["ids"]) for e in train]; print(f"[e2] {len(rows)} rows -> {len(exs)} examples (train {len(train)}, holdout {len(hold)}; direct {sum(e['direct'] for e in exs)}); tokens median {int(np.median(lens))} max {max(lens)}; pos {sum(e['label'] for e in train)}; mean q {np.mean([e['q'] for e in train]):.3f}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(md, torch_dtype=torch.bfloat16, attn_implementation=a.attn, device_map={"": 0})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads(); model.config.use_cache = False
    if a.init_adapter:
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True); print(f"warm start from {a.init_adapter}", flush=True)
    else:
        model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train) / (a.batch * a.grad_accum)); total = max(int(steps_per_epoch * a.epochs), 1); warm = max(int(0.05 * total), 2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(total - warm, 1))))))
    print(f"{total} optimizer steps (eff. batch {a.batch*a.grad_accum}), lambda_ans {a.lambda_ans}", flush=True)

    @torch.no_grad()
    def evaluate(exs_):
        model.eval(); ps = []; ys = []; ce_sum = 0.0; lm_sum = 0.0
        for i in range(0, len(exs_), a.batch):
            b = exs_[i:i + a.batch]; lm, ce, p = losses(model, collate(b, tok.pad_token_id), yes_ids, no_ids); ps += p.tolist(); ys += [e["label"] for e in b]; ce_sum += ce.item() * len(b); lm_sum += lm.item() * len(b)
        model.train(); m = metrics(ys, ps, n_boot=0) if len(set(ys)) == 2 else {"auroc": float("nan"), "auprc": float("nan")}
        return {"lm": lm_sum / max(len(exs_), 1), "ans_ce": ce_sum / max(len(exs_), 1), "auroc": m["auroc"], "auprc": m["auprc"]}

    log = []; step = 0; micro = 0; acc = {"lm": 0.0, "ce": 0.0, "n": 0}; t1 = time.time(); model.train(); ep = 0; done = False
    while not done:
        rng = np.random.default_rng(a.seed + ep); order = rng.permutation(len(train)).tolist()
        chunks = [sorted(order[i:i + 32], key=lambda j: len(train[j]["ids"])) for i in range(0, len(order), 32)]; rng.shuffle(chunks); order = [j for c in chunks for j in c]
        for i in range(0, len(order), a.batch):
            b = [train[j] for j in order[i:i + a.batch]]; lm, ce, _ = losses(model, collate(b, tok.pad_token_id), yes_ids, no_ids)
            loss = (lm + a.lambda_ans * ce) / a.grad_accum; loss.backward(); acc["lm"] += lm.item(); acc["ce"] += ce.item(); acc["n"] += 1; micro += 1
            if micro % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 10 == 0 or step == total:
                    rec = {"step": step, "lm": acc["lm"] / max(acc["n"], 1), "ans_ce": acc["ce"] / max(acc["n"], 1), "lr": sched.get_last_lr()[0], "sec": round(time.time() - t1), "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
                    print(f"ep {ep} step {step}/{total} lm {rec['lm']:.4f} ans_ce {rec['ans_ce']:.4f} lr {rec['lr']:.2e} {rec['sec']}s mem {rec['mem_gb']}GB", flush=True); acc = {"lm": 0.0, "ce": 0.0, "n": 0}; log.append(rec)
                if hold and (step % a.eval_every == 0 or step == total):
                    ev = evaluate(hold); print(f"  [holdout step {step}] lm {ev['lm']:.4f} ans_ce {ev['ans_ce']:.4f} AUROC {ev['auroc']:.4f} AP {ev['auprc']:.4f}", flush=True); log.append({"step": step, "holdout": ev})
                if step >= total: done = True; break
        ep += 1; model.save_pretrained(rdir / f"adapter_ep{ep}")
    model.save_pretrained(rdir / "adapter_final")
    (rdir / "train_log.json").write_text(json.dumps({"args": vars(a), "n_train": len(train), "n_holdout": len(hold), "steps": total, "log": log, "hours": (time.time() - t0) / 3600}, indent=1, default=str))
    print(f"saved {rdir/'adapter_final'}; done in {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
