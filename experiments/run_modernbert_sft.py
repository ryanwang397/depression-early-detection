#!/usr/bin/env python3
"""Clinical ModernBERT (8k BERT encoder) supervised fine-tuning as a patient-level classifier.

Input = header + recent-first packed notes (optionally with per-encounter EHR summaries); tokens are
truncated from the LEFT so the most recent content is always kept. Mean pooling -> linear head.
  python run_modernbert_sft.py --lead 12 --mode notes --max-length 8192 --tuning full \
      --epochs 3 --lr 2e-5 --batch 8 --grad-accum 2 --max-train 3000 --max-eval 800 --tag smoke
"""
from __future__ import annotations
import argparse, glob, json, math, time, os, random
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from exp_common import load_split, build_text, metrics, subgroup_fields, save_run, RESULTS

MODEL_DIR = glob.glob(os.environ.get("DED_HF_HUB", os.path.expanduser("~/.cache/huggingface/hub")) + "/models--Simonlee711--Clinical_ModernBERT/snapshots/*/")[0]


class Enc(Dataset):
    def __init__(self, recs, tok, mode, max_length):
        self.ids = []; self.y = []
        cls, sep = tok.cls_token_id, tok.sep_token_id
        for r in recs:
            text = build_text(r, mode, max_chars=max_length * 7, header=True)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) > max_length - 2:
                ids = ids[-(max_length - 2):]           # keep the most recent tokens
            self.ids.append([cls] + ids + [sep]); self.y.append(r["label"])
        self.lens = [len(x) for x in self.ids]

    def __len__(self): return len(self.ids)
    def __getitem__(self, i): return self.ids[i], self.y[i]


def collate(batch, pad_id):
    L = max(len(x) for x, _ in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long); am = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (x, _) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x); am[i, :len(x)] = 1
    return ids, am, torch.tensor([y for _, y in batch], dtype=torch.float32)


class Clf(nn.Module):
    def __init__(self, enc, hidden, dropout=0.1):
        super().__init__(); self.enc = enc; self.drop = nn.Dropout(dropout); self.head = nn.Linear(hidden, 1)

    def forward(self, ids, am):
        h = self.enc(input_ids=ids, attention_mask=am).last_hidden_state
        m = am.unsqueeze(-1).to(h.dtype); pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(self.drop(pooled.float())).squeeze(-1)


@torch.no_grad()
def predict(model, ds, pad_id, batch, device):
    model.eval(); order = np.argsort(ds.lens)[::-1]; out = np.zeros(len(ds))
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]; ids, am, _ = collate([ds[i] for i in idx], pad_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids.to(device), am.to(device))
        out[idx] = torch.sigmoid(logits.float()).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=int, default=12); ap.add_argument("--mode", default="notes", choices=["notes", "notes_ehr", "ehr"])
    ap.add_argument("--max-length", type=int, default=8192); ap.add_argument("--tuning", default="full", choices=["full", "lora", "frozen"])
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--grad-accum", type=int, default=2); ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--class-weight", default="balanced", choices=["balanced", "none"]); ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01); ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-train", type=int, default=None); ap.add_argument("--max-eval", type=int, default=None); ap.add_argument("--tag", default="full")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--no-grad-ckpt", action="store_true"); ap.add_argument("--eval-steps", type=int, default=0, help="evaluate dev every N optimizer steps (0 = per epoch)")
    ap.add_argument("--patience", type=int, default=0, help="stop after this many dev evaluations without AP improvement (0 = off)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    run_name = f"lead{a.lead}_{a.tag}_modernbert_{a.tuning}_{a.mode}_L{a.max_length}"
    device = torch.device("cuda"); t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    recs = {s: load_split(a.lead, s, a.max_train if s == "train" else a.max_eval) for s in ["train", "dev", "test"]}
    ds = {s: Enc(recs[s], tok, a.mode, a.max_length) for s in recs}
    groups = {s: subgroup_fields(recs[s]) for s in recs}
    for s in recs:  # drop raw text to keep RSS small
        recs[s] = [{"person_id": r["person_id"], "label": r["label"]} for r in recs[s]]
    import gc; gc.collect()
    print(f"[{run_name}] sizes {[len(ds[s]) for s in ds]}; median train tokens {int(np.median(ds['train'].lens))}; truncated share {np.mean(np.array(ds['train'].lens) >= a.max_length):.2f}; prep {time.time()-t0:.0f}s", flush=True)
    enc = AutoModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float32, attn_implementation="sdpa", add_pooling_layer=False)
    if not a.no_grad_ckpt and a.tuning != "frozen":
        enc.gradient_checkpointing_enable()
    if a.tuning == "lora":
        from peft import LoraConfig, get_peft_model
        enc = get_peft_model(enc, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, target_modules=["query", "key", "value", "dense"]))
        enc.print_trainable_parameters()
    if a.tuning == "frozen":
        for p in enc.parameters(): p.requires_grad_(False)
    model = Clf(enc, enc.config.hidden_size).to(device)
    y_tr = np.array(ds["train"].y); pos_w = torch.tensor((1 - y_tr.mean()) / y_tr.mean() if a.class_weight == "balanced" else 1.0, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    enc_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr}, {"params": model.head.parameters(), "lr": a.head_lr}], weight_decay=a.weight_decay)
    dl = DataLoader(ds["train"], batch_size=a.batch, shuffle=True, collate_fn=lambda b: collate(b, tok.pad_token_id), num_workers=2, drop_last=False)
    steps_per_epoch = math.ceil(len(dl) / a.grad_accum); total = steps_per_epoch * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(a.warmup * total), total)
    best = {"auprc": -1}; history = []; step = 0; stale = 0; stop = False
    ck = RESULTS / run_name; ck.mkdir(parents=True, exist_ok=True)

    def evaluate(tag):
        nonlocal best, stale, stop
        p = predict(model, ds["dev"], tok.pad_token_id, a.eval_batch, device); m = metrics(ds["dev"].y, p, n_boot=0)
        rec = {"tag": tag, "step": step, "dev_auroc": m["auroc"], "dev_auprc": m["auprc"], "elapsed_s": round(time.time() - t0)}
        history.append(rec); print(json.dumps(rec), flush=True)
        if m["auprc"] > best["auprc"]:
            best = {"auprc": m["auprc"], "auroc": m["auroc"], "tag": tag, "step": step}
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items() if model.get_parameter(k).requires_grad} if a.tuning != "full" else model.state_dict(), ck / "best.pt")
            pt = predict(model, ds["test"], tok.pad_token_id, a.eval_batch, device)   # provisional test scores at the current best
            prov = {"dev": metrics(ds["dev"].y, p, n_boot=0), "test": metrics(ds["test"].y, pt, n_boot=0)}
            pd.DataFrame({"person_id": [r["person_id"] for r in recs["test"]], "label": ds["test"].y, "prob": pt}).to_csv(ck / "predictions_test_provisional.csv", index=False)
            (ck / "metrics_provisional.json").write_text(json.dumps({"best": best, "metrics": prov, "history": history}, indent=1, default=str))
            print(f"  provisional test @ {tag}: AUROC {prov['test']['auroc']:.3f} AP {prov['test']['auprc']:.3f}", flush=True)
            stale = 0
        else:
            stale += 1
            if a.patience and stale >= a.patience:
                stop = True; print(f"  early stop: {stale} evaluations without dev AP improvement", flush=True)
        model.train()

    model.train()
    for ep in range(a.epochs):
        run_loss = 0.0; n = 0
        for i, (ids, am, y) in enumerate(dl):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids.to(device), am.to(device))
            loss = loss_fn(logits.float(), y.to(device)) / a.grad_accum; loss.backward()
            run_loss += loss.item() * a.grad_accum; n += 1
            if (i + 1) % a.grad_accum == 0 or i + 1 == len(dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 50 == 0:
                    print(f"ep {ep} step {step}/{total} loss {run_loss/n:.4f} {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
                if a.eval_steps and step % a.eval_steps == 0:
                    evaluate(f"ep{ep}_step{step}")
                if stop:
                    break
        if stop:
            break
        evaluate(f"ep{ep}_end")
    # reload best and score dev/test
    sd = torch.load(ck / "best.pt", map_location=device); model.load_state_dict(sd, strict=False)
    preds = {s: predict(model, ds[s], tok.pad_token_id, a.eval_batch, device) for s in ["dev", "test"]}
    ms = {s: metrics(ds[s].y, preds[s], groups[s]) for s in preds}
    print(f"[{run_name}] BEST {best} | dev AUROC {ms['dev']['auroc']:.3f} AP {ms['dev']['auprc']:.3f} | test AUROC {ms['test']['auroc']:.3f} AP {ms['test']['auprc']:.3f} sens@spec90 {ms['test']['sens_at_spec90']:.3f} | {time.time()-t0:.0f}s", flush=True)
    save_run(run_name, vars(a) | {"model": MODEL_DIR, "n_train": len(ds["train"]), "pos_weight": float(pos_w)}, ms,
             {s: pd.DataFrame({"person_id": [r["person_id"] for r in recs[s]], "label": ds[s].y, "prob": preds[s]}) for s in preds},
             extra={"best": best, "history": history})
    if a.tuning == "full":
        os.remove(ck / "best.pt")   # 550MB per run; keep only LoRA/head checkpoints


if __name__ == "__main__":
    main()
