"""Shared utilities for baseline experiments on the depression early-detection dataset.

- load_split(): stream a JSONL split, optionally a deterministic stratified subset (by person hash)
- build_text(): render model input text from a record (notes-only / notes+EHR, recent-first packing)
- metrics(): AUROC / AUPRC / sensitivity@specificity / Brier, with bootstrap CIs and subgroup AUROCs
- save_run(): write metrics.json + predictions.csv into results/<run_name>/
"""
from __future__ import annotations
import json, re, hashlib, time, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DED_DATA_DIR", ROOT / "outputs" / "dataset"))   # override with DED_DATA_DIR=/path/to/dataset_variant
RESULTS = ROOT / "results"
NOTE_RE = re.compile(r'<note type="([^"]*)" specialty="([^"]*)" author="([^"]*)" relevance="(\d+)">\n?(.*?)\n?\s*</note>', re.S)
ENC_RE = re.compile(r'<encounter date="([^"]*)" days_before_cutoff="(\d+)" visit="([^"]*)">(.*?)</encounter>', re.S)
EHR_RE = re.compile(r"<ehr>\n?(.*?)\n?\s*</ehr>", re.S)


def person_hash(pid: int, salt: str = "subset") -> float:
    return int.from_bytes(hashlib.blake2b(f"{salt}|{pid}".encode(), digest_size=8).digest(), "big") / 2**64


def load_split(lead: int, split: str, max_n: int | None = None, keep_text: bool = True, seed_salt: str = "subset") -> list[dict]:
    """Deterministic stratified subset: keeps the first max_n records by (label-stratified) hash order,
    preserving the split's class prevalence."""
    path = DATA / f"lead_{lead}mo" / f"{split}.jsonl"
    recs = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if not keep_text:
                r.pop("encounters", None)
            recs.append(r)
    if max_n and len(recs) > max_n:
        df = pd.DataFrame({"i": range(len(recs)), "y": [r["label"] for r in recs], "h": [person_hash(r["person_id"], seed_salt) for r in recs]})
        prev = df.y.mean(); n_pos = int(round(max_n * prev)); n_neg = max_n - n_pos
        keep = pd.concat([df[df.y == 1].nsmallest(n_pos, "h"), df[df.y == 0].nsmallest(n_neg, "h")]).sort_values("h").i.tolist()
        recs = [recs[i] for i in keep]
    return recs


def parse_encounters(text: str) -> list[dict]:
    """encounter_text -> [{date, days_before_cutoff, visit, ehr, notes:[{type,specialty,author,relevance,text}]}] (chronological)."""
    out = []
    for m in ENC_RE.finditer(text):
        body = m.group(4)
        ehr = EHR_RE.search(body)
        notes = [{"type": n[0], "specialty": n[1], "author": n[2], "relevance": int(n[3]), "text": n[4].strip()} for n in NOTE_RE.findall(body)]
        out.append({"date": m.group(1), "days_before_cutoff": int(m.group(2)), "visit": m.group(3), "ehr": ehr.group(1).strip() if ehr else "", "notes": notes})
    return out


def header_text(r: dict) -> str:
    return f"Patient: {r['sex'].lower() if r.get('sex') else 'unknown sex'}, age {r['age_at_cutoff']:.0f} at prediction time; {r['history_days']} days of prior record."


def build_text(r: dict, mode: str = "notes", max_chars: int = 40000, header: bool = True) -> str:
    """Recent-first packing: walk encounters from the cutoff backwards, add units until max_chars,
    then re-order chronologically. mode: notes | notes_ehr | ehr."""
    encs = parse_encounters(r["encounter_text"])
    units = []  # (order_key, text)
    for e in reversed(encs):
        parts = []
        if mode in ("notes_ehr", "ehr") and e["ehr"]:
            parts.append(f"[EHR] {e['ehr']}")
        if mode in ("notes", "notes_ehr"):
            for n in e["notes"]:
                parts.append(f"[{n['type']}{(' / ' + n['specialty']) if n['specialty'] else ''}] {n['text']}")
        if parts:
            units.append((e["days_before_cutoff"], f"=== {e['days_before_cutoff']} days before prediction ({e['visit']}) ===\n" + "\n".join(parts)))
    kept = []; used = 0
    for k, t in units:
        if used + len(t) > max_chars:
            if not kept:  # always keep at least the most recent unit (truncated)
                kept.append((k, t[:max_chars]))
            break
        kept.append((k, t)); used += len(t)
    kept.sort(key=lambda x: -x[0])  # oldest first
    body = "\n\n".join(t for _, t in kept)
    return (header_text(r) + "\n\n" + body) if header else body


def ppv_at_top(y, s, frac=0.05):
    """Precision among the top `frac` highest-scored samples."""
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float); k = max(int(round(frac * len(y))), 1)
    return float(y[np.argsort(-s)[:k]].mean())


def ece(y, s, n_bins=10):
    """Expected calibration error with equal-width probability bins (scores clipped to [0,1])."""
    y = np.asarray(y).astype(int); s = np.clip(np.asarray(s, dtype=float), 0, 1); b = np.minimum((s * n_bins).astype(int), n_bins - 1); e = 0.0
    for i in range(n_bins):
        m = b == i
        if m.any():
            e += m.mean() * abs(y[m].mean() - s[m].mean())
    return float(e)


def sens_at_spec(y, s, spec=0.90):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= (1 - spec)
    return float(tpr[ok].max()) if ok.any() else 0.0


def metrics(y, s, groups: dict | None = None, n_boot: int = 200, seed: int = 0) -> dict:
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
           "sens_at_spec90": sens_at_spec(y, s, 0.90), "sens_at_spec80": sens_at_spec(y, s, 0.80), "brier": float(brier_score_loss(y, np.clip(s, 0, 1))),
           "ppv_top5": ppv_at_top(y, s, 0.05), "ppv_top10": ppv_at_top(y, s, 0.10), "ece": ece(y, s)}
    rng = np.random.default_rng(seed); aucs = []; aps = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], s[idx])); aps.append(average_precision_score(y[idx], s[idx]))
    if aucs:
        out["auroc_ci95"] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
        out["auprc_ci95"] = [float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))]
    if groups:
        out["subgroups"] = {}
        for name, g in groups.items():
            g = np.asarray(g)
            for v in pd.unique(g):
                m = g == v
                if m.sum() >= 50 and y[m].min() != y[m].max():
                    out["subgroups"][f"{name}={v}"] = {"n": int(m.sum()), "auroc": float(roc_auc_score(y[m], s[m])), "auprc": float(average_precision_score(y[m], s[m]))}
    return out


def subgroup_fields(recs: list[dict]) -> dict:
    age = np.array([r["age_at_cutoff"] for r in recs])
    n_notes = np.array([r.get("features", {}).get("n_notes", 0) for r in recs], dtype=float)
    q = pd.qcut(n_notes, 4, labels=["q1", "q2", "q3", "q4"], duplicates="drop").astype(str) if len(recs) >= 8 else np.array(["q1"] * len(recs))
    return {"sex": [r.get("sex") for r in recs], "age_band": pd.cut(age, [0, 15, 18, 22, 100], labels=["12-15", "15-18", "18-22", "22+"]).astype(str), "note_quartile": q}


def f1_at_threshold(y, s, thr):
    y = np.asarray(y).astype(int); pred = (np.asarray(s) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"threshold": float(thr), "f1": f1, "precision": prec, "recall": rec, "n_flagged": tp + fp}


def best_f1_threshold(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s)
    best = (0.0, 0.5)
    for thr in np.unique(np.quantile(s, np.linspace(0.02, 0.98, 97))):
        f = f1_at_threshold(y, s, thr)["f1"]
        if f > best[0]:
            best = (f, float(thr))
    return best[1]


def save_run(run_name: str, config: dict, split_metrics: dict, predictions: dict[str, pd.DataFrame], extra: dict | None = None) -> Path:
    d = RESULTS / run_name; d.mkdir(parents=True, exist_ok=True)
    if "dev" in predictions and "test" in predictions:   # F1 at the threshold that maximises F1 on dev
        thr = best_f1_threshold(predictions["dev"].label, predictions["dev"].prob)
        for s in predictions:
            split_metrics.setdefault(s, {}).update({f"{k}_devthr": v for k, v in f1_at_threshold(predictions[s].label, predictions[s].prob, thr).items()})
    payload = {"run_name": run_name, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": config, "metrics": split_metrics, **(extra or {})}
    (d / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    for split, df in predictions.items():
        df.to_csv(d / f"predictions_{split}.csv", index=False)
    row = {"run": run_name, **{f"{s}_{k}": round(v, 4) for s, m in split_metrics.items() for k, v in m.items() if isinstance(v, (int, float))}}
    with open(RESULTS / "registry.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    return d
