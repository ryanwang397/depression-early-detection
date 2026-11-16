#!/usr/bin/env python3
"""f1: care-context "coding propensity" features (design 1: disease risk x detection/coding).

  python f1_context_features.py --lead 3
For every patient, the care contexts of the last 365 days before the cutoff (note specialty, note author, note type,
visit type) are target-encoded on TRAIN patients only (5-fold out-of-fold for train rows, full-train statistics for dev/test):
    propensity(context = v) = (sum of labels of train patients exposed to v + m * prior) / (n exposed + m)
i.e. how often patients seen in that setting / by that clinician get a depression code within the lead window — a proxy for
screening and coding practice.  Per-patient features = max / mean / most-recent propensity per context type.
Outputs: outputs/eagrl/lead{L}_context.parquet (features + a human-readable CARE CONTEXT line for prompts) and a quick
HGB experiment (l4_plus_all with vs. without the context features) printed to stdout."""
from __future__ import annotations
import argparse, json, time, math
from collections import defaultdict
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from common import ROOT, OUT, person_hash
from run_tabular import feature_cols

KEYS = ["specialty", "author", "note_type", "visit_type"]
SMOOTH = {"specialty": 30.0, "author": 50.0, "note_type": 30.0, "visit_type": 30.0}


def stream_contexts(lead: int, splits: list[str], window: int = 365) -> pd.DataFrame:
    rows = []; t0 = time.time()
    for split in splits:
        n = 0
        with open(ROOT / "outputs" / "dataset" / f"lead_{lead}mo" / f"{split}.jsonl") as f:
            for line in f:
                r = json.loads(line); n += 1
                for e in r["encounters"]:
                    d = e["days_before_cutoff"]
                    if d > window: continue
                    vt = e["visit_type"] or "unknown"
                    if not e["notes"]:
                        rows.append((r["person_id"], split, r["label"], d, None, None, None, vt))
                    for nt in e["notes"]:
                        rows.append((r["person_id"], split, r["label"], d, nt["note_id"], nt.get("specialty"), nt.get("type"), vt))
        print(f"  {split}: {n} patients, {len(rows)} context rows so far ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows, columns=["person_id", "split", "label", "days", "note_id", "specialty", "note_type", "visit_type"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lead", type=int, default=3); ap.add_argument("--folds", type=int, default=5); a = ap.parse_args(); t0 = time.time()
    cx = stream_contexts(a.lead, ["train", "dev", "test"])
    # author (de-identified clinician id) via note_meta_aligned
    nids = cx.note_id.dropna().astype("int64").unique()
    nm = ds.dataset(str(ROOT / "outputs" / "index" / "note_meta_aligned.parquet")).to_table(columns=["note_id", "author_key"], filter=pc.field("note_id").isin(nids.tolist())).to_pandas().drop_duplicates("note_id")
    cx = cx.merge(nm.rename(columns={"author_key": "author"}), on="note_id", how="left"); cx["author"] = cx.author.astype("Int64").astype(str).where(cx.author.notna(), None)
    print(f"contexts: {len(cx)} rows, authors matched for {cx.author.notna().mean():.1%} of note rows; {time.time()-t0:.0f}s", flush=True)
    pat = cx[["person_id", "split", "label"]].drop_duplicates("person_id").reset_index(drop=True)
    pat["fold"] = [int(person_hash(p, "ctx_fold") * a.folds) for p in pat.person_id]
    prior = pat[pat.split == "train"].label.mean()
    feats = {p: {} for p in pat.person_id}; levels = {}
    for key in KEYS:
        sub = cx[cx[key].notna()][["person_id", "split", "label", "days", key]].rename(columns={key: "v"})
        expo = sub.drop_duplicates(["person_id", "v"]).merge(pat[["person_id", "fold"]], on="person_id")
        tr = expo[expo.split == "train"]
        def enc_table(tr_rows):
            g = tr_rows.groupby("v").label.agg(["sum", "count"]); return ((g["sum"] + SMOOTH[key] * prior) / (g["count"] + SMOOTH[key])).to_dict(), g["count"].to_dict()
        full, full_n = enc_table(tr); oof = {}
        for k in range(a.folds): oof[k] = enc_table(tr[tr.fold != k])[0]
        def prop(row):
            table = oof[row.fold] if row.split == "train" else full
            return table.get(row.v, prior)
        expo = expo.assign(prop=[prop(r) for r in expo.itertuples()])
        most_recent = sub.sort_values("days").drop_duplicates("person_id")[["person_id", "v"]].merge(expo[["person_id", "v", "prop"]], on=["person_id", "v"], how="left")
        agg = expo.groupby("person_id").prop.agg(["max", "mean", "min", "count"])
        for pid, r in agg.iterrows(): feats[pid].update({f"ctx_{key}_max": r["max"], f"ctx_{key}_mean": r["mean"], f"ctx_{key}_min": r["min"], f"ctx_{key}_n": r["count"]})
        for r in most_recent.itertuples(): feats[r.person_id][f"ctx_{key}_recent"] = r.prop
        # tertile levels for the prompt line, computed over TRAIN PATIENTS' exposures (so low/average/high are balanced across patients)
        vals = expo[expo.split == "train"].prop.values; lo, hi = np.quantile(vals, [0.33, 0.67]); levels[key] = (full, lo, hi, full_n)
        print(f"  {key}: {len(full)} contexts encoded (train n>=1), tertiles {lo:.3f}/{hi:.3f}; example top: {sorted(full.items(), key=lambda kv: -kv[1])[:3]}", flush=True)
    F = pd.DataFrame.from_dict(feats, orient="index").reset_index().rename(columns={"index": "person_id"}); F = pat.merge(F, on="person_id", how="left")
    # human-readable CARE CONTEXT line
    def level(key, v):
        full, lo, hi, n = levels[key]; p = full.get(v)
        if p is None or n.get(v, 0) < 20: return "unknown"
        return "high" if p >= hi else ("low" if p <= lo else "average")
    lines = {}
    for pid, g in cx.groupby("person_id"):
        g = g.sort_values("days"); specs = [s for s in g.specialty.dropna().unique()][:4]; auth = g.author.dropna(); vts = [v for v in g.visit_type.unique() if v not in ("none", "unknown")][:3]
        s_spec = ", ".join(f"{s} [{level('specialty', s)}]" for s in specs) if specs else "none recorded"
        s_auth = level("author", auth.iloc[0]) if len(auth) else "unknown"
        lines[pid] = (f"specialties seen in the last 12 months with their depression-coding propensity: {s_spec}; most recent clinician's coding propensity: {s_auth}; "
                      f"visit types: {', '.join(vts) if vts else 'none recorded'}.")
    F["context_text"] = F.person_id.map(lines).fillna("no visits in the last 12 months.")
    F.to_parquet(OUT / f"lead{a.lead}_context.parquet", index=False); print(f"wrote {OUT / f'lead{a.lead}_context.parquet'} ({len(F)} patients, {F.shape[1]} cols); {time.time()-t0:.0f}s", flush=True)
    # ---- quick experiment: context features alone, and HGB l4_plus_all with / without ----
    big = pd.read_parquet(ROOT / "outputs" / "l4" / f"features_lead{a.lead}.parquet"); big["lead_months"] = a.lead
    cols = feature_cols(sorted(c for c in big.columns if c != "split"), "l4_plus_all"); ctx_cols = [c for c in F.columns if c.startswith("ctx_")]
    big = big.merge(F[["person_id"] + ctx_cols], on="person_id", how="left")
    tr = big[big.split == "train"]; hgb = lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=5, min_samples_leaf=40, l2_regularization=1.0, class_weight="balanced", random_state=0)
    hist_bins = pd.cut(big.history_days / 365, [0, 0.5, 1, 2, 3, 6, 100], labels=False)
    def strat_auroc(m, p):
        w = []
        for b in np.unique(hist_bins[m]):
            mm = m & (hist_bins == b).values
            if big.label[mm].nunique() == 2: w.append((mm.sum(), roc_auc_score(big.label[mm], p[mm])))
        return sum(n * s for n, s in w) / sum(n for n, _ in w)
    for name, cset in (("l4_plus_all", cols), ("l4_plus_all + context", cols + ctx_cols), ("context only (LR)", ctx_cols)):
        X = big[cset].astype(float)
        if name.startswith("context only"):
            from sklearn.pipeline import make_pipeline; from sklearn.impute import SimpleImputer; from sklearn.preprocessing import StandardScaler
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(X[big.split == "train"].values, tr.label.values); p = m.predict_proba(X.values)[:, 1]
        else:
            m = hgb().fit(X[big.split == "train"].values, tr.label.values); p = m.predict_proba(X.values)[:, 1]
        for s in ("dev", "test"):
            mm = (big.split == s).values
            print(f"  {name:26s} {s}: AUROC {roc_auc_score(big.label[mm], p[mm]):.4f} AP {average_precision_score(big.label[mm], p[mm]):.4f} | history-stratified AUROC {strat_auroc(mm, p):.4f}", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
