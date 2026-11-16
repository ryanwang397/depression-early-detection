#!/usr/bin/env python3
"""e0: the expert model and its per-patient "attention".

  python e0_expert.py --lead 3 [--top-k 5]
Trains the l4_plus_all HistGradientBoosting expert exactly as experiments/run_tabular.py (same features/hyper-parameters),
then computes, for every patient of every split, a concept-level occlusion attribution:
    delta[c] = logit p(x) - logit p(x with all columns of concept c set to their "absent" reference value)
The top-k concepts by |delta| are the expert's "important features" (with sign) used for the Jaccard alignment reward and
for the hindsight teacher prompts. Writes outputs/eagrl/lead{L}_expert.parquet and lead{L}_expert_summary.json."""
from __future__ import annotations
import os
import argparse, json, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from common import ROOT, OUT, CONCEPT_NAMES, ALIGN_EXCLUDE, ALIGN_EXCLUDE_V3, concept_of_column, absent_value
from run_tabular import feature_cols   # experiments/ is on sys.path via common


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lead", type=int, default=3); ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--hgb-iter", type=int, default=400); ap.add_argument("--hgb-lr", type=float, default=0.05); ap.add_argument("--hgb-depth", type=int, default=5); ap.add_argument("--train-incident-only", action="store_true", help="fit the HGB only on train rows that are not prevalent positives (cohort case_type != prevalent)"); ap.add_argument("--gate-prior-evidence", action="store_true", help="patients with documented prior depression evidence (case_type prevalent) get no expert opinion (gated=True rows)"); ap.add_argument("--suffix", default=None, help="output suffix (default: _v3 for --v3-exclude, _inc when --train-incident-only/--gate-prior-evidence, else empty)"); ap.add_argument("--v3-exclude", action="store_true", help="exclude all demographic/utilisation concepts from the expert top-k (writes *_expert_v3.parquet)")
    a = ap.parse_args(); t0 = time.time(); EXCL = ALIGN_EXCLUDE_V3 if a.v3_exclude else ALIGN_EXCLUDE; SUF = a.suffix if a.suffix is not None else ("_v3" if a.v3_exclude else ("_inc" if (a.train_incident_only or a.gate_prior_evidence) else ""))
    df = pd.read_parquet(ROOT / "outputs" / "l4" / f"features_lead{a.lead}.parquet")
    if "lead_months" not in df or df["lead_months"].isna().all(): df["lead_months"] = a.lead
    cols = feature_cols(sorted(c for c in df.columns if c != "split"), "l4_plus_all")
    groups = {}
    for c in cols:
        k = concept_of_column(c)
        if k: groups.setdefault(k, []).append(c)
    print(f"lead {a.lead}: {len(df)} rows, {len(cols)} features, {len(groups)} concept groups; unmapped: {[c for c in cols if concept_of_column(c) is None]}", flush=True)
    coh = pd.read_parquet(os.environ.get("DED_COHORT_FILE", ROOT / "outputs" / "cohort" / "lead3_aug_cohort.parquet")); coh = coh[coh.lead_months == a.lead][["person_id", "case_type"]].drop_duplicates("person_id") if "case_type" in coh else None
    prevalent = set(coh.person_id[coh.case_type == "prevalent"]) if coh is not None else set()
    df["gated"] = df.person_id.isin(prevalent) & bool(a.gate_prior_evidence)
    tr = df[df.split == "train"]
    if a.train_incident_only: tr = tr[~tr.person_id.isin(prevalent)]; print(f"train rows after dropping prevalent positives: {len(tr)} (pos {int(tr.label.sum())})", flush=True)
    hgb = HistGradientBoostingClassifier(max_iter=a.hgb_iter, learning_rate=a.hgb_lr, max_depth=a.hgb_depth, min_samples_leaf=40, l2_regularization=1.0, class_weight="balanced", random_state=0)
    hgb.fit(tr[cols].astype(float).values, tr.label.values); print(f"HGB trained in {time.time()-t0:.0f}s", flush=True)
    X = df[cols].astype(float); p = hgb.predict_proba(X.values)[:, 1]
    for s in ["dev", "test"]:
        m = (df.split == s).values; print(f"  {s}: AUROC {roc_auc_score(df.label[m], p[m]):.4f} AP {average_precision_score(df.label[m], p[m]):.4f}", flush=True)
    logit = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
    base = logit(p); D = np.zeros((len(df), len(CONCEPT_NAMES)), dtype=np.float32)
    for j, c in enumerate(CONCEPT_NAMES):
        gcols = groups.get(c)
        if not gcols: continue
        Xo = X.copy()
        for col in gcols: Xo[col] = absent_value(col)
        D[:, j] = base - logit(hgb.predict_proba(Xo.values)[:, 1])
    print(f"occlusion done in {time.time()-t0:.0f}s", flush=True)
    order = np.argsort(-np.abs(D), axis=1)
    rows = []
    for i in range(len(df)):
        top = [CONCEPT_NAMES[j] for j in order[i] if abs(D[i, j]) > 1e-6 and CONCEPT_NAMES[j] not in EXCL][:a.top_k]
        signed = [[CONCEPT_NAMES[j], round(float(D[i, j]), 4)] for j in order[i, :10] if abs(D[i, j]) > 1e-6]
        g = bool(df.gated.iloc[i])
        rows.append({"person_id": int(df.person_id.iloc[i]), "split": df.split.iloc[i], "label": int(df.label.iloc[i]), "prob": float("nan") if g else float(p[i]), "top": json.dumps([] if g else top), "signed": json.dumps([] if g else signed), "gated": g})
    out = pd.DataFrame(rows); out.to_parquet(OUT / f"lead{a.lead}_expert{SUF}.parquet", index=False); print(f"gated (no expert opinion) rows: {int(out.gated.sum())}", flush=True)
    y = df.label.values
    summ = {"lead": a.lead, "n": len(df), "hgb": {"auroc_test": float(roc_auc_score(y[df.split == "test"], p[df.split == "test"]))},
            "concepts": {c: {"n_cols": len(groups.get(c, [])), "mean_abs_delta": float(np.abs(D[:, j]).mean()), "mean_delta_pos": float(D[y == 1, j].mean()), "mean_delta_neg": float(D[y == 0, j].mean()),
                             "in_top5_rate": float(np.mean([c in json.loads(r["top"]) for r in rows]))} for j, c in enumerate(CONCEPT_NAMES)}}
    (OUT / f"lead{a.lead}_expert{SUF}_summary.json").write_text(json.dumps(summ, indent=2))
    top_by = sorted(summ["concepts"].items(), key=lambda kv: -kv[1]["in_top5_rate"])[:15]
    print("expert top-5 membership rate:", ", ".join(f"{c} {v['in_top5_rate']:.2f}" for c, v in top_by))
    print(f"done in {time.time()-t0:.0f}s -> {OUT / f'lead{a.lead}_expert{SUF}.parquet'}")


if __name__ == "__main__":
    main()
