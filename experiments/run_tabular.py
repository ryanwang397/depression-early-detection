#!/usr/bin/env python3
"""Tabular baselines and L4 (Evidence Ledger) feature models.

  python run_tabular.py --lead 12 [--max-train N --max-eval N] [--tag smoke]                      # original 3 reference models
  python run_tabular.py --lead 12 --feature-sets utilization_only,structured_L3_meta,l4,l4_plus_all --l4   # ledger features (needs outputs/l4/features_lead{L}.parquet)
  python run_tabular.py --leads 12,9,6,3 --joint --l4 --feature-sets l4_plus_all                   # pooled multi-lead training, per-lead evaluation
Feature sets:
  meta                = age, sex, history_days, recency_gap_days               (LR)
  structured_L3       = JSONL `features` (dx/rx/L3 counts, PHQ/BMI)            (LR)
  structured_L3_meta  = structured_L3 + meta                                   (HGB; the 0.70 reference)
  utilization_only    = meta + note/visit/char counts + utilization trajectory (HGB; the "honest" baseline)
  l4                  = meta + utilization + L4 ledger features                (HGB)
  l4_plus_all         = everything                                             (HGB)
Every run also reports subgroup AUROC by sex, age band and note-count quartile (see exp_common.metrics)."""
from __future__ import annotations
import argparse, time, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from exp_common import load_split, metrics, subgroup_fields, save_run, ROOT, person_hash

META = ["age_at_cutoff", "female", "history_days", "recency_gap_days"]
L4_DIR = ROOT / "outputs" / "l4"
UTIL_BASE = ["n_encounters", "n_notes", "n_visits_365d", "n_ed_visits_365d"]


def frame(recs):
    rows = []
    for r in recs:
        f = {k: v for k, v in r["features"].items() if isinstance(v, (int, float))}
        rows.append({"person_id": r["person_id"], "label": r["label"], "age_at_cutoff": r["age_at_cutoff"], "female": int(r.get("sex") == "FEMALE"),
                     "history_days": r["history_days"], "recency_gap_days": r["recency_gap_days"], **f})
    return pd.DataFrame(rows)


def load_lead(lead, a):
    """Returns {split: (df, groups)}; with --l4 the frame comes from the ledger feature parquet (already holds base features)."""
    out = {}
    if a.l4:
        fp = L4_DIR / f"features_lead{lead}.parquet"
        big = pd.read_parquet(fp)
        for s in ["train", "dev", "test"]:
            d = big[big.split == s].drop(columns=["split"]).reset_index(drop=True)
            mx = a.max_train if s == "train" else a.max_eval
            if mx and len(d) > mx:   # same deterministic label-stratified subset as load_split
                d["h"] = [person_hash(p) for p in d.person_id]; prev = d.label.mean(); n_pos = int(round(mx * prev))
                d = pd.concat([d[d.label == 1].nsmallest(n_pos, "h"), d[d.label == 0].nsmallest(mx - n_pos, "h")]).sort_values("h").drop(columns="h").reset_index(drop=True)
            age = d.age_at_cutoff.values; nn = d.get("n_notes", pd.Series(np.zeros(len(d)))).values.astype(float)
            groups = {"sex": np.where(d.female == 1, "FEMALE", "MALE"), "age_band": pd.cut(age, [0, 15, 18, 22, 100], labels=["12-15", "15-18", "18-22", "22+"]).astype(str),
                      "note_quartile": pd.qcut(nn, 4, labels=["q1", "q2", "q3", "q4"], duplicates="drop").astype(str)}
            out[s] = (d, groups)
    else:
        for s in ["train", "dev", "test"]:
            recs = load_split(lead, s, a.max_train if s == "train" else a.max_eval, keep_text=False)
            out[s] = (frame(recs), subgroup_fields(recs))
    return out


def feature_cols(all_cols, fs):
    util_traj = [c for c in all_cols if c in ("visits_90", "visits_180", "visits_365", "visits_prev365", "slope_visits", "ed_365", "inpatient_365", "n_specialties_365", "days_since_last_visit",
                                               "notes_365", "chars_365", "chars_all", "rel_notes_365", "n_condition_days_365", "n_distinct_icd3_365", "n_meds_365", "bmi_last", "bmi_delta_365", "bmi_n")]
    l4 = [c for c in all_cols if c.startswith("l4_")]
    base_struct = [c for c in all_cols if c not in META and not c.startswith("l4_") and c not in util_traj and c not in ("person_id", "label", "lead_months", "split")]
    lead = ["lead_months"] if "lead_months" in all_cols else []
    sets = {"meta": META + lead, "structured_L3": base_struct + lead, "structured_L3_meta": META + base_struct + lead,
            "utilization_only": META + [c for c in UTIL_BASE if c in all_cols] + util_traj + lead,
            "ehr_only": META + [c for c in base_struct if not c.startswith("notes_") and not c.startswith("phq") and not c.startswith("gad") and c != "n_notes"]
                        + [c for c in util_traj if c not in ("notes_365", "chars_365", "chars_all", "rel_notes_365", "n_specialties_365")] + lead,
            "ehr_only_plus_note_counts": META + [c for c in base_struct if not c.startswith("notes_") and not c.startswith("phq") and not c.startswith("gad")] + util_traj + lead,
            "structured_L3_meta_util": META + base_struct + util_traj + lead,
            "l4_present_only": META + [c for c in UTIL_BASE if c in all_cols] + util_traj + [c for c in l4 if "__present" in c or not "__" in c] + lead,
            "l4": META + [c for c in UTIL_BASE if c in all_cols] + util_traj + l4 + lead, "l4_plus_all": META + base_struct + util_traj + l4 + lead}
    return sets[fs]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lead", type=int, default=12); ap.add_argument("--leads", default=None); ap.add_argument("--joint", action="store_true")
    ap.add_argument("--max-train", type=int, default=None); ap.add_argument("--max-eval", type=int, default=None); ap.add_argument("--tag", default="full")
    ap.add_argument("--l4", action="store_true"); ap.add_argument("--feature-sets", default=None); ap.add_argument("--drop-llm-prevalent", action="store_true")
    ap.add_argument("--no-save", action="store_true", help="print metrics only; do not write results/ or registry"); ap.add_argument("--hgb-iter", type=int, default=400); ap.add_argument("--hgb-lr", type=float, default=0.05); ap.add_argument("--hgb-depth", type=int, default=5)
    a = ap.parse_args(); t0 = time.time()
    leads = [int(x) for x in a.leads.split(",")] if a.leads else [a.lead]
    data = {L: load_lead(L, a) for L in leads}
    fsets = a.feature_sets.split(",") if a.feature_sets else (["meta", "structured_L3", "structured_L3_meta"] if not a.l4 else ["utilization_only", "structured_L3_meta", "l4", "l4_plus_all"])
    all_cols = sorted({c for L in data for s in data[L] for c in data[L][s][0].columns})
    for L in data:
        for s in data[L]:
            d = data[L][s][0]
            for c in all_cols:
                if c not in d:
                    d[c] = np.nan
            if "lead_months" not in d or d["lead_months"].isna().all():
                d["lead_months"] = L
    print(f"loaded leads {leads} sizes {[{s: len(data[L][s][0]) for s in data[L]} for L in leads]} in {time.time()-t0:.0f}s; {len(all_cols)} columns")
    hgb = lambda: HistGradientBoostingClassifier(max_iter=a.hgb_iter, learning_rate=a.hgb_lr, max_depth=a.hgb_depth, min_samples_leaf=40, l2_regularization=1.0, class_weight="balanced", random_state=0)
    lr = lambda C=1.0: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", C=C))
    for fs in fsets:
        cols = feature_cols(all_cols, fs)
        model_name = "lr" if fs in ("meta", "structured_L3") else "hgb"
        train_frames = [data[L]["train"][0] for L in (leads if a.joint else [leads[0]])]
        tr = pd.concat(train_frames, ignore_index=True)
        if a.drop_llm_prevalent and "l4_prevalent_flag" in tr:
            n0 = len(tr); tr = tr[tr.l4_prevalent_flag == 0]; print(f"dropped {n0-len(tr)} LLM-prevalent train rows")
        Xtr = tr[cols].astype(float); ytr = tr.label.values
        if model_name == "lr":
            m = lr(0.3 if fs == "structured_L3" else 1.0).fit(Xtr.fillna(0.0).values, ytr); pred = lambda d: m.predict_proba(d[cols].astype(float).fillna(0.0).values)[:, 1]
        else:
            m = hgb().fit(Xtr.values, ytr); pred = lambda d: m.predict_proba(d[cols].astype(float).values)[:, 1]
        for L in (leads if a.joint else [leads[0]]):
            preds = {s: pred(data[L][s][0]) for s in ["dev", "test"]}
            ms = {s: metrics(data[L][s][0].label, preds[s], data[L][s][1]) for s in preds}
            name = f"{fs}_{model_name}" + ("_joint" if a.joint else "")
            print(f"lead {L:2d} {name:36s} ({len(cols):3d} feats) dev AUROC {ms['dev']['auroc']:.4f} AP {ms['dev']['auprc']:.4f} | test AUROC {ms['test']['auroc']:.4f} AP {ms['test']['auprc']:.4f} "
                  f"sens@sp90 {ms['test']['sens_at_spec90']:.3f} ppv@5 {ms['test']['ppv_top5']:.3f} ece {ms['test']['ece']:.3f} | note-quartile AUROC "
                  + " ".join(f"{k.split('=')[1]}:{v['auroc']:.3f}" for k, v in ms['test'].get('subgroups', {}).items() if k.startswith('note_quartile')))
            if a.no_save:
                continue
            save_run(f"lead{L}_{a.tag}_{name}", {"lead": L, "tag": a.tag, "n_train": len(tr), "feature_set": fs, "n_features": len(cols), "model": model_name, "joint": a.joint,
                                                  "leads_trained": leads if a.joint else [L], "drop_llm_prevalent": a.drop_llm_prevalent}, ms,
                     {s: pd.DataFrame({"person_id": data[L][s][0].person_id, "label": data[L][s][0].label, "prob": preds[s]}) for s in preds})
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
