#!/usr/bin/env python3
"""Step 4: per-(person, lead) longitudinal features from the ledger + utilization/BMI trajectories from the dataset JSONL.

  python s4_ledger_features.py --lead 12 [--splits train,dev,test]
Streams outputs/dataset/lead_{L}mo/{split}.jsonl (never holds encounters for more than one record), filters the
ledger to rows with note_date <= cutoff_date (asserted), and writes outputs/l4/features_lead{L}.parquet with:
  person_id, split, label, base features (JSONL `features` + META), utilization features, BMI trajectory, L4 features.
L4 feature groups (per category and per category/status for the informative statuses):
  cnt_all, cnt_365, cnt_90, days_since_last, days_since_first, trend180 (= cnt_180 - cnt_prev180), rate_all (= cnt_all / n_notes)
plus: n_distinct_somatic_present, n_distinct_stressor_present, n_dep_symptoms_present_365, any_sleep_present_365,
      n_relevant_notes, relevant_note_rate, last PHQ/GAD values, prevalent flag (depression_dx_statement present/history or antidepressant)."""
from __future__ import annotations
import argparse, json, time
from collections import defaultdict
import numpy as np, pandas as pd
from l4_common import OUT, DATASET, get_logger, write_summary
from schema import CATEGORIES

log = get_logger("s4")
META = ["age_at_cutoff", "female", "history_days", "recency_gap_days"]
STATUS_FEAT = ["present", "history", "uncertain", "negated", "family"]
CATS = list(CATEGORIES) + ["other_psych_dx"]   # ledger-only category produced by validate.gate_item
UTIL_NAMES = ["visits_90", "visits_180", "visits_365", "visits_prev365", "slope_visits", "ed_365", "inpatient_365", "n_specialties_365", "days_since_last_visit",
              "notes_365", "chars_365", "chars_all", "rel_notes_365", "n_condition_days_365", "n_distinct_icd3_365", "n_meds_365", "bmi_last", "bmi_delta_365", "bmi_n"]


def util_features(r: dict) -> dict:
    f = defaultdict(float); specs = set(); icd3 = set(); bmi = []; last = None
    for e in r["encounters"]:
        d = e["days_before_cutoff"]; vt = e["visit_type"]
        real = vt not in ("none", "telephone", "unknown")
        if real:
            last = d if last is None else min(last, d)
            for w in (90, 180, 365):
                if d <= w:
                    f[f"visits_{w}"] += 1
            if 365 < d <= 730:
                f["visits_prev365"] += 1
            if d <= 365:
                if "ed" in vt:
                    f["ed_365"] += 1
                if "inpatient" in vt:
                    f["inpatient_365"] += 1
        for n in e["notes"]:
            f["chars_all"] += len(n["text"])
            if d <= 365:
                f["notes_365"] += 1; f["chars_365"] += len(n["text"]); f["rel_notes_365"] += n["relevance"] > 0
                if n.get("specialty"):
                    specs.add(n["specialty"])
        if d <= 365:
            if e["conditions"]:
                f["n_condition_days_365"] += 1
            for c in e["conditions"]:
                icd3.add((c.get("icd") or "")[:3])
            f["n_meds_365"] += len(e["medications"])
        for m in e["measurements"]:
            if m["name"] == "BMI" and m.get("value") is not None:
                bmi.append((d, float(m["value"])))
    f["slope_visits"] = f["visits_365"] - f["visits_prev365"]; f["n_specialties_365"] = len(specs); f["n_distinct_icd3_365"] = len(icd3)
    f["days_since_last_visit"] = float(last if last is not None else r["history_days"])
    if bmi:
        bmi.sort(); f["bmi_last"] = bmi[0][1]; f["bmi_n"] = len(bmi)
        older = [v for d, v in bmi if 365 <= d <= 730]
        f["bmi_delta_365"] = bmi[0][1] - older[0] if older else 0.0
    else:
        f["bmi_last"] = np.nan; f["bmi_delta_365"] = 0.0; f["bmi_n"] = 0
    return {k: float(f[k]) for k in UTIL_NAMES}


def ledger_features(g: pd.DataFrame | None, cutoff: np.datetime64, n_notes: int, n_ledger_notes: int, n_relevant: int) -> dict:
    f = {}
    for cat in CATS:
        for st in [None] + STATUS_FEAT:
            key = cat if st is None else f"{cat}__{st}"
            f[f"l4_{key}_cnt_all"] = 0.0; f[f"l4_{key}_cnt_365"] = 0.0; f[f"l4_{key}_cnt_90"] = 0.0; f[f"l4_{key}_trend180"] = 0.0
            f[f"l4_{key}_days_last"] = 9999.0; f[f"l4_{key}_days_first"] = 9999.0
    f.update({"l4_n_ledger_notes": float(n_ledger_notes), "l4_n_relevant_notes": float(n_relevant), "l4_relevant_rate": n_relevant / max(n_ledger_notes, 1),
              "l4_n_distinct_somatic_present": 0.0, "l4_n_distinct_stressor_present": 0.0, "l4_n_dep_symptoms_present_365": 0.0, "l4_any_sleep_present_365": 0.0,
              "l4_phq9_last": np.nan, "l4_gad7_last": np.nan, "l4_phq9_max": np.nan, "l4_prevalent_flag": 0.0, "l4_si_present_any": 0.0, "l4_items_per_note": 0.0})
    if g is None or len(g) == 0:
        return f
    days = (cutoff - g.note_date.values.astype("datetime64[D]")).astype("timedelta64[D]").astype(int)
    assert days.min() >= 0, "ledger row after cutoff"
    g = g.assign(days=days)
    f["l4_items_per_note"] = len(g) / max(n_ledger_notes, 1)
    for (cat, st), h in g.groupby(["category", "status"]):
        for key in (cat, f"{cat}__{st}") if st in STATUS_FEAT else (cat,):
            d = h.days.values
            f[f"l4_{key}_cnt_all"] += len(d); f[f"l4_{key}_cnt_365"] += (d <= 365).sum(); f[f"l4_{key}_cnt_90"] += (d <= 90).sum()
            f[f"l4_{key}_trend180"] += (d <= 180).sum() - ((d > 180) & (d <= 360)).sum()
            f[f"l4_{key}_days_last"] = min(f[f"l4_{key}_days_last"], float(d.min())); f[f"l4_{key}_days_first"] = min(9999.0, max(f[f"l4_{key}_days_first"] if f[f"l4_{key}_days_first"] < 9999 else 0, float(d.max())))
    for cat in CATS:
        for key in [cat] + [f"{cat}__{s}" for s in STATUS_FEAT]:
            f[f"l4_{key}_rate_all"] = f[f"l4_{key}_cnt_all"] / max(n_notes, 1)
    pres = g[(g.status == "present") & (g.experiencer == "patient")]
    f["l4_n_distinct_somatic_present"] = float(pres[pres.category == "somatic"].subtype.nunique())
    f["l4_n_distinct_stressor_present"] = float(pres[pres.category == "stressor"].subtype.nunique())
    p365 = pres[pres.days <= 365]
    f["l4_n_dep_symptoms_present_365"] = float(p365[p365.category == "dep_symptom"].subtype.nunique())
    f["l4_any_sleep_present_365"] = float(((p365.category == "dep_symptom") & (p365.subtype == "sleep")).any())
    f["l4_si_present_any"] = float((pres.category == "suicidality").any())
    scr = g[(g.category == "screening") & g.value.notna()].sort_values("days")
    for name in ("phq9", "gad7"):
        v = scr[scr.subtype == name]
        if len(v):
            f[f"l4_{name}_last"] = float(v.value.iloc[0])
            if name == "phq9":
                f["l4_phq9_max"] = float(v.value.max())
    dx = g[(g.category == "depression_dx_statement") & (g.status.isin(["present", "history"])) & (g.experiencer == "patient")]
    f["l4_prevalent_flag"] = float(len(dx) > 0)
    return f


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lead", type=int, default=12); ap.add_argument("--splits", default="train,dev,test"); ap.add_argument("--limit", type=int, default=None); a = ap.parse_args()
    t0 = time.time()
    led = pd.read_parquet(OUT / "ledger.parquet", columns=["person_id", "note_id", "note_date", "category", "subtype", "status", "experiencer", "value", "grounding"])
    led = led[led.grounding != "none"].drop(columns="grounding")
    ns = pd.read_parquet(OUT / "note_summary.parquet", columns=["person_id", "note_id", "note_date", "note_relevant"])
    led_by = {k: v for k, v in led.groupby("person_id")}; ns_by = {k: v for k, v in ns.groupby("person_id")}
    log.info(f"ledger {len(led)} rows / {len(led_by)} persons; note summaries {len(ns)}; {time.time()-t0:.0f}s")
    out = []; n_after = 0
    for split in a.splits.split(","):
        path = DATASET / f"lead_{a.lead}mo" / f"{split}.jsonl"; k = 0
        with open(path) as fh:
            for line in fh:
                r = json.loads(line); k += 1
                cutoff = np.datetime64(r["cutoff_date"], "D"); pid = r["person_id"]
                base = {kk: v for kk, v in r["features"].items() if isinstance(v, (int, float))}
                row = {"person_id": pid, "split": split, "label": r["label"], "lead_months": a.lead, "age_at_cutoff": r["age_at_cutoff"], "female": int(r.get("sex") == "FEMALE"),
                       "history_days": r["history_days"], "recency_gap_days": r["recency_gap_days"], **base, **util_features(r)}
                g = led_by.get(pid); n = ns_by.get(pid)
                if n is not None:
                    n = n[n.note_date.values.astype("datetime64[D]") <= cutoff]
                if g is not None:
                    before = g.note_date.values.astype("datetime64[D]") <= cutoff; n_after += int((~before).sum()); g = g[before]
                row.update(ledger_features(g, cutoff, r["features"].get("n_notes", 0), 0 if n is None else len(n), 0 if n is None else int(n.note_relevant.eq(True).sum())))
                out.append(row)
                if a.limit and k >= a.limit:
                    break
        log.info(f"lead {a.lead} {split}: {k} rows; {time.time()-t0:.0f}s")
    df = pd.DataFrame(out); df.to_parquet(OUT / f"features_lead{a.lead}.parquet", index=False)
    summ = {"lead": a.lead, "rows": len(df), "n_features": df.shape[1] - 4, "ledger_rows_after_cutoff_dropped": n_after,
            "share_with_ledger": float((df.l4_n_ledger_notes > 0).mean()), "prevalent_flag_rate_by_label": df.groupby("label").l4_prevalent_flag.mean().to_dict(), "seconds": round(time.time() - t0)}
    write_summary(OUT / f"s4_features_lead{a.lead}_summary.json", summ); log.info(json.dumps(summ))


if __name__ == "__main__":
    main()
