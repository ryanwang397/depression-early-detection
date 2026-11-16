#!/usr/bin/env python3
"""Stage 5: note-evidence washout, final cohort per lead, patient-level stratified splits.

Prior-evidence rule (symmetric for positives and negatives), evaluated on/before cutoff:
  structured : broad depression-spectrum code, or washout-antidepressant prescription   (stage 3)
  notes      : any whitelisted note with dep_asserted==1 (specific diagnosis / explicit assertion,
               current or history, patient-level) or PHQ-9 >= 10 or 'Does this patient have depression: Yes'
Main cohort  = eligible & no prior evidence & >=1 processed note on/before cutoff.
Positives with prior evidence -> case_type 'prevalent' (kept in excluded_prior_evidence.parquet).
Split: one split per person (same across leads), stratified by label x sex x anchor-year bin,
deterministic BLAKE2b ordering.
"""
from __future__ import annotations
import argparse, glob, json, time
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
from common import load_config, out_dir, get_logger, write_summary, stable_hash

log = get_logger("s5")


def refine_dep_asserted(evidence_json: str, features_json: str) -> int:
    """Re-evaluate note-level depression assertion for parts produced before rules v1.6:
    drop dep_dx hits that are a bare ambiguous 'MDD' token without diagnosis context."""
    import re as _re
    from note_text import mdd_is_depression
    f = json.loads(features_json)
    if f.get("score_has_depression_field") == "yes":
        return 1
    for e in json.loads(evidence_json):
        if e["category"] in ("dep_dx", "dep_assert") and e["status"] in ("present", "history"):
            span = e["span"]
            m = _re.search(r"\bMDD\b", span)
            if e["category"] == "dep_dx" and m and not _re.search(r"depress|dysthym|F3[23]", span, _re.I):
                if not mdd_is_depression(span, m.start(), m.end()):
                    continue
            return 1
    return 0


def note_evidence_by_cutoff(cand: pd.DataFrame, ndir) -> pd.DataFrame:
    cols = ["person_id", "note_date", "dep_asserted", "phq9_positive", "mh_relevance", "linkage_note_id", "note_type", "chars_condensed"]
    parts = []
    for p in sorted(glob.glob(str(ndir / "part_*.parquet"))):
        t = pq.read_table(p, columns=cols + ["evidence_json", "features_json", "rules_version"])
        d = t.drop_columns(["evidence_json", "features_json", "rules_version"]).to_pandas()
        flagged = pc.equal(t.column("dep_asserted"), 1)
        if pc.any(flagged).as_py() and t.column("rules_version")[0].as_py() < "notes_v1.6":
            sub = t.filter(flagged).select(["linkage_note_id", "evidence_json", "features_json"]).to_pandas()
            refined = {int(r.linkage_note_id): refine_dep_asserted(r.evidence_json, r.features_json) for r in sub.itertuples(index=False)}
            d["dep_asserted"] = d.apply(lambda r: refined.get(int(r.linkage_note_id), r.dep_asserted) if r.dep_asserted == 1 else 0, axis=1)
        parts.append(d)
    notes = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    notes["note_date"] = pd.to_datetime(notes.note_date)
    # informative = not a short, mental-health-irrelevant telephone/administrative note
    notes["informative"] = ~((notes.note_type == "Telephone Encounter") & (notes.chars_condensed < 150) & (notes.mh_relevance == 0))
    out = []
    for lead, c in cand.groupby("lead_months"):
        m = notes.merge(c[["person_id", "cutoff_date"]], on="person_id", how="inner")
        m = m[m.note_date <= m.cutoff_date]
        g = m.groupby("person_id").agg(n_notes=("informative", "sum"), n_notes_all=("linkage_note_id", "size"), prior_note_evidence=("dep_asserted", "max"),
                                       prior_phq9_positive=("phq9_positive", "max"), n_notes_mh=("mh_relevance", lambda s: int((s > 0).sum())),
                                       last_note_date=("note_date", "max"))
        g["lead_months"] = lead
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def assign_splits(df: pd.DataFrame, cfg) -> pd.Series:
    """df: one row per person with label, sex, anchor_year. Returns split per person."""
    fr = cfg["cohort"]["split_fractions"]; seed = cfg["seed"]
    df = df.copy()
    df["ybin"] = pd.cut(df.anchor_year, [2000, 2015, 2018, 2020, 2022, 2023, 2030], labels=False)
    df["h"] = [stable_hash(seed, "split", p) for p in df.person_id]
    split = pd.Series(index=df.index, dtype=object)
    for _, g in df.groupby(["label", "sex", "ybin"], dropna=False):
        g = g.sort_values("h")
        n = len(g); n_tr = int(round(n * fr["train"])); n_dv = int(round(n * fr["dev"]))
        s = np.array(["train"] * n_tr + ["dev"] * n_dv + ["test"] * (n - n_tr - n_dv))
        split.loc[g.index] = s
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-subdir", default="cohort"); ap.add_argument("--notes-subdir", default="notes"); ap.add_argument("--out-subdir", default="cohort")
    args = ap.parse_args()
    cfg = load_config(); cdir = out_dir(cfg, args.cohort_subdir); ndir = out_dir(cfg, args.notes_subdir); odir = out_dir(cfg, args.out_subdir)
    t0 = time.time()
    cand = pq.read_table(cdir / "candidates.parquet").to_pandas()
    cand = cand[cand.exclusion_reason == ""].copy()
    cand["anchor_date"] = pd.to_datetime(cand.anchor_date); cand["cutoff_date"] = pd.to_datetime(cand.cutoff_date)
    ne = note_evidence_by_cutoff(cand, ndir)
    x = cand.merge(ne, on=["person_id", "lead_months"], how="left")
    x["n_notes"] = x.n_notes.fillna(0).astype(int); x["n_notes_all"] = x.n_notes_all.fillna(0).astype(int); x["n_notes_mh"] = x.n_notes_mh.fillna(0).astype(int)
    x["prior_note_evidence"] = x.prior_note_evidence.fillna(0).astype(int).astype(bool) | x.prior_phq9_positive.fillna(0).astype(int).astype(bool)
    x["prior_any_evidence"] = x.prior_structured_evidence | x.prior_note_evidence
    x["has_notes"] = x.n_notes > 0
    x["case_type"] = np.where(x.label == 1, np.where(x.prior_any_evidence, "prevalent", "incident"), np.where(x.prior_any_evidence, "negative_prior_evidence", "negative"))
    x["anchor_year"] = x.anchor_date.dt.year
    # split per person on the union of persons that are in the main cohort in at least one lead
    main_mask = (~x.prior_any_evidence) & x.has_notes
    persons = x[main_mask].sort_values("lead_months", ascending=False).drop_duplicates("person_id")[["person_id", "label", "sex", "anchor_year"]]
    persons["split"] = assign_splits(persons, cfg).values
    x = x.merge(persons[["person_id", "split"]], on="person_id", how="left")
    main = x[main_mask].copy(); excl = x[~main_mask].copy()
    pq.write_table(pa.Table.from_pandas(main, preserve_index=False), odir / "final_cohort.parquet")
    pq.write_table(pa.Table.from_pandas(excl, preserve_index=False), odir / "excluded_prior_evidence_or_no_notes.parquet")
    summary = {"leads": {}, "seconds": round(time.time() - t0, 1)}
    for lead, g in x.groupby("lead_months"):
        m = g[main_mask.loc[g.index]]
        summary["leads"][int(lead)] = {
            "eligible_structured": g.label.value_counts().to_dict(),
            "prior_structured_evidence": g.groupby("label").prior_structured_evidence.mean().round(4).to_dict(),
            "prior_note_evidence": g.groupby("label").prior_note_evidence.mean().round(4).to_dict(),
            "prior_any_evidence": g.groupby("label").prior_any_evidence.mean().round(4).to_dict(),
            "no_notes_before_cutoff": g.groupby("label").has_notes.apply(lambda s: round(1 - s.mean(), 4)).to_dict(),
            "main_cohort": m.label.value_counts().to_dict(), "positive_rate": round(m.label.mean(), 4) if len(m) else None,
            "case_type_counts": g.case_type.value_counts().to_dict(),
            "split_counts": pd.crosstab(m.split, m.label).to_dict() if len(m) else {},
            "median_notes_before_cutoff": m.groupby("label").n_notes.median().to_dict() if len(m) else {},
            "median_age_at_anchor": m.groupby("label").age_at_anchor.median().round(2).to_dict() if len(m) else {},
            "anchor_year_by_label": pd.crosstab(m.anchor_year, m.label, normalize="columns").round(3).to_dict() if len(m) else {},
        }
    write_summary(odir / "s5_finalize_summary.json", summary); log.info(json.dumps(summary, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
