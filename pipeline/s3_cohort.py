#!/usr/bin/env python3
"""Stage 3: phenotype, anchors, cutoffs, structured washout, negative anchor sampling.

Definitions (config.cohort):
  positive : person has >=1 target ICD code; anchor D = first target-code date (OMOP timeline)
  negative : person has NO target code and NO broad depression-spectrum code anywhere;
             anchor T = one in-person/telemedicine visit, sampled per person with weights that
             match the positives' anchor-year distribution
  lead     : months before anchor at which the prediction is made
  cutoff   : latest in-person/telemedicine visit on or before (anchor - lead), and no more than
             recency_max_days before (anchor - lead); features = everything on/before cutoff
  age >= 12 at anchor; obs_start <= cutoff - min_history_days; negatives: anchor + 365 <= obs_end
  and not dead before anchor + 365
  structured washout flag: broad code or washout-antidepressant prescription on/before cutoff
  (note-based washout is applied in stage 5 after L3 evidence extraction)

Outputs (cohort/):
  phenotype.parquet         one row per person: first_target_date, first_broad_date, first_ad_date, ...
  candidates.parquet        one row per (person, lead): anchor, cutoff, flags, exclusion_reason
  s3_cohort_summary.json    funnels per lead
"""
from __future__ import annotations
import argparse, time, json
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
from common import load_config, out_dir, get_logger, write_summary, icd_prefix_match, stable_hash

log = get_logger("s3")


def first_date_by_person(pf: pq.ParquetFile, person_col: str, date_col: str, mask_fn, extra_cols=()) -> pd.Series:
    """Stream a parquet file by row group; return Series person_id -> min(date) over rows where mask_fn(table) is true."""
    acc: dict[int, object] = {}
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=[person_col, date_col, *extra_cols])
        m = mask_fn(t)
        sub = t.filter(m)
        if sub.num_rows == 0:
            continue
        g = sub.group_by(person_col).aggregate([(date_col, "min")]).to_pandas()
        for pid, d in zip(g[person_col], g[f"{date_col}_min"]):
            if d is None:
                continue
            if pid not in acc or d < acc[pid]:
                acc[pid] = d
    return pd.Series(acc, name=date_col, dtype="object")


def build_phenotype(cfg, idir) -> pd.DataFrame:
    ph = cfg["phenotype"]
    dxf = pq.ParquetFile(idir / "dx.parquet")
    first_target = first_date_by_person(dxf, "person_id", "condition_start_date",
                                        lambda t: icd_prefix_match(t.column("icd"), ph["target_icd_prefixes"]), ("icd",))
    first_broad = first_date_by_person(dxf, "person_id", "condition_start_date",
                                       lambda t: icd_prefix_match(t.column("icd"), ph["broad_icd_prefixes"]), ("icd",))
    dc = pq.read_table(idir / "drug_classes.parquet").to_pandas()
    wash_ids = pa.array(dc.loc[dc.drug_class == "washout_antidepressant", "concept_id"].unique(), pa.int64())
    weak_ids = pa.array(dc.loc[dc.drug_class == "weak_antidepressant", "concept_id"].unique(), pa.int64())
    glp_ids = pa.array(dc.loc[dc.drug_class == "glp1", "concept_id"].unique(), pa.int64())
    rxf = pq.ParquetFile(idir / "rx.parquet")
    first_ad = first_date_by_person(rxf, "person_id", "drug_start_date", lambda t: pc.is_in(t.column("drug_concept_id"), value_set=wash_ids), ("drug_concept_id",))
    first_weak = first_date_by_person(rxf, "person_id", "drug_start_date", lambda t: pc.is_in(t.column("drug_concept_id"), value_set=weak_ids), ("drug_concept_id",))
    first_glp = first_date_by_person(rxf, "person_id", "drug_start_date", lambda t: pc.is_in(t.column("drug_concept_id"), value_set=glp_ids), ("drug_concept_id",))
    person = pq.read_table(idir / "person.parquet", columns=["person_id", "birth_date", "sex", "race", "ethnicity"]).to_pandas()
    op = pq.read_table(idir / "observation_period.parquet").to_pandas().groupby("person_id").agg(obs_start=("obs_start", "min"), obs_end=("obs_end", "max")).reset_index()
    death = pq.read_table(idir / "death.parquet").to_pandas().groupby("person_id").death_date.min()
    df = person.merge(op, on="person_id", how="left")
    for name, s in [("first_target_date", first_target), ("first_broad_date", first_broad), ("first_ad_date", first_ad),
                    ("first_weak_ad_date", first_weak), ("first_glp1_date", first_glp), ("death_date", death)]:
        df[name] = df.person_id.map(s)
    for c in ["birth_date", "obs_start", "obs_end", "first_target_date", "first_broad_date", "first_ad_date", "first_weak_ad_date", "first_glp1_date", "death_date"]:
        df[c] = pd.to_datetime(df[c])
    df["label_group"] = np.where(df.first_target_date.notna(), "positive", np.where(df.first_broad_date.notna(), "broad_only_excluded", "negative"))
    return df


def load_anchor_visits(cfg, idir) -> pd.DataFrame:
    excl = set(cfg["cohort"]["anchor_visit_exclude_source_values"])
    t = pq.read_table(idir / "visits.parquet", columns=["person_id", "visit_start_date", "visit_source_value", "visit_occurrence_id"])
    sv = pc.fill_null(t.column("visit_source_value"), "")
    keep = pc.invert(pc.is_in(sv, value_set=pa.array(sorted(excl))))
    v = t.filter(keep).to_pandas()
    v = v.dropna(subset=["visit_start_date"]).drop_duplicates(["person_id", "visit_start_date"])
    v["visit_start_date"] = pd.to_datetime(v.visit_start_date)
    return v.sort_values(["person_id", "visit_start_date"]).reset_index(drop=True)


class VisitIndex:
    """Vectorised 'last in-person visit on or before a date' lookups per person."""

    def __init__(self, v: pd.DataFrame):
        self.pid = v.person_id.to_numpy()
        self.days = v.visit_start_date.to_numpy().astype("datetime64[D]").astype(np.int64)
        self.vid = v.visit_occurrence_id.to_numpy()
        # composite sortable key: person rank * 2^32 + day ordinal (days fit easily)
        self.persons, self.rank = np.unique(self.pid, return_inverse=True)
        self.key = self.rank.astype(np.int64) * (1 << 32) + (self.days + (1 << 30))
        order = np.argsort(self.key, kind="stable")
        self.key, self.days, self.vid, self.rank = self.key[order], self.days[order], self.vid[order], self.rank[order]

    def last_visit_on_or_before(self, person_ids: np.ndarray, target_days: np.ndarray):
        """returns (visit_day, visit_id) arrays; visit_day = -1 where the person has no visit on/before target."""
        r = np.searchsorted(self.persons, person_ids)
        valid = (r < len(self.persons)) & (self.persons[np.minimum(r, len(self.persons) - 1)] == person_ids)
        q = np.where(valid, r, 0).astype(np.int64) * (1 << 32) + (target_days + (1 << 30))
        j = np.searchsorted(self.key, q, side="right") - 1
        ok = valid & (j >= 0)
        jj = np.maximum(j, 0)
        same = ok & (self.rank[jj] == np.where(valid, r, -1))
        day = np.where(same, self.days[jj], -1)
        vid = np.where(same, self.vid[jj], -1)
        return day, vid


def evaluate_candidates(cfg, ph: pd.DataFrame, cand: pd.DataFrame, vidx: VisitIndex, lead: int) -> pd.DataFrame:
    """cand: person_id, anchor_date, label. Adds cutoff and exclusion_reason for this lead."""
    c = cfg["cohort"]
    lead_days = int(round(lead * c["days_per_month"]))
    x = cand.merge(ph[["person_id", "birth_date", "obs_start", "obs_end", "death_date", "first_broad_date", "first_ad_date", "first_weak_ad_date", "first_target_date", "sex", "race", "ethnicity"]], on="person_id", how="left")
    x["lead_months"] = lead
    x["age_at_anchor"] = (x.anchor_date - x.birth_date).dt.days / 365.25
    target_day = (x.anchor_date - pd.to_timedelta(lead_days, unit="D")).to_numpy().astype("datetime64[D]").astype(np.int64)
    day, vid = vidx.last_visit_on_or_before(x.person_id.to_numpy(), target_day)
    x["cutoff_date"] = pd.to_datetime(np.where(day >= 0, day, 0).astype("datetime64[D]")).where(day >= 0)
    x["cutoff_visit_id"] = np.where(day >= 0, vid, -1)
    x["recency_gap_days"] = np.where(day >= 0, target_day - day, np.nan)
    x["history_days"] = (x.cutoff_date - x.obs_start).dt.days
    reasons = []
    for r in x.itertuples(index=False):
        why = []
        if pd.isna(r.birth_date) or r.age_at_anchor < c["min_age_at_anchor_years"]:
            why.append("age_lt_min")
        if pd.isna(r.cutoff_date):
            why.append("no_prior_visit")
        elif r.recency_gap_days > c["recency_max_days"]:
            why.append("recency_gap")
        if pd.isna(r.obs_start) or (not pd.isna(r.cutoff_date) and r.history_days < c["min_history_days"]):
            why.append("short_history")
        if r.label == 0:
            if pd.isna(r.obs_end) or r.anchor_date + pd.Timedelta(days=c["negative_followup_days"]) > r.obs_end:
                why.append("insufficient_followup")
            if not pd.isna(r.death_date) and r.death_date < r.anchor_date + pd.Timedelta(days=c["negative_followup_days"]):
                why.append("death_in_followup")
        reasons.append(";".join(why))
    x["exclusion_reason"] = reasons
    cut = x.cutoff_date
    x["prior_broad_code"] = x.first_broad_date.notna() & (x.first_broad_date <= cut)
    x["prior_antidepressant"] = x.first_ad_date.notna() & (x.first_ad_date <= cut)
    x["prior_weak_antidepressant"] = x.first_weak_ad_date.notna() & (x.first_weak_ad_date <= cut)
    x["prior_structured_evidence"] = x.prior_broad_code | x.prior_antidepressant
    x["recency_within_90"] = x.recency_gap_days <= c["recency_sensitivity_days"]
    return x


def sample_negative_anchors(cfg, ph: pd.DataFrame, v: pd.DataFrame, vidx: VisitIndex, pos_year_dist: pd.Series, seed: int) -> pd.DataFrame:
    """One anchor per negative person, chosen among visits that satisfy the lead-12 constraints,
    weighted so the anchor-year distribution approaches the positives' distribution."""
    c = cfg["cohort"]
    neg = ph[ph.label_group == "negative"][["person_id", "birth_date", "obs_start", "obs_end", "death_date"]]
    vv = v.merge(neg, on="person_id", how="inner")
    vv["age"] = (vv.visit_start_date - vv.birth_date).dt.days / 365.25
    fu = pd.Timedelta(days=c["negative_followup_days"])
    ok = (vv.age >= c["min_age_at_anchor_years"]) & (vv.visit_start_date + fu <= vv.obs_end) & (vv.death_date.isna() | (vv.death_date >= vv.visit_start_date + fu))
    vv = vv[ok]
    lead12 = int(round(max(c["leads_months"]) * c["days_per_month"]))
    target_day = (vv.visit_start_date - pd.to_timedelta(lead12, unit="D")).to_numpy().astype("datetime64[D]").astype(np.int64)
    day, _ = vidx.last_visit_on_or_before(vv.person_id.to_numpy(), target_day)
    gap = np.where(day >= 0, target_day - day, np.inf)
    hist_ok = (pd.to_datetime(np.where(day >= 0, day, 0).astype("datetime64[D]")) - vv.obs_start.to_numpy()).days >= c["min_history_days"]
    vv = vv[(gap <= c["recency_max_days"]) & hist_ok]
    vv["year"] = vv.visit_start_date.dt.year
    q = vv.year.value_counts(normalize=True)
    w = vv.year.map(lambda y: pos_year_dist.get(y, 0.0) / q.get(y, 1.0))
    vv = vv.assign(w=w.fillna(0.0))
    rng = np.random.default_rng(seed)
    vv["u"] = rng.random(len(vv))
    # weighted sampling of one row per person: Efraimidis-Spirakis keys  u^(1/w)
    with np.errstate(divide="ignore", invalid="ignore"):
        vv["k"] = np.where(vv.w > 0, vv.u ** (1.0 / vv.w), -1.0)
    pick = vv.sort_values(["person_id", "k"], ascending=[True, False]).drop_duplicates("person_id")
    pick = pick[pick.k >= 0]
    return pick[["person_id", "visit_start_date"]].rename(columns={"visit_start_date": "anchor_date"}).assign(label=0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--index-subdir", default="index"); ap.add_argument("--out-subdir", default="cohort"); args = ap.parse_args()
    cfg = load_config(); idir = out_dir(cfg, args.index_subdir); odir = out_dir(cfg, args.out_subdir)
    t0 = time.time()
    ph = build_phenotype(cfg, idir)
    pq.write_table(pa.Table.from_pandas(ph, preserve_index=False), odir / "phenotype.parquet")
    log.info(f"phenotype: {ph.label_group.value_counts().to_dict()} ({time.time()-t0:.0f}s)")
    v = load_anchor_visits(cfg, idir); vidx = VisitIndex(v)
    log.info(f"anchor-eligible visits: {len(v):,} for {v.person_id.nunique():,} persons")
    pos = ph[ph.label_group == "positive"][["person_id", "first_target_date"]].rename(columns={"first_target_date": "anchor_date"}).assign(label=1)
    leads = cfg["cohort"]["leads_months"]
    frames = []; summary = {"phenotype_counts": ph.label_group.value_counts().to_dict(), "leads": {}}
    pos_eval = {L: evaluate_candidates(cfg, ph, pos, vidx, L) for L in leads}
    main_lead = max(leads)
    pos_ok = pos_eval[main_lead][pos_eval[main_lead].exclusion_reason == ""]
    pos_year_dist = pos_ok.anchor_date.dt.year.value_counts(normalize=True)
    neg = sample_negative_anchors(cfg, ph, v, vidx, pos_year_dist, cfg["seed"])
    log.info(f"positives eligible at lead {main_lead}: {len(pos_ok):,}; negative anchors sampled: {len(neg):,}")
    for L in leads:
        pe = pos_eval[L]; ne = evaluate_candidates(cfg, ph, neg, vidx, L)
        x = pd.concat([pe, ne], ignore_index=True)
        frames.append(x)
        el = x[x.exclusion_reason == ""]
        summary["leads"][L] = {
            "candidates": {"positive": int((x.label == 1).sum()), "negative": int((x.label == 0).sum())},
            "eligible": {"positive": int((el.label == 1).sum()), "negative": int((el.label == 0).sum())},
            "exclusion_reasons_positive": x[x.label == 1].exclusion_reason.replace("", "eligible").value_counts().to_dict(),
            "exclusion_reasons_negative": x[x.label == 0].exclusion_reason.replace("", "eligible").value_counts().to_dict(),
            "prior_structured_evidence_among_eligible": el.groupby("label").prior_structured_evidence.mean().round(4).to_dict(),
            "recency_within_90_among_eligible": el.groupby("label").recency_within_90.mean().round(4).to_dict(),
            "anchor_year_dist_eligible": {int(k): v for k, v in pd.crosstab(el.anchor_date.dt.year, el.label, normalize="columns").round(3).to_dict(orient="index").items()},
            "age_at_anchor_median": el.groupby("label").age_at_anchor.median().round(2).to_dict(),
        }
    allc = pd.concat(frames, ignore_index=True)
    allc["cutoff_visit_id"] = allc.cutoff_visit_id.astype("int64")
    pq.write_table(pa.Table.from_pandas(allc, preserve_index=False), odir / "candidates.parquet")
    summary["seconds"] = round(time.time() - t0, 1)
    write_summary(odir / "s3_cohort_summary.json", summary)
    log.info(json.dumps(summary, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
