#!/usr/bin/env python3
"""Stage 2: align note-metadata dates onto the OMOP timeline.

Finding (2026-09-04): note metadata and OMOP carry two independent per-patient date shifts,
offset uniform in [-30, +30] days and constant within a patient.
For every patient we estimate  delta = median(EncounterDateShifted - visit_start_date)  over
notes whose encounter_key (or primary_encounter_key) matches a visit_occurrence_id of the same
person. Aligned note date:
   matched visit           -> visit_start_date
   unmatched, delta known  -> EncounterDateShifted - delta   (fallback: service/contact/created date)
   unmatched, delta unknown-> null (note unusable for time-window logic)

Outputs (index/):
  note_date_offset.parquet : person_id, delta_days, n_matched, frac_at_median, offset_quality
  note_meta_aligned.parquet: note_meta + visit_occurrence_id, note_date (aligned), date_source
"""
from __future__ import annotations
import argparse, time
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
from common import refuse_archived, load_config, out_dir, get_logger, write_summary

log = get_logger("s2")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--index-subdir", default="index"); args = ap.parse_args()
    cfg = load_config(); refuse_archived(cfg, args.index_subdir); idir = out_dir(cfg, args.index_subdir)
    t0 = time.time()
    nm = pq.read_table(idir / "note_meta.parquet").to_pandas()
    vis = pq.read_table(idir / "visits.parquet", columns=["visit_occurrence_id", "person_id", "visit_start_date"]).to_pandas()
    vis = vis.drop_duplicates("visit_occurrence_id").set_index("visit_occurrence_id")
    log.info(f"notes {len(nm):,} visits {len(vis):,} loaded {time.time()-t0:.0f}s")
    # match encounter_key first, then primary_encounter_key; require the visit to belong to the same person
    vid = pd.Series(pd.NA, index=nm.index, dtype="Int64")
    for key in ["encounter_key", "primary_encounter_key"]:
        owner = nm[key].map(vis["person_id"])
        ok = owner.notna() & (owner == nm["person_id"]) & vid.isna()
        vid[ok] = nm.loc[ok, key].astype("Int64")
    nm["visit_occurrence_id"] = vid
    nm["visit_occurrence_id"] = nm["visit_occurrence_id"].astype("Int64")
    nm["visit_start_date"] = nm["visit_occurrence_id"].map(vis["visit_start_date"])
    matched = nm["visit_start_date"].notna()
    base = nm["encounter_date"].fillna(nm["service_date"]).fillna(nm["contact_date"]).fillna(nm["created_date"])
    nm["base_date"] = pd.to_datetime(base)
    d = (nm["base_date"] - pd.to_datetime(nm["visit_start_date"])).dt.days
    dm = d[matched & nm["encounter_date"].notna()]
    # per-patient offset: median of the note-level differences, plus agreement rate
    g = dm.groupby(nm.loc[dm.index, "person_id"])
    off = pd.DataFrame({"delta_days": g.median(), "n_matched": g.size()})
    off["delta_days"] = off["delta_days"].round().astype(int)
    agree = (dm - nm.loc[dm.index, "person_id"].map(off["delta_days"])).abs() <= 1
    off["frac_within_1d"] = agree.groupby(nm.loc[dm.index, "person_id"]).mean()
    off["offset_quality"] = np.where((off.n_matched >= 3) & (off.frac_within_1d >= 0.8), "good",
                                     np.where(off.n_matched >= 1, "weak", "none"))
    off = off.reset_index().rename(columns={"index": "person_id"})
    pq.write_table(pa.Table.from_pandas(off, preserve_index=False), idir / "note_date_offset.parquet")
    # aligned date
    delta = nm["person_id"].map(off.set_index("person_id")["delta_days"])
    aligned = np.where(matched, pd.to_datetime(nm["visit_start_date"]),
                       np.where(delta.notna() & nm["base_date"].notna(), nm["base_date"] - pd.to_timedelta(delta.fillna(0), unit="D"), pd.NaT))
    nm["note_date"] = pd.to_datetime(aligned).date
    nm["date_source"] = np.where(matched, "visit", np.where(delta.notna() & nm["base_date"].notna(), "shifted_by_patient_delta", "unaligned"))
    nm = nm.drop(columns=["visit_start_date", "base_date"])
    out = pa.Table.from_pandas(nm, preserve_index=False)
    pq.write_table(out, idir / "note_meta_aligned.parquet", compression="zstd")
    # residual diagnostics on matched notes
    resid = (dm - nm.loc[dm.index, "person_id"].map(off.set_index("person_id")["delta_days"]))
    info = {"notes": len(nm), "matched_to_visit": int(matched.sum()), "matched_frac": round(matched.mean(), 4),
            "patients_with_offset": len(off), "offset_quality_counts": off.offset_quality.value_counts().to_dict(),
            "delta_quantiles": off.delta_days.quantile([0, .05, .25, .5, .75, .95, 1]).round(1).to_dict(),
            "residual_abs_le1_frac": round((resid.abs() <= 1).mean(), 4), "residual_abs_gt7_frac": round((resid.abs() > 7).mean(), 4),
            "date_source_counts": nm.date_source.value_counts().to_dict(), "seconds": round(time.time() - t0, 1)}
    write_summary(idir / "s2_align_summary.json", info); log.info(info)


if __name__ == "__main__":
    main()
