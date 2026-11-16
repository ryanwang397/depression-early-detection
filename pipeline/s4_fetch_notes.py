#!/usr/bin/env python3
"""Stage 4: fetch and process clinical notes for cohort candidates (L1 clean, L2 condense, L3 evidence).

Selection: persons eligible in ANY lead (candidates.parquet, exclusion_reason == ""), notes of a
whitelisted InpatientNoteType, with an aligned OMOP-timeline date on/before that person's latest
cutoff over leads (so no note after any cutoff is ever fetched).
Sharded by person_id % n_shards so it can run as a slurm array.

Output: notes/part_<shard>.parquet  (one row per note; clean + condensed text, evidence JSON, flat features)
"""
from __future__ import annotations
import argparse, json, time
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
from common import refuse_archived, load_config, out_dir, get_logger, write_summary, fetch_note_texts, ParquetSink
from note_text import process_note, RULES_VERSION

log = get_logger("s4")

FLAT = ["mh_relevance", "dep_asserted", "dep_current_asserted", "si_present", "si_denied", "phq9_positive", "phq2_positive", "n_template_phrases"]
SCHEMA = pa.schema([
    ("linkage_note_id", pa.int64()), ("note_id", pa.int64()), ("person_id", pa.int64()), ("note_date", pa.date32()),
    ("date_source", pa.string()), ("visit_occurrence_id", pa.int64()), ("note_type", pa.string()), ("author_type", pa.string()),
    ("author_specialty", pa.string()), ("chars_raw", pa.int32()), ("chars_clean", pa.int32()), ("chars_condensed", pa.int32()),
    ("clean_text", pa.string()), ("condensed_text", pa.string()), ("sections", pa.string()), ("evidence_json", pa.string()),
    ("features_json", pa.string()), ("mh_relevance", pa.int16()), ("dep_asserted", pa.int8()), ("dep_current_asserted", pa.int8()),
    ("si_present", pa.int8()), ("si_denied", pa.int8()), ("phq9_positive", pa.int8()), ("phq2_positive", pa.int8()),
    ("phq9", pa.int16()), ("phq2", pa.int16()), ("gad7", pa.int16()), ("psych_med_present", pa.int16()), ("n_template_phrases", pa.int16()),
    ("rules_version", pa.string()),
])


def select_notes(cfg, idir, cdir, shard: int, n_shards: int, limit_persons: int | None) -> pd.DataFrame:
    cand = pq.read_table(cdir / "candidates.parquet", columns=["person_id", "cutoff_date", "exclusion_reason"]).to_pandas()
    cand = cand[cand.exclusion_reason == ""]
    maxcut = cand.groupby("person_id").cutoff_date.max()
    persons = maxcut.index.to_numpy()
    persons = persons[(persons % n_shards) == shard]
    if limit_persons:
        persons = persons[:limit_persons]
    pset = pa.array(np.sort(persons), pa.int64())
    wl = pa.array(sorted(cfg["notes"]["whitelist_types"]))
    pf = pq.ParquetFile(idir / "note_meta_aligned.parquet")
    cols = ["linkage_note_id", "note_id", "person_id", "note_date", "date_source", "visit_occurrence_id", "inpatient_note_type", "author_type", "author_specialty"]
    parts = []
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=cols)
        m = pc.and_(pc.is_in(t.column("person_id"), value_set=pset),
                    pc.is_in(pc.utf8_lower(pc.fill_null(t.column("inpatient_note_type"), "")), value_set=wl))
        m = pc.and_(m, pc.not_equal(pc.fill_null(t.column("date_source"), "unaligned"), "unaligned"))
        sub = t.filter(m)
        if sub.num_rows:
            parts.append(sub.to_pandas())
    nm = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    nm["note_date"] = pd.to_datetime(nm.note_date)
    nm["max_cutoff"] = nm.person_id.map(maxcut)
    nm = nm[nm.note_date <= nm.max_cutoff].drop(columns=["max_cutoff"])
    # offsets
    ids = pa.array(np.sort(nm.linkage_note_id.unique()), pa.int64())
    of = pq.ParquetFile(idir / "note_offsets.parquet")
    oparts = []
    for rg in range(of.num_row_groups):
        t = of.read_row_group(rg)
        sub = t.filter(pc.is_in(t.column("linkage_note_id"), value_set=ids))
        if sub.num_rows:
            oparts.append(sub.to_pandas())
    off = pd.concat(oparts, ignore_index=True) if oparts else pd.DataFrame(columns=["linkage_note_id", "file_idx", "offset", "length"])
    off = off.sort_values(["linkage_note_id", "file_idx", "offset"]).drop_duplicates("linkage_note_id")
    nm = nm.drop_duplicates("linkage_note_id").merge(off[["linkage_note_id", "file_idx", "offset", "length"]], on="linkage_note_id", how="left")
    return nm, len(persons)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit-persons", type=int, default=None)
    ap.add_argument("--index-subdir", default="index"); ap.add_argument("--cohort-subdir", default="cohort"); ap.add_argument("--out-subdir", default="notes")
    ap.add_argument("--batch", type=int, default=2000)
    args = ap.parse_args()
    cfg = load_config(); refuse_archived(cfg, args.out_subdir); idir = out_dir(cfg, args.index_subdir); cdir = out_dir(cfg, args.cohort_subdir); odir = out_dir(cfg, args.out_subdir)
    t0 = time.time()
    nm, n_persons = select_notes(cfg, idir, cdir, args.shard, args.n_shards, args.limit_persons)
    n_missing = int(nm.file_idx.isna().sum())
    nm = nm[nm.file_idx.notna()].sort_values(["file_idx", "offset"]).reset_index(drop=True)
    log.info(f"shard {args.shard}/{args.n_shards}: persons {n_persons:,}, notes selected {len(nm)+n_missing:,}, without text offset {n_missing:,} ({time.time()-t0:.0f}s)")
    sink = ParquetSink(odir / f"part_{args.shard:02d}.parquet", SCHEMA)
    rows = []; agg = {"mh_relevance_gt0": 0, "dep_asserted": 0, "phq9_positive": 0, "si_present": 0}
    for start in range(0, len(nm), args.batch):
        chunk = nm.iloc[start:start + args.batch]
        reqs = [(int(r.linkage_note_id), int(r.file_idx), int(r.offset), int(r.length)) for r in chunk.itertuples(index=False)]
        texts = fetch_note_texts(cfg, reqs)
        for r in chunk.itertuples(index=False):
            raw = texts.get(int(r.linkage_note_id), "")
            raw = raw[: cfg["notes"]["max_note_chars_raw"]]
            p = process_note(raw)
            f = p["features"]
            agg["mh_relevance_gt0"] += f["mh_relevance"] > 0; agg["dep_asserted"] += f["dep_asserted"]; agg["phq9_positive"] += f["phq9_positive"]; agg["si_present"] += f["si_present"]
            rows.append({
                "linkage_note_id": int(r.linkage_note_id), "note_id": int(r.note_id), "person_id": int(r.person_id), "note_date": r.note_date.date(),
                "date_source": r.date_source, "visit_occurrence_id": (int(r.visit_occurrence_id) if pd.notna(r.visit_occurrence_id) else None),
                "note_type": r.inpatient_note_type, "author_type": r.author_type, "author_specialty": r.author_specialty,
                "chars_raw": len(raw), "chars_clean": len(p["clean"]), "chars_condensed": len(p["condensed"]),
                "clean_text": p["clean"], "condensed_text": p["condensed"], "sections": json.dumps(p["l2"]["sections"]),
                "evidence_json": json.dumps(p["evidence"]), "features_json": json.dumps(f),
                "mh_relevance": int(f["mh_relevance"]), "dep_asserted": int(f["dep_asserted"]), "dep_current_asserted": int(f["dep_current_asserted"]),
                "si_present": int(f["si_present"]), "si_denied": int(f["si_denied"]), "phq9_positive": int(f["phq9_positive"]), "phq2_positive": int(f["phq2_positive"]),
                "phq9": int(f.get("score_phq9", -1)), "phq2": int(f.get("score_phq2", -1)), "gad7": int(f.get("score_gad7", -1)),
                "psych_med_present": int(f.get("psych_med__present", 0)), "n_template_phrases": int(f["n_template_phrases"]), "rules_version": RULES_VERSION,
            })
        if len(rows) >= args.batch:
            sink.write(pa.Table.from_pylist(rows, schema=SCHEMA)); rows = []
            log.info(f"  processed {sink.rows:,}/{len(nm):,} notes ({time.time()-t0:.0f}s)")
    if rows:
        sink.write(pa.Table.from_pylist(rows, schema=SCHEMA))
    sink.close()
    info = {"shard": args.shard, "n_shards": args.n_shards, "persons": n_persons, "notes_selected": len(nm) + n_missing, "notes_without_offset": n_missing,
            "notes_processed": sink.rows, "rules_version": RULES_VERSION, "agg": agg, "seconds": round(time.time() - t0, 1)}
    write_summary(odir / f"s4_part_{args.shard:02d}_summary.json", info); log.info(info)


if __name__ == "__main__":
    main()
