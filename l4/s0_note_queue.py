#!/usr/bin/env python3
"""Step 0: build the per-note extraction queue for the final cohort.

Rules (plan §6.1):
  * persons in outputs/cohort/final_cohort.parquet only
  * note_date <= the person's LATEST cutoff across leads (lead 3 if present); note_date is already on the OMOP timeline
  * informative notes only: chars_condensed >= 150 or mh_relevance > 0 (same rule as s6's trivial-note filter)
  * dedupe by note_id across all notes shards
  * tier1 = mh_relevance > 0 or sections intersect {hpi, social, psych_mh, assessment_plan}; tier2 = rest
Outputs: outputs/l4/queue/part_XX.parquet (with text), outputs/l4/note_queue_index.parquet (no text), s0_queue_summary.json
"""
from __future__ import annotations
import argparse, json, time
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
from l4_common import OUT, QUEUE_DIR, NOTES_DIR, COHORT, MAX_TEXT_CHARS, NARRATIVE_SECTIONS, get_logger, write_summary

log = get_logger("s0")
COLS = ["note_id", "person_id", "note_date", "note_type", "author_type", "author_specialty", "chars_condensed",
        "mh_relevance", "sections", "condensed_text", "evidence_json"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--min-chars", type=int, default=150); ap.add_argument("--limit-parts", type=int, default=None); a = ap.parse_args()
    t0 = time.time(); QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    coh = pq.read_table(COHORT, columns=["person_id", "lead_months", "cutoff_date", "label", "split"]).to_pandas()
    coh["cutoff_date"] = pd.to_datetime(coh["cutoff_date"])
    per = coh.groupby("person_id").agg(max_cutoff=("cutoff_date", "max"), label=("label", "max"), split=("split", "first")).reset_index()
    max_cut = dict(zip(per.person_id.astype(np.int64), per.max_cutoff.values.astype("datetime64[D]")))
    log.info(f"cohort persons {len(per)}; leads/person mean {coh.groupby('person_id').size().mean():.2f}")
    seen: set[int] = set(); index_frames = []; stats = {"parts": 0, "rows_read": 0, "in_cohort": 0, "before_cutoff": 0, "informative": 0, "deduped": 0, "tier1": 0, "tier2": 0, "chars": 0}
    parts = sorted(NOTES_DIR.glob("part_*.parquet"))[: a.limit_parts]
    for p in parts:
        t = pq.read_table(p, columns=COLS); stats["rows_read"] += t.num_rows; stats["parts"] += 1
        pid = t["person_id"].to_numpy()
        m_coh = np.fromiter((x in max_cut for x in pid), bool, len(pid)); t = t.filter(pa.array(m_coh)); stats["in_cohort"] += t.num_rows
        if t.num_rows == 0:
            continue
        pid = t["person_id"].to_numpy(); nd = t["note_date"].to_numpy().astype("datetime64[D]")
        cut = np.array([max_cut[x] for x in pid], dtype="datetime64[D]")
        m_cut = nd <= cut; t = t.filter(pa.array(m_cut)); stats["before_cutoff"] += t.num_rows
        cc = t["chars_condensed"].to_numpy(); mr = t["mh_relevance"].to_numpy()
        m_inf = (cc >= a.min_chars) | (mr > 0); t = t.filter(pa.array(m_inf)); stats["informative"] += t.num_rows
        nid = t["note_id"].to_numpy(); m_new = np.fromiter((x not in seen for x in nid), bool, len(nid)); t = t.filter(pa.array(m_new)); seen.update(nid[m_new].tolist()); stats["deduped"] += t.num_rows
        if t.num_rows == 0:
            continue
        df = t.to_pandas()
        secs = df["sections"].map(lambda s: set(json.loads(s)) if s else set())
        df["tier"] = np.where((df["mh_relevance"] > 0) | secs.map(lambda s: bool(s & NARRATIVE_SECTIONS)), 1, 2).astype(np.int8)
        df["text"] = df["condensed_text"].str.slice(0, MAX_TEXT_CHARS); df["chars_text"] = df["text"].str.len().astype(np.int32)
        df["label"] = df["person_id"].map(dict(zip(per.person_id, per.label))).astype(np.int8); df["split"] = df["person_id"].map(dict(zip(per.person_id, per.split)))
        df = df.drop(columns=["condensed_text"]).sort_values(["person_id", "note_date"])
        out = QUEUE_DIR / p.name; df.to_parquet(out, index=False)
        stats["tier1"] += int((df.tier == 1).sum()); stats["tier2"] += int((df.tier == 2).sum()); stats["chars"] += int(df.chars_text.sum())
        index_frames.append(df.drop(columns=["text", "evidence_json"]).assign(part=p.name))
        log.info(f"{p.name}: read {t.num_rows if False else len(df)} kept (tier1 {(df.tier==1).sum()}) | cumulative {stats['deduped']} | {time.time()-t0:.0f}s")
    idx = pd.concat(index_frames, ignore_index=True); idx.to_parquet(OUT / "note_queue_index.parquet", index=False)
    stats["est_input_tokens"] = stats["chars"] / 4; stats["persons_with_notes"] = int(idx.person_id.nunique())
    stats["notes_per_person_median"] = float(idx.groupby("person_id").size().median())
    stats["by_note_type"] = idx.note_type.value_counts().head(15).to_dict(); stats["by_split"] = idx.split.value_counts().to_dict()
    stats["seconds"] = round(time.time() - t0)
    write_summary(OUT / "s0_queue_summary.json", stats); log.info(json.dumps({k: v for k, v in stats.items() if not isinstance(v, dict)}))


if __name__ == "__main__":
    main()
