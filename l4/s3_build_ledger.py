#!/usr/bin/env python3
"""Step 3b: merge raw extractions -> validated ledger.

  python s3_build_ledger.py --tags full [--tags full,pilot]
Reads outputs/l4/extractions/<tag>_tier*_shard*.jsonl, re-reads note text from the queue for span grounding, writes
  outputs/l4/ledger.parquet        one row per validated evidence item
  outputs/l4/note_summary.parquet  one row per extracted note (note_relevant, counts, summary, error)
  outputs/l4/s3_ledger_summary.json"""
from __future__ import annotations
import argparse, json, time
import numpy as np, pandas as pd, pyarrow.parquet as pq
from l4_common import OUT, QUEUE_DIR, EXTRACT_DIR, get_logger, write_summary
from validate import validate_extraction

log = get_logger("s3")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", default="full"); a = ap.parse_args()
    t0 = time.time(); tags = a.tags.split(",")
    files = [p for t in tags for p in sorted(EXTRACT_DIR.glob(f"{t}_tier*_shard*.jsonl"))]
    log.info(f"{len(files)} extraction files")
    raw = {}
    for p in files:
        with open(p) as f:
            for line in f:
                e = json.loads(line); raw[e["note_id"]] = e     # last write wins (dedupe across reruns)
    log.info(f"{len(raw)} extracted notes; loading text from queue")
    idx = pd.read_parquet(OUT / "note_queue_index.parquet", columns=["note_id", "part", "person_id", "note_date", "note_type", "tier", "split", "label"])
    idx = idx[idx.note_id.isin(raw.keys())]
    rows = []; notes = []
    for part, g in idx.groupby("part"):
        t = pq.read_table(QUEUE_DIR / part, columns=["note_id", "text"], filters=[("note_id", "in", g.note_id.tolist())]).to_pandas().set_index("note_id")
        for nid, text in t.text.items():
            e = raw[nid]; items, st = validate_extraction(e["raw"], text)
            for it in items:
                rows.append({"note_id": nid, "person_id": e["person_id"], "note_date": e["note_date"], **it})
            notes.append({"note_id": nid, "person_id": e["person_id"], "note_date": e["note_date"], "tier": e["tier"], "json_ok": e["raw"] is not None, "error": e["error"],
                          "finish": e["finish"], "out_tokens": e["out_tokens"], **st})
    led = pd.DataFrame(rows); ns = pd.DataFrame(notes)
    led["note_date"] = pd.to_datetime(led["note_date"]); ns["note_date"] = pd.to_datetime(ns["note_date"])
    led.to_parquet(OUT / "ledger.parquet", index=False); ns.to_parquet(OUT / "note_summary.parquet", index=False)
    summ = {"n_notes": len(ns), "json_valid_rate": float(ns.json_ok.mean()), "n_items": int(ns.n_items.sum()), "n_kept_grounded": int((led.grounding != "none").sum()), "n_ungrounded_kept": int((led.grounding == "none").sum()),
            "grounding_rate": float(1 - ns.n_dropped_span.sum() / max(ns.n_items.sum() - ns.n_dropped_type.sum(), 1)), "note_relevant_rate": float(ns.note_relevant.fillna(False).mean()),
            "persons": int(ns.person_id.nunique()), "items_by_category": led.category.value_counts().to_dict(), "items_by_status": led.status.value_counts().to_dict(),
            "coverage_of_queue_by_tier": {int(k): float(v) for k, v in (ns.groupby("tier").size() / pd.read_parquet(OUT / "note_queue_index.parquet", columns=["tier"]).tier.value_counts()).items()},
            "seconds": round(time.time() - t0)}
    write_summary(OUT / "s3_ledger_summary.json", summ); log.info(json.dumps({k: v for k, v in summ.items() if not isinstance(v, dict)}))


if __name__ == "__main__":
    main()
