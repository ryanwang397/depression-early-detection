"""Stage 6b: build the study dataset = main-cohort records followed by the prevalent-positive records, per split.

  outputs/dataset_main/lead_{L}mo/{split}.jsonl     s6_assemble.py --lead L --split S --out-subdir dataset_main
  outputs/dataset_extras/lead_{L}mo/{split}.jsonl   s6_assemble.py --lead L --split S --cohort-subdir cohort_extras_lead{L} --out-subdir dataset_extras
  -> outputs/dataset/lead_{L}mo/{split}.jsonl       (read by every method; record order = main, then extras)
The record order matters only for seeded shuffles, so it is fixed here.  Counts are checked against the study cohort."""
from __future__ import annotations
import argparse, json, shutil
import pandas as pd
from common import load_config, out_dir, get_logger

log = get_logger("s6b")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lead", type=int, default=3); a = ap.parse_args()
    cfg = load_config(); L = a.lead
    src = [out_dir(cfg, "dataset_main", f"lead_{L}mo"), out_dir(cfg, "dataset_extras", f"lead_{L}mo")]; dst = out_dir(cfg, "dataset", f"lead_{L}mo")
    coh = pd.read_parquet(out_dir(cfg, "cohort") / f"lead{L}_aug_cohort.parquet")
    for s in ["train", "dev", "test"]:
        with open(dst / f"{s}.jsonl", "wb") as fo:
            for d in src:
                with open(d / f"{s}.jsonl", "rb") as fi: shutil.copyfileobj(fi, fo, 1 << 24)
        n = pos = 0; pids = set()
        with open(dst / f"{s}.jsonl") as f:
            for line in f:
                r = json.loads(line); n += 1; pos += int(r["label"]); pids.add(r["person_id"])
        c = coh[coh.split == s]
        assert n == len(c) and pids == set(c.person_id) and pos == int(c.label.sum()), (s, n, len(c))
        log.info(f"lead {L} {s}: {n} records, {pos} positives -> {dst / (s + '.jsonl')}")


if __name__ == "__main__":
    main()
