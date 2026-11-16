"""Stage 5b: add the prevalent ("explicit") positives back to the lead-L study cohort.

Stage 5 keeps only patients WITHOUT documented depression evidence before the cutoff (incident positives + negatives) in
outputs/cohort/final_cohort.parquet and parks everyone else in excluded_prior_evidence_or_no_notes.parquet.  The study dataset
also contains the prevalent positives of that lead: label = 1, case_type = prevalent (target/broad depression code,
antidepressant, note-asserted diagnosis or PHQ-9 >= 10 before the cutoff) and at least one informative note.  They satisfy every
other eligibility rule (age, recency, history), because stage 3 applied those before stage 5.

Split: a patient who already has a split in the main cohort (any lead) keeps it; everybody else is assigned by the patient hash
with the same 70/15/15 fractions (salt 'aug_orig_split').  Negatives are untouched.

Outputs
  outputs/cohort_extras_lead{L}/final_cohort.parquet   prevalent positives only (input of s6_assemble --cohort-subdir)
  outputs/cohort/lead{L}_aug_cohort.parquet            main lead-L cohort + prevalent positives = the study cohort
  outputs/cohort/s5b_lead{L}_summary.json              split x class counts
"""
from __future__ import annotations
import argparse, json
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from common import load_config, out_dir, stable_hash, get_logger, write_summary

log = get_logger("s5b")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lead", type=int, default=3); a = ap.parse_args()
    cfg = load_config(); fr = cfg["cohort"]["split_fractions"]; cdir = out_dir(cfg, "cohort")
    fc = pd.read_parquet(cdir / "final_cohort.parquet"); ex = pd.read_parquet(cdir / "excluded_prior_evidence_or_no_notes.parquet")
    extra = ex[(ex.lead_months == a.lead) & (ex.label == 1) & (ex.case_type == "prevalent") & (ex.has_notes == True)].copy()  # noqa: E712
    known = fc.drop_duplicates("person_id").set_index("person_id").split

    def split_of(pid):
        if pid in known.index: return known.loc[pid]
        u = stable_hash(cfg["seed"], "aug_orig_split", pid) / 2**64
        return "train" if u < fr["train"] else ("dev" if u < fr["train"] + fr["dev"] else "test")

    extra["split"] = extra.person_id.map(split_of)
    main3 = fc[fc.lead_months == a.lead]
    assert not set(extra.person_id) & set(main3.person_id), "a prevalent positive is already in the main cohort"
    edir = out_dir(cfg, f"cohort_extras_lead{a.lead}")
    pq.write_table(pa.Table.from_pandas(extra, preserve_index=False), edir / "final_cohort.parquet")
    cols = [c for c in main3.columns if c in extra.columns]
    study = pd.concat([main3[cols], extra[cols]], ignore_index=True); assert study.person_id.is_unique
    pq.write_table(pa.Table.from_pandas(study, preserve_index=False), cdir / f"lead{a.lead}_aug_cohort.parquet")
    tab = {}
    for s in ["train", "dev", "test"]:
        x = study[study.split == s]
        tab[s] = {"size": int(len(x)), "neg": int((x.label == 0).sum()), "pos_incident": int((x.case_type == "incident").sum()), "pos_prevalent": int((x.case_type == "prevalent").sum())}
    tab["total"] = {k: int(sum(tab[s][k] for s in ["train", "dev", "test"])) for k in ["size", "neg", "pos_incident", "pos_prevalent"]}
    summ = {"lead": a.lead, "n_extras": int(len(extra)), "extras_with_existing_split": int(extra.person_id.isin(known.index).sum()), "table": tab}
    write_summary(cdir / f"s5b_lead{a.lead}_summary.json", summ); log.info(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
