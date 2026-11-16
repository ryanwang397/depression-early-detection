#!/usr/bin/env python3
"""Stage 6: assemble one JSONL record per patient sample for a given lead and split.

Record = metadata + structured events on/before cutoff (grouped by encounter date) + processed notes
(L2 condensed text, L3 features) + an XML-like `encounter_text` rendering + aggregated L3/structured
patient-level features. Labels, person ids and visit ids are never placed inside encounter_text.

Every event / note date is asserted <= cutoff_date (hard leakage guard).
Usage: python s6_assemble.py --lead 12 --split train [--limit-persons N]
"""
from __future__ import annotations
import argparse, json, time, html, re
from collections import defaultdict
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow.dataset as ds
from common import load_config, out_dir, get_logger, write_summary, icd_prefix_match

log = get_logger("s6")

DX_GROUPS = {  # ICD prefixes -> feature group (structured comorbidity features)
    "anxiety": ["F41", "F40", "F93.0", "300.0"], "adhd": ["F90", "314"], "asd": ["F84", "299"], "ptsd_adjust": ["F43"],
    "sleep": ["G47", "F51", "780.5"], "obesity": ["E66", "278.0"], "diabetes": ["E11", "E10", "250"], "headache": ["G43", "R51", "G44"],
    "abdominal_pain": ["R10"], "fatigue": ["R53"], "conduct_odd": ["F91"], "eating_disorder": ["F50"], "self_harm_code": ["X7", "X8", "T14.91", "R45.85", "R45.88"],
    "emotional_symptom": ["R45"], "social_z": ["Z63", "Z65", "Z62", "Z60", "Z55"], "substance": ["F1", "Z71.4", "Z71.5", "Z72.0"],
    "intellectual_dev": ["F70", "F71", "F72", "F79", "F80", "F81", "F82", "F88", "F89"], "epilepsy": ["G40"], "asthma": ["J45"],
}


def read_filtered(path, columns, persons: pa.Array, extra_filter=None) -> pd.DataFrame:
    pf = pq.ParquetFile(path); parts = []
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=columns)
        m = pc.is_in(t.column("person_id"), value_set=persons)
        sub = t.filter(m)
        if extra_filter is not None and sub.num_rows:
            sub = sub.filter(extra_filter(sub))
        if sub.num_rows:
            parts.append(sub.to_pandas())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def xml_attr(v) -> str:
    return html.escape(str(v), quote=True)


_PH = re.compile(r"&lt;([A-Z_]+)&gt;")


def esc(s: str) -> str:
    """escape for the XML-like rendering but keep de-id placeholders readable (<DATE>, <NAME>, ...)."""
    return _PH.sub(r"<\1>", html.escape(s, quote=False))


MEAS_NAMES = {"SBP - Unknown Method": "SBP", "DBP - Unknown Method": "DBP", "BP - Non-invasive SBP": "SBP", "BP - Non-invasive DBP": "DBP",
              "WEIGHT": "weight", "HEIGHT": "height", "BMI": "BMI", "WEIGHT Admit": "weight", "HEIGHT Admit": "height", "BMI Admit": "BMI"}


def render_encounter_text(encs: list[dict]) -> str:
    lines = ["<patient>"]
    for e in encs:
        lines.append(f'<encounter date="{e["date"]}" days_before_cutoff="{e["days_before_cutoff"]}" visit="{xml_attr(e["visit_type"])}">')
        ehr = []
        if e["conditions"]:
            ehr.append("conditions: " + "; ".join(f'{c["name"]} [{c["icd"]}]' for c in e["conditions"]) + ".")
        if e["medications"]:
            ehr.append("medications: " + "; ".join(m["name"] for m in e["medications"]) + ".")
        if e["procedures"]:
            ehr.append("procedures: " + "; ".join(f'{p["name"]} [{p["code"]}]' for p in e["procedures"]) + ".")
        if e["measurements"]:
            ehr.append("measurements: " + "; ".join(f'{m["name"]} {m["value"]}{(" " + m["unit"]) if m["unit"] else ""}' for m in e["measurements"]) + ".")
        if e["observations"]:
            ehr.append("observations: " + "; ".join(f'{o["name"]}: {o["value"]}' for o in e["observations"]) + ".")
        if ehr:
            lines.append("  <ehr>\n    " + "\n    ".join(esc(x) for x in ehr) + "\n  </ehr>")
        for n in e["notes"]:
            lines.append(f'  <note type="{xml_attr(n["type"])}" specialty="{xml_attr(n["specialty"] or "")}" author="{xml_attr(n["author_type"] or "")}" relevance="{n["relevance"]}">\n    {esc(n["text"])}\n  </note>')
        lines.append("</encounter>")
    lines.append("</patient>")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=int, required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--limit-persons", type=int, default=None)
    ap.add_argument("--index-subdir", default="index"); ap.add_argument("--cohort-subdir", default="cohort"); ap.add_argument("--notes-subdir", default="notes"); ap.add_argument("--out-subdir", default="dataset")
    args = ap.parse_args()
    cfg = load_config(); idir = out_dir(cfg, args.index_subdir); cdir = out_dir(cfg, args.cohort_subdir); ndir = out_dir(cfg, args.notes_subdir)
    odir = out_dir(cfg, args.out_subdir, f"lead_{args.lead}mo"); t0 = time.time()
    coh = pq.read_table(cdir / "final_cohort.parquet").to_pandas()
    coh = coh[(coh.lead_months == args.lead) & (coh.split == args.split)].sort_values("person_id")
    if args.limit_persons:
        coh = coh.head(args.limit_persons)
    coh["anchor_date"] = pd.to_datetime(coh.anchor_date); coh["cutoff_date"] = pd.to_datetime(coh.cutoff_date)
    persons = pa.array(np.sort(coh.person_id.unique()), pa.int64())
    cutoff = coh.set_index("person_id").cutoff_date
    log.info(f"lead {args.lead} split {args.split}: {len(coh):,} samples")
    # ------------------------------------------------------------------ reference tables
    concept = pq.read_table(idir / "concept_lite.parquet", columns=["concept_id", "concept_name"]).to_pandas().set_index("concept_id").concept_name
    dclass = pq.read_table(idir / "drug_classes.parquet").to_pandas().drop_duplicates("concept_id").set_index("concept_id")
    prov = pq.read_table(idir / "provider.parquet", columns=["provider_id", "specialty"]).to_pandas().set_index("provider_id").specialty
    loinc = cfg["measurements"]["keep_loinc"]
    ph = cfg["phenotype"]
    # ------------------------------------------------------------------ events (all <= cutoff)
    def by_cutoff(df, datecol):
        df = df.dropna(subset=[datecol]).copy(); df[datecol] = pd.to_datetime(df[datecol])
        return df[df[datecol] <= df.person_id.map(cutoff)]
    vis = by_cutoff(read_filtered(idir / "visits.parquet", ["person_id", "visit_occurrence_id", "visit_start_date", "visit_source_value", "provider_id"], persons), "visit_start_date")
    vis["specialty"] = vis.provider_id.map(prov)
    dx = by_cutoff(read_filtered(idir / "dx.parquet", ["person_id", "condition_start_date", "icd", "condition_concept_id", "condition_type_concept_id"], persons), "condition_start_date")
    assert not icd_prefix_match(pa.array(dx.icd.fillna("").astype(str)), ph["target_icd_prefixes"]).to_pandas().any(), "target code found before cutoff"
    dx["name"] = dx.condition_concept_id.map(concept).fillna(dx.icd)
    rx = by_cutoff(read_filtered(idir / "rx.parquet", ["person_id", "drug_start_date", "drug_concept_id", "drug_source_value", "drug_type_concept_id", "route"], persons), "drug_start_date")
    rx["name"] = rx.drug_concept_id.map(concept).fillna("RxNorm " + rx.drug_source_value.astype(str))
    rx["drug_class"] = rx.drug_concept_id.map(dclass.drug_class)
    px = by_cutoff(read_filtered(idir / "px.parquet", ["person_id", "procedure_date", "code", "procedure_concept_id"], persons), "procedure_date")
    px["name"] = px.procedure_concept_id.map(concept).fillna(px.code)
    me = by_cutoff(read_filtered(idir / "meas.parquet", ["person_id", "measurement_date", "source_value", "value_as_number", "unit"], persons), "measurement_date")
    me["name"] = me.source_value.map(lambda s: loinc.get(s, MEAS_NAMES.get(s, s)))
    me = me.dropna(subset=["value_as_number"])
    ob = by_cutoff(read_filtered(idir / "obs.parquet", ["person_id", "observation_date", "source_value", "value_as_string", "observation_concept_id"], persons), "observation_date")
    ob = ob[ob.source_value != "Zipcode"]
    is_social = ob.source_value.isin(cfg["observations"]["keep_source_values"])
    ob["name"] = np.where(is_social, ob.source_value, ob.observation_concept_id.map(concept).fillna(ob.source_value))
    # social/administrative observations carry their value in value_as_string; ICD-coded observations show the code
    ob["value"] = np.where(is_social, ob.value_as_string.fillna("unknown"), "[" + ob.source_value.astype(str) + "]")
    # notes
    nparts = []
    for p in sorted((ndir).glob("part_*.parquet")):
        t = pq.read_table(p, columns=["person_id", "note_id", "note_date", "note_type", "author_type", "author_specialty", "condensed_text", "chars_condensed", "mh_relevance", "features_json"])
        t = t.filter(pc.is_in(t.column("person_id"), value_set=persons))
        if t.num_rows:
            nparts.append(t.to_pandas())
    notes = pd.concat(nparts, ignore_index=True) if nparts else pd.DataFrame(columns=["person_id", "note_id", "note_date", "note_type", "author_type", "author_specialty", "condensed_text", "chars_condensed", "mh_relevance", "features_json"])
    notes = by_cutoff(notes, "note_date").drop_duplicates(["person_id", "note_id"])
    trivial = (notes.note_type == "Telephone Encounter") & (notes.chars_condensed < 150) & (notes.mh_relevance == 0)
    n_trivial = int(trivial.sum()); notes = notes[~trivial]
    log.info(f"events loaded: visits {len(vis):,} dx {len(dx):,} rx {len(rx):,} px {len(px):,} meas {len(me):,} obs {len(ob):,} notes {len(notes):,} ({time.time()-t0:.0f}s)")
    groups = {k: {p: g for p, g in d.groupby("person_id")} for k, d in [("vis", vis), ("dx", dx), ("rx", rx), ("px", px), ("me", me), ("ob", ob), ("notes", notes)]}
    # ------------------------------------------------------------------ assemble
    out_path = odir / f"{args.split}.jsonl"; n = 0; stats = defaultdict(list)
    with open(out_path, "w") as fo:
        for r in coh.itertuples(index=False):
            pid = int(r.person_id); cut = r.cutoff_date
            g = {k: groups[k].get(pid) for k in groups}
            enc = defaultdict(lambda: {"conditions": [], "medications": [], "procedures": [], "measurements": [], "observations": [], "notes": [], "visit_types": set()})
            if g["vis"] is not None:
                for v in g["vis"].itertuples(index=False):
                    e = enc[v.visit_start_date.date()]; e["visit_types"].add(v.visit_source_value or "unknown")
            if g["dx"] is not None:
                seen = set()
                for d in g["dx"].sort_values("condition_start_date").itertuples(index=False):
                    key = (d.condition_start_date.date(), d.icd)
                    if key in seen: continue
                    seen.add(key); enc[d.condition_start_date.date()]["conditions"].append({"date": str(d.condition_start_date.date()), "icd": d.icd, "name": d.name, "type": int(d.condition_type_concept_id) if pd.notna(d.condition_type_concept_id) else None})
            if g["rx"] is not None:
                seen = set()
                for d in g["rx"].sort_values("drug_start_date").itertuples(index=False):
                    key = (d.drug_start_date.date(), d.name)
                    if key in seen: continue
                    seen.add(key); enc[d.drug_start_date.date()]["medications"].append({"date": str(d.drug_start_date.date()), "name": d.name, "class": d.drug_class if isinstance(d.drug_class, str) else None, "route": d.route, "type": int(d.drug_type_concept_id) if pd.notna(d.drug_type_concept_id) else None})
            if g["px"] is not None:
                seen = set()
                for d in g["px"].sort_values("procedure_date").itertuples(index=False):
                    key = (d.procedure_date.date(), d.code)
                    if key in seen: continue
                    seen.add(key); enc[d.procedure_date.date()]["procedures"].append({"date": str(d.procedure_date.date()), "code": d.code, "name": d.name})
            if g["me"] is not None:
                last = g["me"].sort_values("measurement_date").groupby(["measurement_date", "name"]).tail(1)
                for d in last.itertuples(index=False):
                    enc[d.measurement_date.date()]["measurements"].append({"date": str(d.measurement_date.date()), "name": d.name, "value": round(float(d.value_as_number), 2), "unit": d.unit if isinstance(d.unit, str) else ""})
            if g["ob"] is not None:
                last = g["ob"].sort_values("observation_date").groupby(["observation_date", "name"]).tail(1).sort_values("observation_date")
                state = {}
                for d in last.itertuples(index=False):
                    if d.name in ("Payer", "SMOKING STATUS", "Marital Status") and state.get(d.name) == d.value:
                        continue   # render social/administrative state only when it changes
                    state[d.name] = d.value
                    enc[d.observation_date.date()]["observations"].append({"date": str(d.observation_date.date()), "name": d.name, "value": d.value})
            if g["notes"] is not None:
                for d in g["notes"].sort_values("note_date").itertuples(index=False):
                    enc[d.note_date.date()]["notes"].append({"note_id": int(d.note_id), "date": str(d.note_date.date()), "type": d.note_type, "specialty": d.author_specialty, "author_type": d.author_type, "relevance": int(d.mh_relevance), "features": json.loads(d.features_json), "text": d.condensed_text})
            encs = []
            for day in sorted(enc):
                e = enc[day]; assert pd.Timestamp(day) <= cut, "event after cutoff"
                if not any([e["conditions"], e["medications"], e["procedures"], e["measurements"], e["observations"], e["notes"]]):
                    continue
                e["date"] = str(day); e["days_before_cutoff"] = int((cut - pd.Timestamp(day)).days); e["visit_type"] = "/".join(sorted(e["visit_types"])) or "none"
                del e["visit_types"]; encs.append(e)
            # aggregated features
            feats = {"n_encounters": len(encs), "n_notes": sum(len(e["notes"]) for e in encs), "n_visits_365d": int((g["vis"].visit_start_date >= cut - pd.Timedelta(days=365)).sum()) if g["vis"] is not None else 0,
                     "n_ed_visits_365d": int(((g["vis"].visit_start_date >= cut - pd.Timedelta(days=365)) & g["vis"].visit_source_value.fillna("").str.contains("ed")).sum()) if g["vis"] is not None else 0,
                     "any_psychiatry_visit": int(g["vis"].specialty.fillna("").str.contains("Psychiat|Psycholog", regex=True).any()) if g["vis"] is not None else 0}
            icds = g["dx"].icd.fillna("").astype(str) if g["dx"] is not None else pd.Series(dtype=str)
            for grp, pref in DX_GROUPS.items():
                feats[f"dx_{grp}"] = int(icds.str.startswith(tuple(pref)).any()) if len(icds) else 0
            feats["rx_weak_antidepressant"] = int((g["rx"].drug_class == "weak_antidepressant").any()) if g["rx"] is not None else 0
            feats["rx_glp1"] = int((g["rx"].drug_class == "glp1").any()) if g["rx"] is not None else 0
            feats["rx_psych_any"] = int(g["rx"].name.fillna("").str.contains(r"aripiprazole|risperidone|quetiapine|guanfacine|clonidine|methylphenidate|amphetamine|atomoxetine|hydroxyzine|buspirone|lamotrigine|olanzapine", case=False, regex=True).any()) if g["rx"] is not None else 0
            if g["me"] is not None and (g["me"].name == "BMI").any():
                b = g["me"][g["me"].name == "BMI"].sort_values("measurement_date"); feats["bmi_last"] = round(float(b.value_as_number.iloc[-1]), 1); feats["bmi_last_days_before_cutoff"] = int((cut - b.measurement_date.iloc[-1]).days)
            nf = [json.loads(x) for x in g["notes"].features_json] if g["notes"] is not None else []
            for k in ["mh_relevance", "dep_mention__present", "dep_assert__present", "suicidality__present", "self_harm__present", "sx_mood__present", "sx_neuroveg__present", "stressor__present", "school__present", "treatment__present", "anxiety__present", "psych_med__present", "fam_mh__family", "n_template_phrases"]:
                feats[f"notes_{k}"] = int(sum(x.get(k, 0) for x in nf))
            if g["notes"] is not None:
                sc = [(pd.Timestamp(d), f) for d, f in zip(g["notes"].note_date, nf)]
                for inst in ["phq9", "phq2", "gad7"]:
                    vals = [(d, f[f"score_{inst}"]) for d, f in sc if f"score_{inst}" in f]
                    if vals:
                        d, v = max(vals); feats[f"{inst}_last"] = int(v); feats[f"{inst}_last_days_before_cutoff"] = int((cut - d).days); feats[f"{inst}_max"] = int(max(v for _, v in vals))
            rec = {"person_id": pid, "label": int(r.label), "case_type": r.case_type, "lead_months": args.lead, "split": args.split,
                   "anchor_date": str(r.anchor_date.date()), "cutoff_date": str(cut.date()), "cutoff_visit_id": int(r.cutoff_visit_id),
                   "age_at_cutoff": round(float((cut - pd.Timestamp(r.birth_date)).days / 365.25), 2), "sex": r.sex, "race": r.race, "ethnicity": r.ethnicity,
                   "history_days": int(r.history_days), "recency_gap_days": int(r.recency_gap_days),
                   "encounters": encs, "features": feats,
                   "encounter_text": render_encounter_text(encs)}
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
            stats["n_notes"].append(feats["n_notes"]); stats["n_enc"].append(len(encs)); stats["chars"].append(len(rec["encounter_text"])); stats["label"].append(int(r.label))
    s = pd.DataFrame(stats)
    info = {"lead": args.lead, "split": args.split, "samples": n, "label_counts": s.label.value_counts().to_dict(),
            "median_notes": s.groupby("label").n_notes.median().to_dict(), "median_encounters": s.groupby("label").n_enc.median().to_dict(),
            "median_chars_encounter_text": s.groupby("label").chars.median().to_dict(), "trivial_telephone_notes_dropped": n_trivial, "seconds": round(time.time() - t0, 1), "output": str(out_path)}
    write_summary(odir / f"s6_{args.split}_summary.json", info); log.info(info)


if __name__ == "__main__":
    main()
