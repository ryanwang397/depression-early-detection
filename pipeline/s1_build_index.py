#!/usr/bin/env python3
"""Stage 1: stream the OMOP CSVs and the note-metadata CSV into typed parquet files.

Usage:
  python s1_build_index.py --table visits [--limit-rows N]
  python s1_build_index.py --table all
Tables: person observation_period death provider care_site visits dx rx px obs meas note_meta concept
"""
from __future__ import annotations
import argparse, re, time
from pathlib import Path
import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
from common import (refuse_archived, load_config, out_dir, get_logger, write_summary, stream_csv, ParquetSink,
                    to_int64, to_date32, norm_str)

log = get_logger("s1")


def col(batch: pa.RecordBatch, name: str):
    return batch.column(batch.schema.get_field_index(name))


# ----------------------------------------------------------------------------- per-table transforms
def t_person(b):
    return pa.table({
        "person_id": to_int64(col(b, "person_id")),
        "birth_date": to_date32(col(b, "birth_datetime")),
        "year_of_birth": to_int64(col(b, "year_of_birth")),
        "sex": norm_str(col(b, "gender_source_value")),
        "race": norm_str(col(b, "race_source_value")),
        "ethnicity": norm_str(col(b, "ethnicity_source_value")),
        "location_id": to_int64(col(b, "location_id")),
        "pcp_provider_id": to_int64(col(b, "provider_id")),
    })


def t_obs_period(b):
    return pa.table({
        "person_id": to_int64(col(b, "person_id")),
        "obs_start": to_date32(col(b, "observation_period_start_date")),
        "obs_end": to_date32(col(b, "observation_period_end_date")),
    })


def t_death(b):
    return pa.table({"person_id": to_int64(col(b, "person_id")), "death_date": to_date32(col(b, "death_date"))})


def t_provider(b):
    return pa.table({
        "provider_id": to_int64(col(b, "provider_id")),
        "specialty": norm_str(col(b, "specialty_source_value")),
        "specialty_concept_id": to_int64(col(b, "specialty_concept_id")),
    })


def t_care_site(b):
    return pa.table({"care_site_id": to_int64(col(b, "care_site_id")), "location_id": to_int64(col(b, "location_id"))})


def t_visits(b):
    return pa.table({
        "visit_occurrence_id": to_int64(col(b, "visit_occurrence_id")),
        "person_id": to_int64(col(b, "person_id")),
        "visit_start_date": to_date32(col(b, "visit_start_date")),
        "visit_end_date": to_date32(col(b, "visit_end_date")),
        "visit_source_value": pc.utf8_lower(norm_str(col(b, "visit_source_value"))),
        "visit_concept_id": to_int64(col(b, "visit_concept_id")),
        "provider_id": to_int64(col(b, "provider_id")),
        "care_site_id": to_int64(col(b, "care_site_id")),
        "discharge_to": norm_str(col(b, "discharge_to_source_value")),
    })


def t_dx(b):
    return pa.table({
        "condition_occurrence_id": to_int64(col(b, "condition_occurrence_id")),
        "person_id": to_int64(col(b, "person_id")),
        "visit_occurrence_id": to_int64(col(b, "visit_occurrence_id")),
        "condition_start_date": to_date32(col(b, "condition_start_date")),
        "condition_end_date": to_date32(col(b, "condition_end_date")),
        "icd": norm_str(col(b, "condition_source_value")),
        "condition_concept_id": to_int64(col(b, "condition_concept_id")),
        "condition_type_concept_id": to_int64(col(b, "condition_type_concept_id")),
        "status": norm_str(col(b, "condition_status_source_value")),
    })


def t_rx(b):
    sig = norm_str(col(b, "sig"))
    return pa.table({
        "drug_exposure_id": to_int64(col(b, "drug_exposure_id")),
        "person_id": to_int64(col(b, "person_id")),
        "visit_occurrence_id": to_int64(col(b, "visit_occurrence_id")),
        "drug_start_date": to_date32(col(b, "drug_exposure_start_date")),
        "drug_end_date": to_date32(col(b, "drug_exposure_end_date")),
        "drug_concept_id": to_int64(col(b, "drug_concept_id")),
        "drug_source_value": norm_str(col(b, "drug_source_value")),
        "drug_type_concept_id": to_int64(col(b, "drug_type_concept_id")),
        "quantity": pc.cast(col(b, "quantity"), pa.float64(), safe=False),
        "refills": pc.cast(col(b, "refills"), pa.float64(), safe=False),
        "sig": pc.utf8_slice_codeunits(sig, 0, 200),
        "route": norm_str(col(b, "route_source_value")),
        "dose_unit": norm_str(col(b, "dose_unit_source_value")),
    })


def t_px(b):
    return pa.table({
        "procedure_occurrence_id": to_int64(col(b, "procedure_occurrence_id")),
        "person_id": to_int64(col(b, "person_id")),
        "visit_occurrence_id": to_int64(col(b, "visit_occurrence_id")),
        "procedure_date": to_date32(col(b, "procedure_date")),
        "code": norm_str(col(b, "procedure_source_value")),
        "procedure_concept_id": to_int64(col(b, "procedure_concept_id")),
        "procedure_type_concept_id": to_int64(col(b, "procedure_type_concept_id")),
    })


def make_t_obs(cfg):
    keep_vals = set(cfg["observations"]["keep_source_values"])
    icd_re = cfg["observations"]["keep_icd_regex"]

    def t(b):
        sv = norm_str(col(b, "observation_source_value"))
        mask = pc.or_(pc.is_in(sv, value_set=pa.array(sorted(keep_vals))),
                      pc.fill_null(pc.match_substring_regex(sv, icd_re), False))
        tbl = pa.table({
            "observation_id": to_int64(col(b, "observation_id")),
            "person_id": to_int64(col(b, "person_id")),
            "visit_occurrence_id": to_int64(col(b, "visit_occurrence_id")),
            "observation_date": to_date32(col(b, "observation_date")),
            "source_value": sv,
            "value_as_string": norm_str(col(b, "value_as_string")),
            "value_as_number": pc.cast(col(b, "value_as_number"), pa.float64(), safe=False),
            "observation_concept_id": to_int64(col(b, "observation_concept_id")),
            "observation_type_concept_id": to_int64(col(b, "observation_type_concept_id")),
        })
        return tbl.filter(mask)
    return t


def make_t_meas(cfg):
    vit_re = cfg["measurements"]["keep_source_regex"]
    loinc = cfg["measurements"]["keep_loinc"]
    codes = pa.array(sorted(loinc.keys()))

    def t(b):
        sv = norm_str(col(b, "measurement_source_value"))
        mask = pc.or_(pc.fill_null(pc.match_substring_regex(sv, vit_re, ignore_case=True), False),
                      pc.is_in(sv, value_set=codes))
        tbl = pa.table({
            "measurement_id": to_int64(col(b, "measurement_id")),
            "person_id": to_int64(col(b, "person_id")),
            "visit_occurrence_id": to_int64(col(b, "visit_occurrence_id")),
            "measurement_date": to_date32(col(b, "measurement_date")),
            "source_value": sv,
            "measurement_concept_id": to_int64(col(b, "measurement_concept_id")),
            "value_as_number": pc.cast(col(b, "value_as_number"), pa.float64(), safe=False),
            "unit": norm_str(col(b, "unit_source_value")),
            "value_source_value": pc.utf8_slice_codeunits(norm_str(col(b, "value_source_value")), 0, 40),
        })
        return tbl.filter(mask)
    return t


def t_note_meta(b):
    return pa.table({
        "linkage_note_id": to_int64(col(b, "DEID_LINKAGE_NOTE_ID")),
        "note_id": to_int64(col(b, "DEID_NOTE_ID")),
        "contact_number": to_int64(col(b, "ContactNumber")),
        "encounter_date": to_date32(col(b, "EncounterDateShifted")),
        "contact_date": to_date32(col(b, "ContactDateShifted")),
        "service_date": to_date32(col(b, "ServiceDatetimeShifted")),
        "created_date": to_date32(col(b, "CreatedDatetimeShifted")),
        "person_id": to_int64(col(b, "deid_patient_key")),
        "note_type": norm_str(col(b, "NoteType")),
        "inpatient_note_type": norm_str(col(b, "InpatientNoteType")),
        "encounter_key": to_int64(col(b, "deid_encounter_key")),
        "primary_encounter_key": to_int64(col(b, "deid_primary_encounter_key")),
        "author_key": to_int64(col(b, "DEID_AuthoringProviderKey")),
        "author_type": norm_str(col(b, "AuthoringProviderType")),
        "author_specialty": norm_str(col(b, "AuthoringProviderSpecialty")),
    })


TABLES = {
    "person": ("person.csv", ["person_id", "birth_datetime", "year_of_birth", "gender_source_value", "race_source_value", "ethnicity_source_value", "location_id", "provider_id"], t_person),
    "observation_period": ("observation_period.csv", ["person_id", "observation_period_start_date", "observation_period_end_date"], t_obs_period),
    "death": ("death.csv", ["person_id", "death_date"], t_death),
    "provider": ("provider.csv", ["provider_id", "specialty_source_value", "specialty_concept_id"], t_provider),
    "care_site": ("care_site.csv", ["care_site_id", "location_id"], t_care_site),
    "visits": ("visit_occurrence.csv", ["visit_occurrence_id", "person_id", "visit_start_date", "visit_end_date", "visit_source_value", "visit_concept_id", "provider_id", "care_site_id", "discharge_to_source_value"], t_visits),
    "dx": ("condition_occurrence.csv", ["condition_occurrence_id", "person_id", "visit_occurrence_id", "condition_start_date", "condition_end_date", "condition_source_value", "condition_concept_id", "condition_type_concept_id", "condition_status_source_value"], t_dx),
    "rx": ("drug_exposure.csv", ["drug_exposure_id", "person_id", "visit_occurrence_id", "drug_exposure_start_date", "drug_exposure_end_date", "drug_concept_id", "drug_source_value", "drug_type_concept_id", "quantity", "refills", "sig", "route_source_value", "dose_unit_source_value"], t_rx),
    "px": ("procedure_occurrence.csv", ["procedure_occurrence_id", "person_id", "visit_occurrence_id", "procedure_date", "procedure_source_value", "procedure_concept_id", "procedure_type_concept_id"], t_px),
    "obs": ("observation.csv", ["observation_id", "person_id", "visit_occurrence_id", "observation_date", "observation_source_value", "value_as_string", "value_as_number", "observation_concept_id", "observation_type_concept_id"], None),
    "meas": ("measurement.csv", ["measurement_id", "person_id", "visit_occurrence_id", "measurement_date", "measurement_source_value", "measurement_concept_id", "value_as_number", "unit_source_value", "value_source_value"], None),
    "note_meta": (None, ["DEID_LINKAGE_NOTE_ID", "DEID_NOTE_ID", "ContactNumber", "ContactDateShifted", "CreatedDatetimeShifted", "ServiceDatetimeShifted", "deid_patient_key", "NoteType", "InpatientNoteType", "EncounterDateShifted", "deid_encounter_key", "deid_primary_encounter_key", "DEID_AuthoringProviderKey", "AuthoringProviderType", "AuthoringProviderSpecialty"], t_note_meta),
}


def build_table(cfg, name: str, limit_rows: int | None, odir: Path, dedupe_only: bool = False) -> dict:
    fname, cols, tf = TABLES[name]
    if name == "note_meta" and dedupe_only:
        info = {"table": "note_meta", "dedupe_only": True, "output": str(odir / "note_meta.parquet")}
        info.update(dedupe_note_meta(odir))
        return info
    if name == "obs":
        tf = make_t_obs(cfg)
    if name == "meas":
        tf = make_t_meas(cfg)
    src = Path(cfg["paths"]["note_metadata"]) if name == "note_meta" else Path(cfg["paths"]["omop_dir"]) / fname
    out = odir / (("note_meta_raw" if name == "note_meta" else name) + ".parquet")
    t0 = time.time(); n_in = 0
    sink = ParquetSink(out)
    for b in stream_csv(src, cols, limit_rows=limit_rows):
        n_in += b.num_rows
        sink.write(tf(b))
        if n_in % 5_000_000 < b.num_rows:
            log.info(f"{name}: {n_in:,} rows in, {sink.rows:,} out, {time.time()-t0:.0f}s")
    sink.close()
    info = {"table": name, "source": str(src), "rows_in": n_in, "rows_out": sink.rows, "seconds": round(time.time() - t0, 1), "output": str(out)}
    log.info(info)
    if name == "note_meta":
        info.update(dedupe_note_meta(odir))
    return info


def dedupe_note_meta(odir: Path) -> dict:
    """Keep one row per DEID_NOTE_ID: the highest ContactNumber (final version). Ties -> lowest linkage id."""
    raw = pq.read_table(odir / "note_meta_raw.parquet", columns=["linkage_note_id", "note_id", "contact_number"])
    df = raw.to_pandas()
    n_raw = len(df)
    df = df.sort_values(["note_id", "contact_number", "linkage_note_id"], ascending=[True, False, True])
    keep = df.drop_duplicates("note_id", keep="first")["linkage_note_id"]
    keep_set = pa.array(keep.values, pa.int64())
    del df, raw
    pf = pq.ParquetFile(odir / "note_meta_raw.parquet")
    sink = ParquetSink(odir / "note_meta.parquet")
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg)
        sink.write(t.filter(pc.is_in(t.column("linkage_note_id"), value_set=keep_set)))
    sink.close()
    info = {"note_meta_rows_raw": n_raw, "note_meta_rows_dedup": sink.rows}
    log.info(info)
    return info


def build_concepts(cfg, odir: Path) -> dict:
    """concept_lite: concepts referenced by dx/rx/px/meas/obs/visits; drug_classes: antidepressant/GLP-1 descendants."""
    cdir = Path(cfg["paths"]["concept_dir"])
    ids = set()
    for tbl, c in [("dx", "condition_concept_id"), ("rx", "drug_concept_id"), ("px", "procedure_concept_id"),
                   ("meas", "measurement_concept_id"), ("obs", "observation_concept_id"), ("visits", "visit_concept_id"),
                   ("provider", "specialty_concept_id")]:
        p = odir / f"{tbl}.parquet"
        if p.exists():
            ids.update(pc.unique(pq.read_table(p, columns=[c]).column(c)).drop_null().to_pylist())
    log.info(f"distinct concept ids referenced: {len(ids):,}")
    concept = pq.read_table  # placeholder to keep linters quiet
    # stream Concept.csv
    cols = ["concept_id", "concept_name", "domain_id", "vocabulary_id", "concept_class_id", "standard_concept", "concept_code"]
    idset = pa.array(sorted(ids), pa.int64())
    ph = cfg["phenotype"]
    want = {n.lower(): "washout_antidepressant" for n in ph["washout_antidepressants"]}
    want.update({n.lower(): "weak_antidepressant" for n in ph["weak_antidepressants"]})
    want.update({n.lower(): "glp1" for n in ph["glp1_ingredients"]})
    sink = ParquetSink(odir / "concept_lite.parquet")
    ingredients = []
    for b in stream_csv(cdir / "Concept.csv", cols):
        cid = to_int64(col(b, "concept_id"))
        t = pa.table({"concept_id": cid, "concept_name": col(b, "concept_name"), "domain_id": col(b, "domain_id"),
                      "vocabulary_id": col(b, "vocabulary_id"), "concept_class_id": col(b, "concept_class_id"),
                      "standard_concept": col(b, "standard_concept"), "concept_code": col(b, "concept_code")})
        sink.write(t.filter(pc.is_in(cid, value_set=idset)))
        ing = t.filter(pc.and_(pc.equal(t.column("vocabulary_id"), "RxNorm"), pc.equal(t.column("concept_class_id"), "Ingredient")))
        for r in ing.to_pylist():
            k = (r["concept_name"] or "").lower()
            if k in want:
                ingredients.append((r["concept_id"], k, want[k]))
    sink.close()
    log.info(f"concept_lite rows {sink.rows:,}; ingredient concepts found {len(ingredients)}")
    # descendants via ConceptAncestor
    anc_ids = pa.array([i for i, _, _ in ingredients], pa.int64())
    anc_map = {i: (n, c) for i, n, c in ingredients}
    rows = []
    for b in stream_csv(cdir / "ConceptAncestor.csv", ["ancestor_concept_id", "descendant_concept_id"]):
        a = to_int64(col(b, "ancestor_concept_id")); d = to_int64(col(b, "descendant_concept_id"))
        m = pc.is_in(a, value_set=anc_ids)
        for aa, dd in zip(a.filter(m).to_pylist(), d.filter(m).to_pylist()):
            rows.append({"concept_id": dd, "ingredient_concept_id": aa, "ingredient": anc_map[aa][0], "drug_class": anc_map[aa][1]})
    dc = pa.Table.from_pylist(rows)
    pq.write_table(dc, odir / "drug_classes.parquet")
    info = {"concept_lite_rows": sink.rows, "ingredients": ingredients, "drug_class_rows": dc.num_rows}
    log.info(info)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, help="table name or 'all'")
    ap.add_argument("--limit-rows", type=int, default=None)
    ap.add_argument("--out-subdir", default="index")
    ap.add_argument("--dedupe-only", action="store_true", help="note_meta: skip CSV streaming, only dedupe note_meta_raw.parquet")
    args = ap.parse_args()
    cfg = load_config(); refuse_archived(cfg, args.out_subdir)
    odir = out_dir(cfg, args.out_subdir)
    names = list(TABLES) + ["concept"] if args.table == "all" else [args.table]
    infos = []
    for n in names:
        infos.append(build_concepts(cfg, odir) if n == "concept" else build_table(cfg, n, args.limit_rows, odir, args.dedupe_only))
        write_summary(odir / f"s1_{n}_summary.json", infos[-1])


if __name__ == "__main__":
    main()
