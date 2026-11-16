# Evidence-grounded, case-aware multi-agent framework for early detection of depression

Code for the paper **"An evidence-grounded, case-aware multi-agent framework for early detection of depression from
longitudinal EHR data"** (under review).

The task is to predict a first coded depression diagnosis in youth 3–12 months ahead from everything recorded before a
cutoff. The framework has three agents:

1. **Evidence collector agent** (`l4/`): MedGemma-27B reads each narrative note under schema-constrained JSON decoding
   and returns typed evidence items with a verbatim quote (16 categories, 86 subtypes; `l4/schema.py`). Quotes that do
   not occur verbatim in the note are dropped. The validated items form an evidence ledger, which is rendered as a
   one-page evidence timeline per patient.
2. **Case-aware expert agent** (`eagrl/e0_expert.py`): a gradient-boosting model on structured, utilisation and ledger
   features, trained on implicit positives and negatives only. A rule gate consults it only for records without
   documented depression before the cutoff.
3. **Diagnosis agent** (`experiments/run_llm_lora_cls.py`, `eagrl/`): Llama-3.1-8B-Instruct with LoRA. It is first
   trained as a Yes/No classifier on the timeline (stage 0). It is then fine-tuned on hindsight-distilled reasoning
   traces against the soft target 0.8 · p_stage0 + 0.2 · y. The risk is P(Yes) at the final-answer token, and the
   cited evidence can be checked against the timeline.

**No data are included.** The code expects a de-identified OMOP CDM extract plus clinical notes, which are available only
under a data use agreement. This repository contains no patient-level data, predictions or model weights.

**Terminology.** The code calls explicit positives `prevalent` and implicit positives `incident`. Explicit positives have
documented depression evidence before the cutoff: an antidepressant, a broad depression code, a note stating the
diagnosis, or PHQ-9 ≥ 10. Implicit positives have none.

## Layout

| Path | Content |
|---|---|
| `config.yaml` | source paths, seed, code lists (target / broad ICD, antidepressants), cohort rules, note whitelist |
| `pipeline/` | data organisation: source extract → `outputs/dataset/lead_3mo/{train,dev,test}.jsonl` |
| `l4/` | evidence collector agent: note queue, MedGemma-27B extraction (vLLM), validation, ledger, ledger features, timeline rendering |
| `eagrl/` | case-aware expert and rule gate (`e0`), care context (`f1`), hindsight distillation (`e1`), soft targets (`e5`), reasoning fine-tuning (`e2`), inference and grounding checks (`e4`) |
| `experiments/` | baselines (Clinical ModernBERT, Llama zero-shot, Llama LoRA-SFT on the XML record) and the stage-0 classifier (`run_llm_lora_cls.py`; `--input xml` = baseline, default = evidence timeline) |
| `evaluation/make_table.py` | AUROC / AP / F1 / precision / recall per method, optional bootstrap CIs |
| `scripts/` | numbered run scripts (Slurm submission with dependencies); `env.sh` holds every environment setting |
| `slurm/job.sbatch` | generic job wrapper used by all scripts |

## Setup

* Python 3.10 and the packages in `requirements.txt` (torch 2.7.0 + CUDA 12.8, transformers 4.52.4, peft 0.15.2, vllm 0.9.1).
* Model snapshots in a HuggingFace hub cache (`DED_HF_HUB`, default `~/.cache/huggingface/hub`):
  `meta-llama/Llama-3.1-8B-Instruct`, `google/medgemma-27b-text-it`, `Simonlee711/Clinical_ModernBERT`.
* One 80 GB-class GPU per GPU job. The vLLM stages need recent kernels (B200/A100 class).
* Edit the four source paths and `notes_glob` in `config.yaml`. Then set `PY`, `DED_HF_HUB` and the Slurm account,
  QOS and partitions in `scripts/env.sh`.
* Disk: about 60 GB for `outputs/`.

## Running

Run order: `10` → (`20`, `21`, `22` in parallel) and `30` → (`31`, `32` in parallel) → `33` → `40`.
Each script submits Slurm jobs and prints the last job id. Scripts `31`–`33` pick up the ids of the previous step from
`logs/last_*_job`.

```
bash scripts/10_data_pipeline.sh          # cohort, splits, dataset JSONL
bash scripts/20_baseline_modernbert.sh    # baselines
bash scripts/21_baseline_llama_zeroshot.sh
bash scripts/22_baseline_llama_lora_xml.sh
bash scripts/30_ours_agent1_extraction.sh # evidence collector agent (MedGemma-27B, ~100 GPU-h for the 3-month cohort)
bash scripts/31_ours_stage0_classifier.sh # diagnosis agent, stage 0
bash scripts/32_ours_expert_gate.sh       # case-aware expert + rule gate, care context
bash scripts/33_ours_reasoning.sh         # hindsight distillation, reasoning fine-tuning, inference
bash scripts/40_make_table.sh             # results table
```

By default (`FROM_SOURCE=1`) every intermediate is built from the raw extract. With `FROM_SOURCE=0`, scripts `10` and `30`
instead link precomputed, read-only intermediates from an earlier run (`DED_ARCHIVE`, `DED_ARCHIVE_AUG`; see
`scripts/env.sh`). This skips the slow indexing and note-fetching stages and the MedGemma extraction. Stages that would
write into linked directories refuse to run.

Every method writes `results/<run>/{metrics.json, predictions_dev.csv, predictions_test.csv}`.
`evaluation/make_table.py` reports each method on the test split, with F1, precision and recall at the F1-optimal
threshold of the same run's development predictions.

The scripts are written for the 3-month lead. The 6-, 9- and 12-month leads use the same stages with `--lead 6/9/12`.

## Data pipeline (`scripts/10_data_pipeline.sh`)

| Stage | Script | What it does |
|---|---|---|
| 1 | `s1_build_index.py --table all` | streams the OMOP CSVs and the note metadata to parquet; de-duplicates notes; measurement whitelist; antidepressant ingredients via the concept ancestor table |
| 1b | `s1b_note_offsets.py` | byte-offset index over the note CSV files for random access |
| 2 | `s2_align_dates.py` | the note metadata and the OMOP tables carry independent per-patient date shifts; estimates each patient's offset from notes linked to OMOP visits and aligns note dates to the OMOP timeline |
| 3 | `s3_cohort.py` | label = first target ICD code (F32, F33, F34.1, 296.2x, 296.3x, 311, 300.4); age ≥ 12; cutoff = last in-person/telehealth visit on or before anchor − lead, within 180 days; negatives: real visit anchor, no depression-spectrum code ever, ≥ 365 days of follow-up, anchor year matched to positives |
| 4 | `s4_fetch_notes.py` | fetches whitelisted notes before the cutoff; cleaning, section condensing, regex evidence flags (`note_text.py`) |
| 5 | `s5_finalize_cohort.py` | note-evidence flags (stated diagnosis, PHQ-9 ≥ 10), case type, patient-level stratified 70/15/15 split |
| 5b | `s5b_add_prevalent.py --lead 3` | adds the explicit positives of the lead under the same eligibility rules |
| 6 | `s6_assemble.py` | one JSON record per patient: metadata, dated structured events, cleaned notes, XML-formatted record; nothing after the cutoff |
| 6b | `s6b_merge_dataset.py --lead 3` | merges the two record sets per split → `outputs/dataset/lead_3mo/` |

## Reproducibility

The study data cannot be shared, but nothing in the code is tied to them. The full pipeline can be reproduced on any
comparable de-identified EHR extract that has structured data in the OMOP common data model and clinical notes linked to
OMOP visits. All models are public HuggingFace checkpoints.

Required inputs:

* **OMOP CDM tables** as CSV files in `omop_dir`: `person`, `observation_period`, `death`, `provider`, `care_site`,
  `visit_occurrence`, `condition_occurrence`, `drug_exposure`, `procedure_occurrence`, `observation` and `measurement`.
  The columns used are listed in `TABLES` in `pipeline/s1_build_index.py`. Diagnoses must carry ICD-9/ICD-10 codes in
  `condition_source_value`.
* **OMOP vocabulary** in `concept_dir`: `Concept.csv` and `ConceptAncestor.csv`. Antidepressants are found through their
  RxNorm ingredients.
* **Clinical notes**:
  * the note text, as CSV files matching `notes_glob`, with one record `"note_id","text"` per note;
  * a note-metadata CSV with one row per note version: note id, version number, patient id, encounter date, note type,
    linked visit id, and author type and specialty. The column names are listed in `t_note_meta` in
    `pipeline/s1_build_index.py`; rename your columns to match, or edit that function.

To run on another dataset:

1. Set the source paths and `notes_glob` in `config.yaml`. Set the environment and Slurm settings in `scripts/env.sh`.
2. Adapt the site-specific vocabularies in `config.yaml`:
   * the note-type whitelist (`notes.whitelist_types`);
   * the visit types that cannot serve as anchors (`cohort.anchor_visit_exclude_source_values`);
   * the measurement and observation source values;
   * if needed, the outcome and washout code lists (`phenotype`).
3. Stage 2 estimates and removes a per-patient offset between note dates and OMOP dates. In a dataset where both sources
   share the same dates, it estimates an offset of zero and leaves the dates unchanged.
4. Run the scripts in the order given under "Running", then `scripts/40_make_table.sh`.

The cohort definition (lead times, age, look-back and follow-up rules, explicit-evidence criteria) is set entirely in
`config.yaml`, so the same study design can be applied to other health systems.

## License

MIT (see `LICENSE`).
