#!/bin/bash
# Build the study dataset outputs/dataset/lead_3mo/{train,dev,test}.jsonl
#   FROM_SOURCE=1 (default): build index/ and notes/ from the raw OMOP + note CSVs (config.yaml), then run stages 3, 5, 5b, 6, 6b
#   FROM_SOURCE=0            : link precomputed stage-1/1b/2/4 outputs ($DED_ARCHIVE/index, $DED_ARCHIVE/notes) and run stages 3, 5, 5b, 6, 6b
set -euo pipefail; source "$(dirname "$0")/env.sh"; P=$DED_ROOT/pipeline; O=$DED_ROOT/outputs
if [ "$FROM_SOURCE" = "1" ]; then
  no_links $O/index $O/notes
  J1=$(submit ded_s1  "$(CPU 4 48gb 04:00:00)" $P '$PY s1_build_index.py --table all')                                 # OMOP + note metadata -> parquet, note dedupe
  J1b=$(submit ded_s1b "$(CPU 2 32gb 04:00:00)" $P 'for i in $(seq 0 44); do $PY s1b_note_offsets.py --file-idx $i; done; $PY s1b_note_offsets.py --merge')
  J2=$(AFTER=$J1 submit ded_s2 "$(CPU 4 48gb 02:00:00)" $P '$PY s2_align_dates.py')                                   # per-patient note/OMOP date-shift alignment
  J3=$(AFTER=$J2 submit ded_s3 "$(CPU 4 48gb 02:00:00)" $P '$PY s3_cohort.py')
  J4=$(AFTER=$J3:$J1b submit ded_s4 "$(CPU 2 24gb 08:00:00) --array=0-15" $P '$PY s4_fetch_notes.py --shard $SLURM_ARRAY_TASK_ID --n-shards 16')
  PRE=$J4
else
  link_archive $DED_ARCHIVE/index $O/index; link_archive $DED_ARCHIVE/notes $O/notes; echo "linked precomputed index/ and notes/ from $DED_ARCHIVE"
  J3=$(submit ded_s3 "$(CPU 4 48gb 02:00:00)" $P '$PY s3_cohort.py')                                                  # phenotype, anchors, cutoffs, structured washout, negative sampling
  PRE=$J3
fi
J5=$(AFTER=$PRE submit ded_s5 "$(CPU 4 48gb 02:00:00)" $P '$PY s5_finalize_cohort.py && $PY s5b_add_prevalent.py --lead 3')   # note-evidence washout, splits; + prevalent positives
J6=""
for sp in train dev test; do
  a=$(AFTER=$J5 submit ded_s6m_$sp "$(CPU 2 64gb 06:00:00)" $P "\$PY s6_assemble.py --lead 3 --split $sp --out-subdir dataset_main")
  b=$(AFTER=$J5 submit ded_s6x_$sp "$(CPU 2 64gb 06:00:00)" $P "\$PY s6_assemble.py --lead 3 --split $sp --cohort-subdir cohort_extras_lead3 --out-subdir dataset_extras")
  J6="${J6:+$J6:}$a:$b"
done
J7=$(AFTER=$J6 submit ded_s6b "$(CPU 2 16gb 02:00:00)" $P '$PY s6b_merge_dataset.py --lead 3')
echo "data pipeline submitted (FROM_SOURCE=$FROM_SOURCE); last job: $J7"; echo $J7 > $DED_ROOT/logs/last_data_job
