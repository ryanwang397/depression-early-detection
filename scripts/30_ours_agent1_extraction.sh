#!/bin/bash
# Ours, Agent 1: XML record -> grounded evidence ledger -> ledger features + evidence timeline
#   FROM_SOURCE=1 (default): run the MedGemma-27B extraction (guided JSON, temperature 0; ~1.25 M notes, ~100 GPU-h) on NSHARD GPUs
#   FROM_SOURCE=0            : link precomputed extraction shards from $DED_ARCHIVE / $DED_ARCHIVE_AUG and rebuild queue, ledger, features, timelines on CPU
set -euo pipefail; source "$(dirname "$0")/env.sh"; L=$DED_ROOT/l4; X=$DED_ROOT/outputs/l4/extractions; NSHARD=${NSHARD:-8}
AFTER=$(alive "${AFTER:-$(cat $DED_ROOT/logs/last_data_job 2>/dev/null || true)}")   # wait for the data pipeline if it is still running
Q=$(submit ded_l4_s0 "$(CPU 8 96gb 04:00:00)" $L '$PY s0_note_queue.py')                                               # tier-1 narrative notes before the cutoff
if [ "$FROM_SOURCE" = "1" ]; then
  no_links $X
  E=$(AFTER=$Q submit ded_l4_s2 "$(GPU 64gb 36:00:00) --array=0-$((NSHARD-1))" $L '$PY s2_extract_vllm.py --tier 1 --shard $SLURM_ARRAY_TASK_ID --n-shards $SLURM_ARRAY_TASK_COUNT --out-tag full --batch 4000 --gpu-mem 0.92 --max-num-seqs 512')
  TAGS=full; DEP=$E
else
  for f in $DED_ARCHIVE/l4/extractions/full_tier1_shard*.jsonl $DED_ARCHIVE_AUG/l4/extractions/aug3_tier1_shard*.jsonl; do link_archive $f $X/$(basename $f); done
  echo "linked $(ls $X | wc -l) precomputed extraction shards"; TAGS=full,aug3; DEP=$Q
fi
T=$(AFTER=$DEP submit ded_l4_s347 "$(CPU 8 96gb 06:00:00)" $L "\$PY s3_build_ledger.py --tags $TAGS && \$PY s4_ledger_features.py --lead 3 && \$PY s7_render_timeline.py --lead 3")
echo "agent 1 submitted (FROM_SOURCE=$FROM_SOURCE); last job: $T"; echo $T > $DED_ROOT/logs/last_agent1_job
