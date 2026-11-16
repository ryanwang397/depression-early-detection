#!/bin/bash
# Row 1: Clinical ModernBERT, full fine-tuning on the XML record (recent-first packing, 8,192 tokens), <=6 epochs with early stopping
set -euo pipefail; source "$(dirname "$0")/env.sh"
submit ded_mbert "$(GPU 64gb 24:00:00)" $DED_ROOT/experiments '$PY run_modernbert_sft.py --lead 3 --tag mbert --mode notes_ehr --max-length 8192 --tuning full --epochs 6 --batch 8 --grad-accum 2 --eval-steps 800 --patience 5'
