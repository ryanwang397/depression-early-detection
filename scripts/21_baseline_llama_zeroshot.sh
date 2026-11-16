#!/bin/bash
# Row 2: Llama-3.1-8B-Instruct zero-shot on the XML record (Yes/No first-token log-odds); no training
set -euo pipefail; source "$(dirname "$0")/env.sh"
submit ded_zsxml "$(GPU 40gb 08:00:00)" $DED_ROOT/experiments '$PY run_llm_zeroshot.py --lead 3 --model llama-3.1-8b --input xml --tag zsxml --max-model-len 8192 --gpu-mem 0.92 --batch 500'
