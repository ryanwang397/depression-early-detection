#!/bin/bash
# Row 3: Llama-3.1-8B LoRA classifier on the XML record, two stages (1 epoch @1e-4, then 1 epoch @5e-5 from the first adapter)
set -euo pipefail; source "$(dirname "$0")/env.sh"; E=$DED_ROOT/experiments
X1=$(submit ded_xml1 "$(GPU 40gb 24:00:00)" $E '$PY run_llm_lora_cls.py --lead 3 --model llama-3.1-8b --input xml --tag xml1 --epochs 1 --lr 1e-4 --max-len 4096')
AFTER=$X1 submit ded_xml2 "$(GPU 40gb 24:00:00)" $E "\$PY run_llm_lora_cls.py --lead 3 --model llama-3.1-8b --input xml --tag xml2 --init-adapter $DED_ROOT/results/lead3_xml1_llama-3.1-8b_lora/adapter_best --epochs 1 --lr 5e-5 --max-len 4096"
