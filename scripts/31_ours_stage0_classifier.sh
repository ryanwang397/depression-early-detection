#!/bin/bash
# Ours, stage 0: LoRA classifier on the evidence timeline, two stages (teacher + initialisation of the reasoning model)
set -euo pipefail; source "$(dirname "$0")/env.sh"; E=$DED_ROOT/experiments; AFTER=$(alive "${AFTER:-$(cat $DED_ROOT/logs/last_agent1_job 2>/dev/null || true)}")
L1=$(submit ded_lora1 "$(GPU 40gb 24:00:00)" $E '$PY run_llm_lora_cls.py --lead 3 --model llama-3.1-8b --tag lora1 --epochs 1 --lr 1e-4 --max-len 4096 --batch 2 --grad-accum 8 --eval-every 300 --dev-subset 2000')
L2=$(AFTER=$L1 submit ded_lora2 "$(GPU 40gb 24:00:00)" $E "\$PY run_llm_lora_cls.py --lead 3 --model llama-3.1-8b --tag lora2 --init-adapter $DED_ROOT/results/lead3_lora1_llama-3.1-8b_lora/adapter_best --epochs 1 --lr 5e-5 --max-len 4096 --batch 2 --grad-accum 8 --eval-every 300 --dev-subset 2000")
echo $L2 > $DED_ROOT/logs/last_stage0_job
