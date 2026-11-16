#!/bin/bash
# Ours, Agent 2: hindsight distillation -> soft targets -> reasoning SFT -> inference (results/lead3_ours_final_llama-3.1-8b)
set -euo pipefail; source "$(dirname "$0")/env.sh"; G=$DED_ROOT/eagrl; R=$DED_ROOT/results
S0=$(alive "${S0:-$(cat $DED_ROOT/logs/last_stage0_job 2>/dev/null || true)}"); EX=$(alive "${EX:-$(cat $DED_ROOT/logs/last_expert_job 2>/dev/null || true)}")
E1=$(AFTER=$EX submit ded_e1 "$(GPU 40gb 14:00:00)" $G '$PY e1_distill.py --lead 3 --model llama-3.1-8b --expert-variant _inc --tag sft_inc --n-pos 1500 --n-neg 1500 --k-unguided 3 --k-guided 3 --keep 2')
E5=$(AFTER=$E1${S0:+:$S0} submit ded_e5 "$(GPU 40gb 06:00:00)" $G "\$PY e5_direct_score.py --lead 3 --adapter $R/lead3_lora2_llama-3.1-8b_lora/adapter_best --splits train --pids-file $DED_ROOT/outputs/eagrl/lead3_sft_inc.jsonl --tag lora2_sftpat_inc")
E2=$(AFTER=$E5 submit ded_e2 "$(GPU 40gb 14:00:00)" $G "\$PY e2_sft.py --lead 3 --model llama-3.1-8b --tag sft_expctx_inc --expert-variant _inc --init-adapter $R/lead3_lora2_llama-3.1-8b_lora/adapter_best --epochs 2 --lr 5e-5 --lambda-ans 2.0 --expert-opinion --care-context --soft-targets $R/eagrl/direct_lora2_sftpat_inc.csv --soft-mix 0.8 --direct-multitask")
E4=$(AFTER=$E2 submit ded_e4 "$(GPU 40gb 06:00:00)" $G "\$PY e4_eval.py --lead 3 --model llama-3.1-8b --expert-variant _inc --adapter $R/eagrl/lead3_sft_expctx_inc_llama-3.1-8b/adapter_final --tag ours_final --faithfulness 300 --expert-opinion --care-context")
echo "reasoning chain submitted; last job: $E4"
