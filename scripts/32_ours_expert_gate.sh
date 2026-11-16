#!/bin/bash
# Ours, auxiliary signals: tabular expert fit on implicit patients only + prior-evidence rule gate (-> lead3_expert_inc.parquet), care context
set -euo pipefail; source "$(dirname "$0")/env.sh"; AFTER=$(alive "${AFTER:-$(cat $DED_ROOT/logs/last_agent1_job 2>/dev/null || true)}")
G=$(submit ded_expert "$(CPU 8 64gb 04:00:00)" $DED_ROOT/eagrl '$PY e0_expert.py --lead 3 --train-incident-only --gate-prior-evidence && $PY f1_context_features.py --lead 3')
echo $G > $DED_ROOT/logs/last_expert_job
