#!/bin/bash
set -euo pipefail; source "$(dirname "$0")/env.sh"
$PY $DED_ROOT/evaluation/make_table.py --boot ${BOOT:-1000}     # runs under results/; 95% CIs from patient-level bootstrap
