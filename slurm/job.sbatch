#!/bin/bash
#SBATCH --nodes=1 --ntasks=1
# Generic job wrapper.  usage: sbatch [resources] slurm/job.sbatch WORKDIR "command line"   ($PY = python interpreter)
set -euo pipefail
cd "$1"; shift
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
# one vLLM compile cache per job: concurrent jobs sharing ~/.cache/vllm corrupt each other's torch.compile artefacts
export VLLM_CACHE_ROOT=${DED_ROOT:?}/outputs/vllm_cache/job_${SLURM_JOB_ID}; mkdir -p "$VLLM_CACHE_ROOT"
hostname; date; echo "CMD: $1"; eval "$1"; date
