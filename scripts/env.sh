#!/bin/bash
# Cluster / environment settings shared by every run script.  Edit here, nowhere else.
export DED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PY=${PY:-python}                                         # python 3.10, see requirements.txt
export DED_HF_HUB=${DED_HF_HUB:-$HOME/.cache/huggingface/hub}   # HuggingFace hub cache holding the three model snapshots
# FROM_SOURCE=1 (default) builds every intermediate from the raw extract.  FROM_SOURCE=0 instead links precomputed,
# read-only intermediates from an earlier run: $DED_ARCHIVE/{index,notes} (stages 1, 1b, 2, 4) and the MedGemma extraction
# shards $DED_ARCHIVE/l4/extractions/full_tier1_shard*.jsonl (main cohort) + $DED_ARCHIVE_AUG/l4/extractions/aug3_tier1_shard*.jsonl
# (explicit positives).  See README.
export DED_ARCHIVE=${DED_ARCHIVE:-}
export DED_ARCHIVE_AUG=${DED_ARCHIVE_AUG:-}
FROM_SOURCE=${FROM_SOURCE:-1}
link_archive() { local src=$1 dst=$2     # link an archived file/dir into outputs/, never over real data
  [ -e "$src" ] || { echo "archived input missing: $src" >&2; exit 1; }
  if [ -e "$dst" ] && [ ! -L "$dst" ]; then echo "refusing: $dst exists and is not a link (remove it or run with FROM_SOURCE=1)" >&2; exit 1; fi
  mkdir -p "$(dirname "$dst")"; ln -sfn "$src" "$dst"; }
no_links() { local d; for d in "$@"; do if [ -L "$d" ] || { [ -d "$d" ] && [ -n "$(find "$d" -maxdepth 1 -type l -print -quit)" ]; }; then echo "refusing FROM_SOURCE=1: $d is (or contains) a link to the archive; remove the links first" >&2; exit 1; fi; done; }
# Slurm settings: set these to your cluster's values (drop --qos in CPU()/GPU() below if your cluster has no QOS).
ACCOUNT=${ACCOUNT:-your_account}; QOS_GPU=${QOS_GPU:-your_gpu_qos}; QOS_CPU=${QOS_CPU:-your_cpu_qos}
PART_GPU=${PART_GPU:-your_gpu_partition}   # one 80 GB-class GPU per job; the vLLM stages need B200/A100-class kernels
PART_CPU=${PART_CPU:-your_cpu_partition}
mkdir -p "$DED_ROOT/logs"
# submit NAME "sbatch resource flags" WORKDIR "command"   -> prints the job id.  Set AFTER=<jobid[:jobid]> to chain.
submit() { local name=$1 res=$2 wd=$3 cmd=$4
  sbatch --parsable --account=$ACCOUNT --job-name=$name --output=$DED_ROOT/logs/%x_%j.out --error=$DED_ROOT/logs/%x_%j.err \
         --export=ALL,DED_ROOT=$DED_ROOT,PY=$PY,DED_HF_HUB=$DED_HF_HUB ${AFTER:+--dependency=afterok:$AFTER} $res $DED_ROOT/slurm/job.sbatch "$wd" "$cmd"; }
alive() { local out="" j; for j in ${1//:/ }; do [ -n "$(squeue -h -j "$j" 2>/dev/null)" ] && out="${out:+$out:}$j"; done; echo "$out"; }   # keep only job ids still queued/running (a finished id cannot be a dependency)
CPU()  { echo "--qos=$QOS_CPU --partition=$PART_CPU --cpus-per-task=${1:-4} --mem=${2:-48gb} --time=${3:-06:00:00}"; }
GPU()  { echo "--qos=$QOS_GPU --partition=$PART_GPU --gres=gpu:1 --cpus-per-task=4 --mem=${1:-40gb} --time=${2:-24:00:00}"; }
