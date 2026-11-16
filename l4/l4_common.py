"""Shared paths/helpers for the L4 Evidence Ledger pipeline (LLM structured extraction from notes).

Layout (all under outputs/l4/):
  queue/part_XX.parquet     per-note extraction queue (text included), one file per notes shard
  note_queue_index.parquet  compact index of the queue (no text)
  extractions/*.jsonl       raw guided-JSON extractions (one line per note, resumable)
  ledger.parquet            validated, exploded evidence rows
  features_lead{L}.parquet  per (person, lead) longitudinal features
"""
from __future__ import annotations
import glob, hashlib, json, logging, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "l4"
QUEUE_DIR = OUT / "queue"
EXTRACT_DIR = OUT / "extractions"
NOTES_DIR = ROOT / "outputs" / "notes"
COHORT = Path(os.environ.get("DED_COHORT_FILE", ROOT / "outputs" / "cohort" / "lead3_aug_cohort.parquet"))   # study cohort = lead-3 main cohort + prevalent positives (pipeline/s5b_add_prevalent.py)
DATASET = ROOT / "outputs" / "dataset"
MODEL_GLOB = os.environ.get("DED_HF_HUB", os.path.expanduser("~/.cache/huggingface/hub")) + "/models--google--medgemma-27b-text-it/snapshots/*/"
MAX_TEXT_CHARS = 6000          # condensed text cap stored in the queue
NARRATIVE_SECTIONS = {"hpi", "social", "psych_mh", "assessment_plan"}


def model_dir() -> str:
    m = sorted(glob.glob(MODEL_GLOB))
    if not m:
        raise FileNotFoundError(MODEL_GLOB)
    return m[0]


def get_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        lg.addHandler(h)
    lg.setLevel(logging.INFO)
    return lg


def write_summary(path: Path, payload: dict) -> None:
    payload = dict(payload); payload.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def note_shard(note_id: int, n_shards: int) -> int:
    """Deterministic shard assignment so array jobs partition the queue without coordination."""
    return int.from_bytes(hashlib.blake2b(f"l4|{note_id}".encode(), digest_size=4).digest(), "big") % n_shards


def load_done_ids(paths: list[Path]) -> set[int]:
    done = set()
    for p in paths:
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                try:
                    done.add(int(json.loads(line)["note_id"]))
                except Exception:
                    continue
    return done
