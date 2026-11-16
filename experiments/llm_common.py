"""Shared pieces for LLM-only predictors (zero-shot and LoRA classifier) over the evidence timelines.

Timelines: outputs/l4/timeline/lead{L}_{split}.jsonl (from l4/s7_render_timeline.py).
Scoring convention (TRIAGE / logprob literature): the risk score is the log-odds of the answer tokens "Yes" vs "No"
at the first generated position; never a verbalized number."""
from __future__ import annotations
import os, json, glob
from pathlib import Path
from exp_common import ROOT, person_hash

TL_DIR = ROOT / "outputs" / "l4" / "timeline"
_HUB = os.environ.get("DED_HF_HUB", os.path.expanduser("~/.cache/huggingface/hub"))   # HuggingFace hub cache that holds the model snapshots (override with DED_HF_HUB)
MODELS = {
    "medgemma-27b": _HUB + "/models--google--medgemma-27b-text-it/snapshots/*/",
    "llama-3.1-8b": _HUB + "/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/",
    "medgemma-4b": _HUB + "/models--google--medgemma-4b-it/snapshots/*/",
}
INSTRUCTION = ("You are a clinical risk assessment model for a depression early-detection research study in adolescents and young adults. "
               "You are given a de-identified, compressed timeline of one patient's health record up to a prediction date: demographics, "
               "healthcare utilization, structured diagnoses and medications, evidence extracted from clinical notes (with status: present, "
               "negated, history, family, uncertain), one-line summaries of recent notes, and an excerpt of the most recent relevant note. "
               "Nothing after the prediction date is shown.\n\n"
               "Task: estimate whether this patient will receive a FIRST clinical diagnosis of depression (ICD-10 F32/F33/F34.1 or equivalent) "
               "within the next {lead} months after the prediction date. Consider mood/anhedonia/sleep/appetite/energy symptoms, anxiety, "
               "suicidality or self-harm, psychosocial stressors (family conflict, bullying, abuse, school problems), functional and somatic "
               "complaints, family psychiatric history, prior treatment or referrals, screening results, medications, chronic illness burden, "
               "the trend in healthcare use, and the patient's age and sex. Weigh recent evidence more than old evidence, and negated or "
               "family-only evidence less than evidence present in the patient.\n\n"
               "Answer with a single word: Yes or No.")


def model_dir(name: str) -> str:
    m = sorted(glob.glob(MODELS[name]))
    if not m:
        raise FileNotFoundError(f"{name}: {MODELS[name]}")
    return m[0]


def load_timeline(lead: int, split: str, max_n: int | None = None, seed_salt: str = "subset") -> list[dict]:
    recs = [json.loads(l) for l in open(TL_DIR / f"lead{lead}_{split}.jsonl")]
    if max_n and len(recs) > max_n:   # same deterministic label-stratified subsetting as exp_common.load_split
        import pandas as pd
        df = pd.DataFrame({"i": range(len(recs)), "y": [r["label"] for r in recs], "h": [person_hash(r["person_id"], seed_salt) for r in recs]})
        prev = df.y.mean(); n_pos = int(round(max_n * prev)); n_neg = max_n - n_pos
        keep = pd.concat([df[df.y == 1].nsmallest(n_pos, "h"), df[df.y == 0].nsmallest(n_neg, "h")]).sort_values("h").i.tolist()
        recs = [recs[i] for i in keep]
    return recs


def build_messages(text: str, lead: int) -> list[dict]:
    return [{"role": "system", "content": INSTRUCTION.format(lead=lead)},
            {"role": "user", "content": f"PREDICTION HORIZON: next {lead} months.\n\n{text}\n\nWill this patient receive a first depression diagnosis within the next {lead} months? Answer Yes or No."}]


def yes_no_token_ids(tok) -> tuple[list[int], list[int]]:
    """All single-token encodings of Yes/No variants (with/without leading space, case variants)."""
    def ids(words):
        out = []
        for w in words:
            t = tok.encode(w, add_special_tokens=False)
            if len(t) == 1: out.append(t[0])
        return sorted(set(out))
    return ids(["Yes", " Yes", "yes", " yes", "YES"]), ids(["No", " No", "no", " no", "NO"])
