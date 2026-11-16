"""Validation / normalisation of raw extractions into ledger rows.

validate_extraction(raw_obj, text) -> (rows, stats)
  rows:  list of dicts {type, category, subtype, status, experiencer, severity, value, span, grounding}
  stats: {n_items, n_kept, n_dropped_type, n_dropped_span, n_loose, note_relevant, summary}
Items whose span is not found in the note (after whitespace/case and then punctuation normalisation) are kept but flagged grounding='none'
(n_dropped_span counts them); downstream feature building uses grounded rows only."""
from __future__ import annotations
import re
from schema import normalize_type, normalize_enum, ground_span, STATUS, EXPERIENCER, SEVERITY, SPAN_MAX

_NAMEY = re.compile(r"\b(?:Dr|Mr|Mrs|Ms)\.?\s+[A-Z][a-z]+")

# psych_med gate: keep the item only if the span names a psychiatric/sleep drug; subtype is then assigned from the drug.
_PSYCH_DRUGS = {
    "antidepressant": r"sertraline|zoloft|fluoxetine|prozac|escitalopram|lexapro|citalopram|celexa|paroxetine|paxil|fluvoxamine|luvox|venlafaxine|effexor|desvenlafaxine|pristiq|duloxetine|cymbalta|bupropion|wellbutrin|mirtazapine|remeron|vilazodone|viibryd|vortioxetine|trintellix|amitriptyline|elavil|nortriptyline|pamelor|imipramine|doxepin|clomipramine|ssri|snri|antidepressant",
    "anxiolytic": r"buspirone|buspar|hydroxyzine|atarax|vistaril|lorazepam|ativan|clonazepam|klonopin|alprazolam|xanax|diazepam|valium|propranolol",
    "stimulant": r"methylphenidate|ritalin|concerta|daytrana|focalin|dexmethylphenidate|amphetamine|adderall|vyvanse|lisdexamfetamine|atomoxetine|strattera|guanfacine|intuniv|tenex|clonidine|kapvay|viloxazine|qelbree",
    "antipsychotic_or_mood_stabilizer": r"aripiprazole|abilify|risperidone|risperdal|quetiapine|seroquel|olanzapine|zyprexa|ziprasidone|lurasidone|latuda|haloperidol|haldol|lamotrigine|lamictal|lithium|valpro|depakote|antipsychotic|mood stabilizer",
    "sleep_aid": r"melatonin|trazodone|desyrel|zolpidem|ambien|doxylamine|unisom|diphenhydramine for sleep",
}
_PSYCH_RE = {k: re.compile(v, re.I) for k, v in _PSYCH_DRUGS.items()}


def psych_med_subtype(span: str) -> str | None:
    for sub, rx in _PSYCH_RE.items():
        if rx.search(span):
            return sub
    return None

# ---- deterministic category gates (the LLM over-uses some categories for generic problem-list / family-history / screening text)
_DEP_DX = re.compile(r"depress|dysthym|\bmdd\b|major depressive|mood disorder|\b(?:311|300\.4|296\.[23]\d*|f3[23]\.?\d*)\b|adjustment (?:disorder|reaction)[^.]{0,40}depress|antidepressant|\bssri\b", re.I)
_ADHD = re.compile(r"\badhd\b|\badd\b|attention deficit", re.I)
_ANX_DX = re.compile(r"anxiety|panic|ptsd|post-?traumatic|adjustment (?:disorder|reaction)|\bocd\b|obsessive", re.I)
_OTHER_PSYCH = re.compile(r"bipolar|schizo|psychosis|psychotic|oppositional|conduct disorder|autis|asperger|intermittent explosive|tourette|eating disorder|anorexia|bulimia|mood swings|psychiatric (?:illness|history|disorder)|mental illness", re.I)
_FAM_MH = re.compile(r"depress|anxiety|bipolar|schizo|psych|suicid|substance|alcohol|drug|mental|adhd|autis|ptsd|mood|\bocd\b|self[- ]harm|overdose", re.I)
_ROS_ARTIFACT = re.compile(r"^for (?:confusion|agitation|anxiety|depression|substance|suicid|hallucinat|sleep|behavior|mood|inappropriate|altered)", re.I)
_SCREEN = re.compile(r"phq|gad-?7|c-?ssrs|columbia|depress|anxiety|mood|suicid|mental|psychosocial|sdq|psc|hopeless|little interest|feeling down|screen", re.I)


def gate_item(t: str, span: str) -> str | None:
    """Return the (possibly re-mapped) type, or None to drop the item."""
    cat, sub = t.split("/", 1)
    if cat == "depression_dx_statement":
        if _DEP_DX.search(span):
            return t
        if _ADHD.search(span):
            return "school_behavior/adhd_symptoms"
        if _ANX_DX.search(span):
            return "anxiety/other"
        if _OTHER_PSYCH.search(span):
            return "other_psych_dx/other"
        return None
    if cat == "family_mh_history" and not _FAM_MH.search(span):      # LLM labels e.g. "diabetes maternal grandmother" as family depression
        return None
    if _ROS_ARTIFACT.match(span):                                      # L2 section splitter drops "Negative" from "Psychiatric: Negative for confusion"
        return None
    if cat == "screening" and sub == "other" and not _SCREEN.search(span):
        return None
    return t


def scrub_span(span: str) -> str:
    return _NAMEY.sub("<NAME>", span)[:SPAN_MAX]


def validate_extraction(raw: dict | None, text: str) -> tuple[list[dict], dict]:
    stats = {"n_items": 0, "n_kept": 0, "n_dropped_type": 0, "n_dropped_span": 0, "n_dropped_drug": 0, "n_dropped_gate": 0, "n_loose": 0, "note_relevant": None, "summary": None}
    rows: list[dict] = []
    if not isinstance(raw, dict):
        return rows, stats
    stats["note_relevant"] = bool(raw.get("note_relevant")) if raw.get("note_relevant") is not None else None
    s = raw.get("summary"); stats["summary"] = s[:300] if isinstance(s, str) else None
    items = raw.get("items") or []
    if not isinstance(items, list):
        return rows, stats
    seen = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        stats["n_items"] += 1
        t = normalize_type(it.get("type"))
        if t is None:
            stats["n_dropped_type"] += 1; continue
        span = it.get("span") if isinstance(it.get("span"), str) else ""
        t = gate_item(t, span)
        if t is None:
            stats["n_dropped_gate"] += 1; continue
        if t.startswith("psych_med/"):
            sub = psych_med_subtype(span)
            if sub is None:
                stats["n_dropped_drug"] += 1; continue
            t = f"psych_med/{sub}"
        g = ground_span(span, text)
        if g == "none":
            stats["n_dropped_span"] += 1      # kept in the ledger with grounding="none"; feature step ignores them by default
        if g == "loose":
            stats["n_loose"] += 1
        status = normalize_enum(it.get("status"), STATUS, "uncertain")
        key = (t, status, span.strip().lower()[:80])
        if key in seen:
            continue
        seen.add(key)
        val = it.get("value"); val = float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else None
        cat, sub = t.split("/", 1)
        rows.append({"type": t, "category": cat, "subtype": sub, "status": status,
                     "experiencer": normalize_enum(it.get("experiencer"), EXPERIENCER, "patient"),
                     "severity": normalize_enum(it.get("severity"), SEVERITY, "unknown"), "value": val,
                     "span": scrub_span(span.strip()), "grounding": g})
        stats["n_kept"] += 1
    return rows, stats
