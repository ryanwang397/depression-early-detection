"""L4 extraction schema: a flat `type` enum (category/subtype), status/experiencer/severity enums, JSON schema for
vLLM guided decoding, and enum normalisation helpers.  SCHEMA_VERSION is stored with every extraction."""
from __future__ import annotations
import re

SCHEMA_VERSION = "l4_v1_2026-09-06"

CATEGORIES: dict[str, list[str]] = {
    "dep_symptom": ["depressed_mood", "anhedonia", "irritability", "hopelessness", "worthlessness_guilt", "crying",
                    "social_withdrawal", "sleep", "appetite_weight", "fatigue", "concentration", "psychomotor"],
    "anxiety": ["generalized_worry", "panic", "social_anxiety", "other"],
    "somatic": ["headache", "abdominal", "chest_pain_palpitations", "dizziness", "nausea", "musculoskeletal_pain", "pain_other", "unexplained_or_functional"],
    "suicidality": ["ideation", "plan_or_intent", "attempt"],
    "self_harm": ["nssi", "other"],
    "eating_body_image": ["restricting_bingeing_purging", "body_image_concern", "other"],
    "stressor": ["bullying", "parental_separation_or_conflict", "foster_dcf_cps", "abuse_or_violence", "bereavement", "relationship_breakup",
                 "academic_pressure", "financial_or_housing", "legal_or_incarceration", "family_member_illness", "other"],
    "school_behavior": ["iep_504_special_ed", "suspension_expulsion", "grades_decline_or_failing", "truancy_or_refusal", "conduct_or_aggression", "adhd_symptoms"],
    "family_mh_history": ["depression", "anxiety", "bipolar_or_psychosis", "suicide", "substance_use", "other"],
    "treatment": ["counseling_or_therapy", "psychiatry_visit_or_referral", "behavioral_health_referral", "crisis_or_inpatient_psych", "other"],
    "psych_med": ["antidepressant", "anxiolytic", "stimulant", "antipsychotic_or_mood_stabilizer", "sleep_aid", "other"],
    "screening": ["phq9", "phq2", "gad7", "cssrs", "other"],
    "substance": ["vaping_or_nicotine", "cannabis", "alcohol", "other"],
    "protective": ["family_support", "friends_or_peers", "activities_or_sports", "coping_or_resilience"],
    "repro_derm": ["contraception", "pregnancy_or_postpartum", "menstrual_problems", "acne_or_isotretinoin"],
    "depression_dx_statement": ["current", "past", "on_antidepressant"],
}
ALL_TYPES = [f"{c}/{s}" for c, subs in CATEGORIES.items() for s in subs]
STATUS = ["present", "negated", "history", "family", "uncertain", "template"]
EXPERIENCER = ["patient", "family_member", "other"]
SEVERITY = ["mild", "moderate", "severe", "unknown"]
MAX_ITEMS = 25
SPAN_MAX = 200

ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ALL_TYPES},
        "status": {"type": "string", "enum": STATUS},
        "experiencer": {"type": "string", "enum": EXPERIENCER},
        "severity": {"type": "string", "enum": SEVERITY},
        "value": {"type": ["number", "null"]},
        "span": {"type": "string", "maxLength": SPAN_MAX},
    },
    "required": ["type", "status", "experiencer", "severity", "value", "span"],
    "additionalProperties": False,
}
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "note_relevant": {"type": "boolean"},
        "items": {"type": "array", "maxItems": MAX_ITEMS, "items": ITEM_SCHEMA},
        "summary": {"type": "string", "maxLength": 300},
    },
    "required": ["note_relevant", "items", "summary"],
    "additionalProperties": False,
}

_TYPE_SET = set(ALL_TYPES)
_CAT_SET = set(CATEGORIES)


def normalize_type(t: str) -> str | None:
    """Map a model-emitted type to the enum; falls back to category/other when the subtype is unknown."""
    if not isinstance(t, str):
        return None
    t = t.strip().lower().replace(" ", "_").replace("-", "_")
    if t in _TYPE_SET:
        return t
    cat = t.split("/")[0]
    if cat in _CAT_SET:
        return f"{cat}/other" if "other" in CATEGORIES[cat] else f"{cat}/{CATEGORIES[cat][0]}"
    return None


def normalize_enum(v, allowed: list[str], default: str) -> str:
    if isinstance(v, str):
        v = v.strip().lower()
        if v in allowed:
            return v
    return default


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9<>]+")


def norm_text(s: str) -> str:
    return _WS.sub(" ", s.lower()).strip()


def norm_loose(s: str) -> str:
    return _PUNCT.sub(" ", s.lower()).strip()


def ground_span(span: str, text: str) -> str:
    """'exact' if the span is a verbatim substring (whitespace/case-insensitive), 'loose' if it matches after
    punctuation stripping, else 'none'."""
    if not span or not span.strip():
        return "none"
    if norm_text(span) in norm_text(text):
        return "exact"
    ls = norm_loose(span)
    if ls and ls in norm_loose(text):
        return "loose"
    return "none"
