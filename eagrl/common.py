"""EAG-RL (arXiv 2508.13579) adapted to the depression early-detection evidence timelines.

Shared pieces: concept vocabulary (the "features" an LLM may cite and the expert model is aligned on), feature-column ->
concept mapping for the HGB expert, question-decomposition prompts, trajectory parsing, rewards, tokenizer helpers.

Adaptations vs. the paper (see README_eagrl.md):
  * expert = HGB on l4_plus_all features; "attention" = per-patient occlusion attribution aggregated per concept (e0)
  * risk score = P(Yes) at the token after "<Final answer>" (never a verbalised probability)
  * Stage 1 = hindsight-guided + unguided rejection sampling (STaR-style) instead of MCTS (e1), SFT with an extra
    answer-position cross-entropy (e2); Stage 2 = GRPO with entropy-adaptive up-clipping + answer CE (e3)
  * reward adds a grounding penalty: cited concepts must have evidence in the timeline
"""
from __future__ import annotations
import json, re, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from exp_common import person_hash, metrics, save_run, RESULTS   # noqa: E402,F401
from llm_common import MODELS, model_dir, yes_no_token_ids   # noqa: E402,F401
import llm_common as _lc   # noqa: E402
TL_SUFFIX = ""


def set_tl_suffix(sfx: str | None):
    global TL_SUFFIX; TL_SUFFIX = sfx or ""


def load_timeline(lead: int, split: str, max_n: int | None = None, seed_salt: str = "subset") -> list[dict]:
    """llm_common.load_timeline with the global timeline suffix (e.g. '_x3' -> lead3_train_x3.jsonl)."""
    return _lc.load_timeline(lead, f"{split}{TL_SUFFIX}", max_n, seed_salt)

OUT = ROOT / "outputs" / "eagrl"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------------------------------------------------
# Concept vocabulary: (name, grounding regex on the timeline text; None = always present in the header lines)
# ----------------------------------------------------------------------------------------------------------------------
CONCEPTS: list[tuple[str, str | None]] = [
    ("age", None), ("sex", None),
    ("visit frequency", None), ("change in visit frequency", None), ("ED visits", r"ED visits last year [1-9]"),
    ("inpatient admission", r"inpatient last year [1-9]|inpatient"), ("number of specialties", None), ("time since last visit", None),
    ("length of record", None), ("note volume", None), ("number of diagnoses", r"DIAGNOSES \("), ("number of medications", r"MEDICATIONS"),
    ("BMI or weight", r"BMI|obes|weight|GLP|semaglutide|liraglutide|tirzepatide"),
    ("depressive symptoms", r"dep_symptom/|depress|mood|anhedonia|R45|irritab|hopeless|cry"),
    ("sleep problems", r"sleep|insomnia|melatonin"),
    ("anxiety", r"anxiety|anxious|panic|worry|nervous"),
    ("somatic complaints", r"somatic/|headache|migraine|abdominal|nausea|fatigue|dizz|pain|palpitation"),
    ("suicidality", r"suicid"), ("self-harm", r"self_harm|self-harm|self harm|cutting|NSSI"),
    ("eating or body image concerns", r"eating|body image|anorexia|bulimia|purg|binge"),
    ("psychosocial stressors", r"stressor/|bully|abuse|divorce|separat|foster|DCF|CPS|bereave|homeless|housing|financial|legal|conflict|Z6[0-5]|Z5[5-9]"),
    ("school or behavior problems", r"school_behavior/|school|IEP|504|suspen|truan|conduct|oppositional|aggress"),
    ("ADHD", r"ADHD|adhd|attention deficit|stimulant|methylphenidate|amphetamine|hyperactiv"),
    ("family psychiatric history", r"family_mh_history/|\[family|/family_member|family history"),
    ("mental health treatment or referral", r"treatment/|psychiatr|counsel|therap|behavioral health|Psychology"),
    ("psychiatric medication", r"psych_med/|antidepress|SSRI|sertraline|fluoxetine|escitalopram|citalopram|bupropion|trazodone|hydroxyzine|clonazepam|lorazepam|aripiprazole|risperidone|quetiapine|Remeron|mirtazapine|Prozac|Zoloft|Lexapro"),
    ("screening questionnaires", r"screening/|PHQ|GAD-?7|C-?SSRS|questionnaire"),
    ("substance use", r"substance/|smok|tobacco|nicotine|vap|alcohol|cannabis|marijuana|THC|drug use|narcot"),
    ("protective factors", r"protective/|support"),
    ("reproductive or dermatologic", r"repro_derm/|contracept|pregnan|postpartum|menstru|acne|isotretinoin|OCP|Gynecology|Obstetrics"),
    ("prior depression mention", r"depression_dx_statement/|depress"),
    ("other psychiatric diagnosis", r"other_psych_dx/|PTSD|adjustment|bipolar|psychos|OCD|obsessive|tic|Tourette"),
    ("autism or developmental disorder", r"autis|Asperger|developmental|intellectual|learning dis"),
    ("PTSD or adjustment disorder", r"PTSD|post-?traumatic|adjustment"),
    ("diabetes", r"diabet|insulin|A1c"), ("chronic physical illness", r"epilep|seizure|asthma"),
]
CONCEPT_NAMES = [c for c, _ in CONCEPTS]
ALIGN_EXCLUDE = {"note volume", "length of record"}   # never counted in the expert/LLM Jaccard (documentation artefacts)
ALIGN_EXCLUDE_V3 = ALIGN_EXCLUDE | {"age", "sex", "visit frequency", "change in visit frequency", "time since last visit", "number of specialties", "number of diagnoses", "number of medications", "ED visits", "inpatient admission"}


def set_align_exclude_v3():
    """v3: alignment reward counts clinical concepts only and the expert top-5 comes from e0 --v3-exclude (lead{L}_expert_v3.parquet)."""
    global ALIGN_EXCLUDE; ALIGN_EXCLUDE = set(ALIGN_EXCLUDE_V3); set_expert_variant("_v3")
_GROUND = {c: (re.compile(p, re.I) if p else None) for c, p in CONCEPTS}
SYNONYMS = {"ed visit": "ED visits", "emergency visits": "ED visits", "emergency department visits": "ED visits", "gender": "sex", "female": "sex", "male": "sex",
            "bmi": "BMI or weight", "weight": "BMI or weight", "obesity": "BMI or weight", "sleep": "sleep problems", "insomnia": "sleep problems",
            "depression": "depressive symptoms", "depressed mood": "depressive symptoms", "mood symptoms": "depressive symptoms", "low mood": "depressive symptoms",
            "suicidal ideation": "suicidality", "self harm": "self-harm", "stressors": "psychosocial stressors", "psychosocial stressor": "psychosocial stressors",
            "school problems": "school or behavior problems", "behavior problems": "school or behavior problems", "family history": "family psychiatric history",
            "family mental health history": "family psychiatric history", "treatment": "mental health treatment or referral", "referral": "mental health treatment or referral",
            "psychiatry referral": "mental health treatment or referral", "counseling": "mental health treatment or referral", "therapy": "mental health treatment or referral",
            "medication": "psychiatric medication", "antidepressant": "psychiatric medication", "psychiatric medications": "psychiatric medication",
            "screening": "screening questionnaires", "phq-9": "screening questionnaires", "phq9": "screening questionnaires", "phq-9 score": "screening questionnaires",
            "substance": "substance use", "smoking": "substance use", "alcohol use": "substance use", "cannabis use": "substance use",
            "somatic symptoms": "somatic complaints", "somatic": "somatic complaints", "headache": "somatic complaints", "abdominal pain": "somatic complaints", "fatigue": "somatic complaints",
            "healthcare utilization": "visit frequency", "utilization": "visit frequency", "visits": "visit frequency", "visit count": "visit frequency", "number of visits": "visit frequency",
            "increasing visits": "change in visit frequency", "visit trend": "change in visit frequency", "trend in visits": "change in visit frequency",
            "specialties": "number of specialties", "days since last visit": "time since last visit", "recency": "time since last visit", "record length": "length of record",
            "notes": "note volume", "number of notes": "note volume", "diagnoses": "number of diagnoses", "medications": "number of medications",
            "eating disorder": "eating or body image concerns", "body image": "eating or body image concerns", "autism": "autism or developmental disorder", "asd": "autism or developmental disorder",
            "ptsd": "PTSD or adjustment disorder", "adjustment disorder": "PTSD or adjustment disorder", "asthma": "chronic physical illness", "epilepsy": "chronic physical illness",
            "prior depression": "prior depression mention", "history of depression": "prior depression mention", "contraception": "reproductive or dermatologic", "pregnancy": "reproductive or dermatologic",
            "acne": "reproductive or dermatologic", "protective": "protective factors", "social support": "protective factors", "social withdrawal": "depressive symptoms", "social isolation": "depressive symptoms", "bereavement": "psychosocial stressors", "appetite": "depressive symptoms", "low appetite": "depressive symptoms", "energy": "depressive symptoms", "irritability": "depressive symptoms", "epds": "screening questionnaires", "benzodiazepine": "psychiatric medication", "diazepam": "psychiatric medication", "alprazolam": "psychiatric medication", "buspirone": "psychiatric medication", "venlafaxine": "psychiatric medication", "duloxetine": "psychiatric medication", "amitriptyline": "psychiatric medication", "lamotrigine": "psychiatric medication", "lithium": "psychiatric medication", "anhedonia": "depressive symptoms", "hopelessness": "depressive symptoms", "concentration": "depressive symptoms", "self-harm": "self-harm", "sex and age": "sex", "age and sex": "age"}


def _norm(s: str) -> str:
    s = re.sub(r"\([^)]*\)", " ", s.lower())              # drop "(uncertain)" style qualifiers
    s = re.sub(r"[^a-z0-9\- ]", " ", s); return re.sub(r"\s+", " ", s).strip()


_NORM2NAME = {_norm(c): c for c in CONCEPT_NAMES}


_STOP = {"of", "the", "in", "and", "or", "a", "an", "to", "with", "for", "recent", "history", "current", "prior", "any", "level", "levels", "score", "scores", "signs", "symptoms", "symptom", "burden", "issues", "problems", "problem", "concerns", "concern", "status", "use"}
_KEYWORDS = [  # ordered; first hit wins
    ("family psychiatric history", r"family|parent|mother|father|sibling"),
    ("prior depression mention", r"prior depress|history of depress|previous depress|past depress|depression_dx"), ("suicidality", r"suicid"), ("self-harm", r"self.?harm|nssi|cutting"), ("screening questionnaires", r"phq|gad-?7|cssrs|screen|questionnaire"), ("sleep problems", r"sleep|insomnia"),
    ("eating or body image concerns", r"eating|body image|anorexi|bulimi"), ("psychosocial stressors", r"stressor|stress|bully|abuse|conflict|trauma"),
    ("school or behavior problems", r"school|behavio|conduct|truan"), ("adhd", r"adhd|attention deficit"), ("substance use", r"substance|alcohol|cannabis|marijuana|smok|vap|nicotine|drug"),
    ("psychiatric medication", r"medication|antidepress|ssri|psych.?med|drug therapy|prescri|benzo|zepam|zolam"), ("mental health treatment or referral", r"therap|counsel|psychiatr|referral|treatment|behavioral health"),
    ("somatic complaints", r"somatic|headache|abdominal|pain|fatigue|nausea|dizz"), ("anxiety", r"anxi|panic|worry"), ("depressive symptoms", r"depress|mood|anhedonia|hopeless|irritab"),
    ("PTSD or adjustment disorder", r"ptsd|adjustment"), ("autism or developmental disorder", r"autis|developmental|intellectual"), ("diabetes", r"diabet"), ("chronic physical illness", r"asthma|epilep|chronic"),
    ("BMI or weight", r"bmi|weight|obes"), ("ED visits", r"\bed\b|emergency"), ("inpatient admission", r"inpatient|admission|hospitali"), ("change in visit frequency", r"increas|trend|change|slope|rising"),
    ("number of specialties", r"specialt"), ("time since last visit", r"since last|recency|last visit"), ("length of record", r"record length|length of record|history length"),
    ("note volume", r"\bnotes?\b|documentation"), ("number of diagnoses", r"diagnos"), ("number of medications", r"medications?"), ("visit frequency", r"visit|utili|encounter|healthcare use"),
    ("protective factors", r"protective|support"), ("reproductive or dermatologic", r"contracept|pregnan|menstru|acne|reproduct|derm|gyn"), ("other psychiatric diagnosis", r"bipolar|ocd|psychos|psychiatric diagnosis"),
    ("age", r"\bage\b"), ("sex", r"\bsex\b|gender|female|male"),
]
_KEYWORDS = [(("ADHD" if c == "adhd" else c), re.compile(p)) for c, p in _KEYWORDS]


LEDGER_CAT = {"dep_symptom": "depressive symptoms", "anxiety": "anxiety", "somatic": "somatic complaints", "suicidality": "suicidality", "self_harm": "self-harm",
              "eating_body_image": "eating or body image concerns", "stressor": "psychosocial stressors", "school_behavior": "school or behavior problems",
              "family_mh_history": "family psychiatric history", "treatment": "mental health treatment or referral", "psych_med": "psychiatric medication",
              "screening": "screening questionnaires", "substance": "substance use", "protective": "protective factors", "repro_derm": "reproductive or dermatologic",
              "depression_dx_statement": "prior depression mention", "other_psych_dx": "other psychiatric diagnosis"}
_LEDGER_RE = re.compile(r"^\s*(" + "|".join(LEDGER_CAT) + r")(?:/([a-z0-9_]+))?", re.I)


def canon_concept(raw: str) -> str | None:
    """Map a model-written feature name to the canonical concept: ledger type -> exact -> synonym -> word containment -> keyword."""
    m = _LEDGER_RE.match(raw.strip().lower())
    if m:
        cat, sub = m.group(1), m.group(2) or ""
        if cat == "dep_symptom" and sub == "sleep": return "sleep problems"
        if cat == "school_behavior" and sub == "adhd_symptoms": return "ADHD"
        return LEDGER_CAT[cat]
    n = _norm(raw)
    if not n: return None
    if n in _NORM2NAME: return _NORM2NAME[n]
    if n in SYNONYMS: return SYNONYMS[n]
    w = set(n.split()) - _STOP; best = (0.0, None)
    for k, c in _NORM2NAME.items():
        kw = set(k.split()) - _STOP
        if not w or not kw: continue
        if kw <= w or w <= kw:
            sc = len(w & kw) / max(len(w | kw), 1) + 1.0
        else:
            sc = len(w & kw) / len(w | kw)
        if sc > best[0]: best = (sc, c)
    if best[0] >= 0.5: return best[1]
    for c, r in _KEYWORDS:
        if r.search(n): return c
    return None


def grounded(concept: str, text: str) -> bool:
    r = _GROUND.get(concept)
    return True if r is None else bool(r.search(text))


# ----------------------------------------------------------------------------------------------------------------------
# HGB feature column -> concept (first matching rule wins; the __family rule must precede the category rules)
# ----------------------------------------------------------------------------------------------------------------------
FEAT_RULES = [
    (r"^age_at_cutoff$", "age"), (r"^female$", "sex"),
    (r"^(visits_90|visits_180|visits_365|n_visits_365d|n_encounters|n_condition_days_365)$", "visit frequency"),
    (r"^(visits_prev365|slope_visits)$", "change in visit frequency"), (r"^(ed_365|n_ed_visits_365d)$", "ED visits"), (r"^inpatient_365$", "inpatient admission"),
    (r"^n_specialties_365$", "number of specialties"), (r"^(days_since_last_visit|recency_gap_days)$", "time since last visit"), (r"^history_days$", "length of record"),
    (r"^(n_notes|notes_365|chars_365|chars_all|rel_notes_365|notes_mh_relevance|notes_n_template_phrases|l4_n_ledger_notes|l4_n_relevant_notes|l4_relevant_rate|l4_items_per_note)$", "note volume"),
    (r"^n_distinct_icd3_365$", "number of diagnoses"), (r"^n_meds_365$", "number of medications"), (r"^(bmi_.*|dx_obesity|rx_glp1)$", "BMI or weight"),
    (r"^l4_.*__family_", "family psychiatric history"),
    (r"^(l4_dep_symptom.*|l4_n_dep_symptoms_present_365|notes_sx_mood__present|notes_sx_neuroveg__present|notes_dep_mention__present|dx_emotional_symptom)$", "depressive symptoms"),
    (r"^(l4_any_sleep_present_365|dx_sleep)$", "sleep problems"), (r"^(l4_anxiety.*|dx_anxiety|notes_anxiety__present)$", "anxiety"),
    (r"^(l4_somatic.*|l4_n_distinct_somatic_present|dx_headache|dx_abdominal_pain|dx_fatigue)$", "somatic complaints"),
    (r"^(l4_suicidality.*|l4_si_present_any|notes_suicidality__present)$", "suicidality"), (r"^(l4_self_harm.*|dx_self_harm_code|notes_self_harm__present)$", "self-harm"),
    (r"^(l4_eating_body_image.*|dx_eating_disorder)$", "eating or body image concerns"),
    (r"^(l4_stressor.*|l4_n_distinct_stressor_present|notes_stressor__present|dx_social_z)$", "psychosocial stressors"),
    (r"^(l4_school_behavior.*|notes_school__present|dx_conduct_odd)$", "school or behavior problems"), (r"^dx_adhd$", "ADHD"),
    (r"^(l4_family_mh_history.*|notes_fam_mh__family)$", "family psychiatric history"),
    (r"^(l4_treatment.*|notes_treatment__present|any_psychiatry_visit)$", "mental health treatment or referral"),
    (r"^(l4_psych_med.*|rx_psych_any|rx_weak_antidepressant|notes_psych_med__present)$", "psychiatric medication"),
    (r"^(l4_screening.*|l4_phq9_.*|l4_gad7_.*|phq9_.*|phq2_.*|gad7_.*)$", "screening questionnaires"),
    (r"^(l4_substance.*|dx_substance)$", "substance use"), (r"^l4_protective.*$", "protective factors"), (r"^l4_repro_derm.*$", "reproductive or dermatologic"),
    (r"^(l4_depression_dx_statement.*|l4_prevalent_flag|notes_dep_assert__present)$", "prior depression mention"),
    (r"^l4_other_psych_dx.*$", "other psychiatric diagnosis"), (r"^(dx_asd|dx_intellectual_dev)$", "autism or developmental disorder"), (r"^dx_ptsd_adjust$", "PTSD or adjustment disorder"),
    (r"^dx_diabetes$", "diabetes"), (r"^(dx_epilepsy|dx_asthma)$", "chronic physical illness"),
]
_FEAT_RULES = [(re.compile(p), c) for p, c in FEAT_RULES]


def concept_of_column(col: str) -> str | None:
    for r, c in _FEAT_RULES:
        if r.search(col): return c
    return None


def absent_value(col: str) -> float:
    """Reference ("feature removed") value used by the occlusion attribution in e0."""
    if col.startswith("l4_") and ("days_last" in col or "days_first" in col): return 9999.0
    if col.endswith("_days_before_cutoff") or col in ("bmi_last", "l4_phq9_last", "l4_gad7_last", "l4_phq9_max", "age_at_cutoff", "female", "history_days",
                                                       "recency_gap_days", "days_since_last_visit") or re.match(r"^(phq9|phq2|gad7)_(last|max)$", col):
        return float("nan")
    return 0.0


# ----------------------------------------------------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------------------------------------------------
SYSTEM = ("You are a clinical risk assessment model for a depression early-detection research study in adolescents and young adults. "
          "You are given a de-identified, compressed timeline of one patient's health record up to a prediction date: demographics, healthcare "
          "utilization, structured diagnoses and medications, evidence extracted from clinical notes (with status: present, negated, history, "
          "family, uncertain), one-line summaries of recent notes, and an excerpt of the most recent relevant note. Nothing after the prediction date is shown.\n\n"
          "Task: estimate whether this patient will receive a FIRST clinical diagnosis of depression (ICD-10 F32/F33/F34.1 or equivalent) within the next "
          "{lead} months after the prediction date.\n\n"
          "Reason like a clinician doing a structured case review and use EXACTLY this format:\n"
          "<Subquestion 1> one focused question about this record </Subquestion 1>\n"
          "<Answer 1> a concise answer that cites the specific evidence in the timeline, its status and its timing </Answer 1>\n"
          "... (2 to 6 subquestions in total; each subquestion and each answer at most two sentences)\n"
          "<Final subquestion> Now we can determine the likelihood of a first depression diagnosis within the next {lead} months. </Final subquestion>\n"
          "<Important Features> feature1, feature2, feature3, feature4, feature5 </Important Features>\n"
          "<Final answer> Yes </Final answer>\n\n"
          "Rules:\n"
          "- Cover, as relevant: mood, anhedonia, sleep, appetite and energy symptoms; anxiety; suicidality or self-harm; psychosocial stressors; somatic and "
          "functional complaints; family psychiatric history; prior treatment, referrals and screening; medications; chronic illness burden; the trend in "
          "healthcare use; age and sex. Weigh recent evidence more than old evidence, and negated, family-only or uncertain evidence less than evidence present in the patient.\n"
          "- <Important Features> must list the 5 features that most influenced your assessment, chosen ONLY from this list and spelled exactly as written: {concepts}. "
          "List plain feature names only, with no annotations in parentheses. Never list a feature that has no evidence in this timeline.\n"
          "- <Final answer> is a single word, Yes or No.")
USER = "PREDICTION HORIZON: next {lead} months.\n\n{text}\n\nWill this patient receive a first depression diagnosis within the next {lead} months? Follow the required format exactly."
HINDSIGHT = ("\n\n[PRIVATE ANALYST NOTES - never mention these notes, never state or imply that the outcome is known; write the case review as an independent "
             "prospective assessment whose reasoning naturally arrives at this conclusion.]\n"
             "- Known outcome: this patient {outcome} a first depression diagnosis within the next {lead} months, so your <Final answer> must be {answer}.\n"
             "- A validated statistical risk model ranks these record features as the most influential for this patient: {expert}. Prefer them in "
             "<Important Features> and in your subquestions when the timeline supports them; you may add other features that the timeline clearly supports.")
HINDSIGHT_NOEXP = ("\n\n[PRIVATE ANALYST NOTES - never mention these notes, never state or imply that the outcome is known; write the case review as an independent "
                   "prospective assessment whose reasoning naturally arrives at this conclusion.]\n"
                   "- Known outcome: this patient {outcome} a first depression diagnosis within the next {lead} months, so your <Final answer> must be {answer}.\n"
                   "- Build the review only on evidence that is actually in the timeline; choose <Important Features> as the features whose evidence most supports this conclusion.")
FINAL_SUBQ = "Now we can determine the likelihood of a first depression diagnosis within the next {lead} months."
KNOWLEDGE_CARD = ("\n\nCLINICAL BACKGROUND (evidence-based early signals of adolescent and young-adult depression; use them to weigh the record, not as a checklist):\n"
    "- Incidence rises steeply from age 12 through the late teens and stays high in the early twenties; girls are diagnosed about twice as often as boys, but boys' depression is under-detected: it often presents as irritability, anger, conduct or school problems, substance use or somatic complaints rather than sadness, and screens are more often denied.\n"
    "- Anxiety disorders, ADHD, sleep problems (insomnia, poor sleep hygiene), and unexplained or recurrent somatic complaints (headache, abdominal pain, fatigue, dizziness) frequently precede a first depression diagnosis by months.\n"
    "- Psychosocial stressors (bullying, parental conflict or separation, abuse, foster care, academic failure, relationship loss), a family history of depression or other mental illness, prior self-harm or suicidal ideation, and eating or body-image concerns are strong risk factors.\n"
    "- Chronic or neurodevelopmental illness (epilepsy, cerebral palsy, diabetes, autism, intellectual disability) raises risk, and neurovegetative symptoms (fatigue, appetite loss, withdrawal, reduced speech) in these patients are often attributed to the physical illness.\n"
    "- A negative depression screen at a well visit weeks before diagnosis is common, especially in boys; the absence of a documented symptom is weak evidence, whereas any documented mental-health contact (counseling, psychiatry referral, psychiatric medication, PHQ score) is strong evidence.\n"
    "- Care context matters: patients seen by clinicians or clinics that screen routinely are more likely to have depression recorded; a rising visit frequency across several specialties in the last year is itself a signal.")
EXPERT_BLOCK = ("\n\nSTATISTICAL MODEL OPINION (a gradient-boosting model trained on the structured features of 28,000 patients; treat it as one colleague's input, "
                "verify it against the timeline, and say in your final subquestion whether you agree): estimated risk {level} (score {prob:.2f}, higher than {pct:.0%} of patients); "
                "features that drove its score: {feats}.")
CONTEXT_BLOCK = ("\n\nCARE CONTEXT (how often patients seen in these settings or by this clinician receive a depression code within the prediction window; this reflects "
                 "screening and coding practice rather than the patient's own symptoms, so separate 'is this patient likely depressed' from 'is a diagnosis likely to be recorded here'): {context}")


def expert_block(entry: dict) -> str:
    p = entry["prob"]; level = "high" if p >= 0.6 else ("elevated" if p >= 0.35 else ("moderate" if p >= 0.15 else "low"))
    sign = dict(entry["signed"]); feats = ", ".join(f"{c} ({'raises' if sign.get(c, 0) > 0 else 'lowers'} risk)" for c in entry["top"])
    return EXPERT_BLOCK.format(level=level, prob=p, pct=entry.get("rank", 0.5), feats=feats)


KNOWLEDGE = False


def set_knowledge_card(on: bool):
    global KNOWLEDGE; KNOWLEDGE = bool(on)


def build_messages(text: str, lead: int, hindsight: dict | None = None, expert: dict | None = None, context: str | None = None) -> list[dict]:
    u = USER.format(lead=lead, text=text)
    if expert is not None: u += expert_block(expert)
    if context: u += CONTEXT_BLOCK.format(context=context)
    if hindsight:
        fmt = dict(outcome="DID receive" if hindsight["label"] == 1 else "did NOT receive", lead=lead, answer="Yes" if hindsight["label"] == 1 else "No")
        if hindsight.get("expert"):   # expert-guided hindsight (EAG-RL style)
            u += HINDSIGHT.format(expert=", ".join(f"{c} ({'raises' if s > 0 else 'lowers'} risk)" for c, s in hindsight["expert"]), **fmt)
        else:                         # label-only hindsight (expert-free recipe v4)
            u += HINDSIGHT_NOEXP.format(**fmt)
    sys_ = SYSTEM.format(lead=lead, concepts=", ".join(CONCEPT_NAMES)) + (KNOWLEDGE_CARD if KNOWLEDGE else "")
    return [{"role": "system", "content": sys_}, {"role": "user", "content": u}]


def encode_prompt(tok, messages: list[dict], max_prompt_tokens: int) -> list[int]:
    """Chat-template prompt ids; over-long prompts keep the head (instructions) and the tail (most recent evidence)."""
    p = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(p, add_special_tokens=False)
    if len(ids) > max_prompt_tokens:
        head = 1400; ids = ids[:head] + tok.encode("\n...[older evidence truncated]...\n", add_special_tokens=False) + ids[-(max_prompt_tokens - head - 12):]
    return ids


# ----------------------------------------------------------------------------------------------------------------------
# Trajectory parsing & rewards
# ----------------------------------------------------------------------------------------------------------------------
_SUBQ = re.compile(r"<Subquestion\s*(\d+)>(.*?)</Subquestion\s*\d*>\s*<Answer\s*\d*>(.*?)</Answer\s*\d*>", re.S | re.I)
_FEAT = re.compile(r"<Important Features>(.*?)</Important Features>", re.S | re.I)
_FEAT_LINE = re.compile(r"(?:\*\*)?Important Features(?:\*\*)?\s*:\s*(.+)", re.I)
_ANS = re.compile(r"<Final answer>\s*\**\s*(Yes|No)\b", re.I)
_ANS_LINE = re.compile(r"(?:\*\*)?Final answer(?:\*\*)?\s*:\s*\**\s*(Yes|No)\b", re.I)
_FINAL = re.compile(r"Now we can determine", re.I)
_QLABEL = re.compile(r"^\s*(?:<Subquestion\s*\d*>|\**\s*Subquestion\s*\d*\s*[:.]?\**|\d+[.)])\s*", re.I)
_ALABEL = re.compile(r"^\s*(?:<Answer\s*\d*>|\**\s*Answer\s*\d*\s*[:.]?\**)\s*", re.I)
_STOPLINE = re.compile(r"Now we can determine|Important Features|Final answer|Final subquestion", re.I)
MAX_SUBQ = 10


_OPEN_Q = re.compile(r"<Subquestion\s*\d*>", re.I); _OPEN_A = re.compile(r"<Answer\s*\d*>", re.I)
_CLOSE = re.compile(r"</?(?:Subquestion|Answer)\s*\d*>", re.I)


def _pairs_tagged(text: str) -> list[tuple[str, str]]:
    """(question, answer) pairs from tagged output; closing tags are optional (Llama often omits them)."""
    end = len(text); m = re.search(r"<Final subquestion>|Now we can determine|<Important Features>", text, re.I)
    if m: end = m.start()
    body = text[:end]; opens = list(_OPEN_Q.finditer(body)); pairs = []
    for i, o in enumerate(opens):
        seg = body[o.end(): opens[i + 1].start() if i + 1 < len(opens) else len(body)]
        am = _OPEN_A.search(seg)
        if not am: continue
        q = _CLOSE.sub("", seg[: am.start()]).strip(); a = _CLOSE.sub("", seg[am.end():]).strip()
        if q and a: pairs.append((q, a))
    return pairs


def _pairs_plain(text: str) -> list[tuple[str, str]]:
    """(question, answer) pairs from untagged output: a question line ends with '?', its answer is the following text until the
    next question / final section; leading 'Subquestion N:' / 'Answer N:' labels are stripped."""
    lines = text.split("\n"); pairs = []; q = None; buf = []
    def flush():
        if q is not None:
            a = " ".join(x.strip() for x in buf if x.strip()); a = _ALABEL.sub("", a).strip()
            if a: pairs.append((q, a))
    for ln in lines:
        st = ln.strip()
        if not st: continue
        if _STOPLINE.search(st): break
        core = _QLABEL.sub("", st).replace("</Subquestion>", "").strip()
        is_q = core.endswith("?") and len(core) < 400 and not _ALABEL.match(st)
        if is_q:
            flush(); q = core.rstrip("*").strip(); buf = []
        elif q is not None:
            buf.append(st)
    flush(); return pairs


def parse_trajectory(text: str, timeline: str | None = None) -> dict:
    qa = _pairs_tagged(text); tagged = bool(qa)
    if not qa: qa = _pairs_plain(text)
    fm = _FEAT.search(text); feats_raw = fm.group(1) if fm else None
    if feats_raw is None:
        m = re.search(r"<Important Features>\s*([^\n<]+)", text, re.I) or _FEAT_LINE.search(text)
        if m: feats_raw = m.group(1).split("</Important")[0]
    raw = [re.sub(r"[.\s]+$", "", f.strip().strip("*")) for f in feats_raw.split(",") if f.strip()] if feats_raw else []
    concepts = []; unknown = []
    for f in raw:
        c = canon_concept(f)
        if c and c not in concepts: concepts.append(c)
        elif not c: unknown.append(f)
    concepts = concepts[:5]
    ungrounded = [c for c in concepts if timeline is not None and not grounded(c, timeline)]
    am = _ANS.search(text) or _ANS_LINE.search(text); answer = am.group(1).capitalize() if am else None
    leak = bool(re.search(r"known outcome|analyst note|private note|as instructed|statistical risk model|outcome is known|we are told|given that (the|this) patient (did|will)", text, re.I))
    has_final = bool(_FINAL.search(text))
    valid = (1 <= len(qa) <= MAX_SUBQ) and has_final and feats_raw is not None and len(concepts) >= 3 and answer is not None and not leak
    return {"qa": qa, "n_sub": len(qa), "concepts": concepts, "raw_features": raw, "unknown_features": unknown, "ungrounded": ungrounded, "answer": answer, "valid": valid, "leak": leak,
            "has_final_subq": has_final, "has_features": feats_raw is not None, "tagged": tagged}


def canonical_text(parsed: dict, lead: int, use_canonical_concepts: bool = True) -> str:
    """Rebuild a parsed trajectory in the canonical tag format (used for SFT targets and for scoring prefixes)."""
    qa = "\n".join(f"<Subquestion {i+1}> {q} </Subquestion {i+1}>\n<Answer {i+1}> {a} </Answer {i+1}>" for i, (q, a) in enumerate(parsed["qa"]))
    feats = parsed["concepts"] if use_canonical_concepts else parsed["raw_features"]
    return (f"{qa}\n<Final subquestion> {FINAL_SUBQ.format(lead=lead)} </Final subquestion>\n<Important Features> {', '.join(feats)} </Important Features>\n"
            f"<Final answer> {parsed['answer']} </Final answer>")


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def reward(parsed: dict, p_yes: float, label: int, expert_top: list[str], w_att: float = 0.5, w_ground: float = 0.1, beta: float = 0.5) -> dict:
    """R = log p(y*) + beta*margin(if correct side) + w_att*Jaccard(concepts, expert top) - w_ground*#ungrounded ; invalid format -> -3."""
    if not parsed["valid"]:
        return {"R": -3.0, "r_cls": float("nan"), "r_att": 0.0, "n_ungrounded": len(parsed["ungrounded"]), "p_correct": float("nan")}
    p = p_yes if label == 1 else 1.0 - p_yes; p = min(max(p, 1e-4), 1 - 1e-4)
    r_cls = math.log(p) + (beta * (p - 0.5) if p > 0.5 else 0.0)
    r_att = jaccard([c for c in parsed["concepts"] if c not in ALIGN_EXCLUDE], [c for c in expert_top if c not in ALIGN_EXCLUDE])
    R = r_cls + w_att * r_att - w_ground * len(parsed["ungrounded"])
    return {"R": R, "r_cls": r_cls, "r_att": r_att, "n_ungrounded": len(parsed["ungrounded"]), "p_correct": p}


# ----------------------------------------------------------------------------------------------------------------------
# Token-level helpers (answer position, feature span) that work for any tokenizer: cumulative per-token decode
# ----------------------------------------------------------------------------------------------------------------------
def token_spans(tok, ids: list[int]) -> tuple[int | None, tuple[int, int] | None]:
    """Returns (answer_pos, (feat_start, feat_end)) as indices into `ids` (feat_end exclusive):
    answer_pos = index of the token generated right after '<Final answer>' (the Yes/No token);
    feature span = tokens strictly inside <Important Features> ... </Important Features>.
    Implemented with per-token decode + char offsets so merged tokens such as '>\n' do not break the search."""
    pieces = [tok.decode([t]) for t in ids]; starts = []; pos = 0
    for p in pieces:
        starts.append(pos); pos += len(p)
    s = "".join(pieces)

    def tok_at(char_pos):   # first token whose start offset >= char_pos
        for i, st in enumerate(starts):
            if st >= char_pos: return i
        return None

    def tok_containing(char_pos):   # last token whose start offset <= char_pos
        j = None
        for i, st in enumerate(starts):
            if st <= char_pos: j = i
            else: break
        return j
    ans = None; m = re.search(r"<Final answer>|Final answer\**\s*:", s, re.I)
    if m:
        ans = tok_at(m.end())
    span = None; m1 = re.search(r"<Important Features>|Important Features\**\s*:", s, re.I)
    if m1:
        fs = tok_at(m1.end()); m2 = re.search(r"</Important Features>|\n", s[m1.end():])
        if fs is not None:
            fe = tok_containing(m1.end() + m2.start()) if m2 else len(ids)
            span = (fs, max(fs + 1, fe if fe is not None else len(ids)))
    return ans, span


def p_yes_from_logits(logits_row, yes_ids: list[int], no_ids: list[int]) -> float:
    import torch
    ly = torch.logsumexp(logits_row[yes_ids].float(), 0); ln = torch.logsumexp(logits_row[no_ids].float(), 0)
    return float(torch.sigmoid(ly - ln))


def p_yes_from_logprobs(lp: dict, yes_ids: list[int], no_ids: list[int]) -> float:
    ly = max([lp[t].logprob for t in yes_ids if t in lp], default=-30.0); ln = max([lp[t].logprob for t in no_ids if t in lp], default=-30.0)
    return 1.0 / (1.0 + math.exp(-(ly - ln)))


def resolve_path(p: str | None) -> str | None:
    """Absolute path; relative paths are tried against the cwd and then the project root."""
    if p is None: return None
    q = Path(p)
    if q.is_absolute() or q.exists(): return str(q.resolve())
    return str((ROOT / p).resolve())


def balanced_subset(recs: list[dict], n_pos: int, n_neg: int, salt: str, exclude: set | None = None) -> list[dict]:
    """Deterministic label-balanced subset by person hash (same idea as exp_common.load_split)."""
    ex = exclude or set(); pos = [r for r in recs if r["label"] == 1 and r["person_id"] not in ex]; neg = [r for r in recs if r["label"] == 0 and r["person_id"] not in ex]
    pos.sort(key=lambda r: person_hash(r["person_id"], salt)); neg.sort(key=lambda r: person_hash(r["person_id"], salt))
    out = pos[:n_pos] + neg[:n_neg]; out.sort(key=lambda r: person_hash(r["person_id"], salt + "_mix")); return out


EXPERT_VARIANT = ""


def set_expert_variant(v: str | None):
    global EXPERT_VARIANT; EXPERT_VARIANT = v or ""


def load_expert(lead: int) -> dict:
    """person_id -> {"top": [concepts], "signed": [(concept, delta)], "prob": p} from e0 (variant '' or '_v3')."""
    import pandas as pd
    df = pd.read_parquet(OUT / f"lead{lead}_expert{EXPERT_VARIANT}.parquet")
    if "gated" not in df: df["gated"] = False
    tr = np.sort(df.prob[(df.split == "train") & ~df.gated].values)
    rank = np.searchsorted(tr, np.nan_to_num(df.prob.values, nan=0.0)) / max(len(tr), 1)
    return {int(r.person_id): {"top": json.loads(r.top), "signed": json.loads(r.signed), "prob": float(r.prob), "rank": float("nan") if r.gated else float(rk), "gated": bool(r.gated)} for r, rk in zip(df.itertuples(), rank)}


def null_expert() -> dict:
    """Expert-free pipeline (--no-expert): every patient gets an empty top list and undefined probability, so no stage depends on e0."""
    import collections
    return collections.defaultdict(lambda: {"top": [], "signed": [], "prob": float("nan"), "rank": float("nan")})


def load_context(lead: int) -> dict:
    """person_id -> CARE CONTEXT line from f1_context_features.py."""
    import pandas as pd
    df = pd.read_parquet(OUT / f"lead{lead}_context.parquet", columns=["person_id", "context_text"])
    return {int(r.person_id): r.context_text for r in df.itertuples()}


def prompt_extras(pid: int, expert: dict | None, context: dict | None) -> dict:
    """kwargs for build_messages given the --expert-opinion / --care-context lookups (None when disabled)."""
    e = expert.get(pid) if expert is not None else None
    if e is not None and e.get("gated"): e = None   # gated patient (documented prior depression evidence): the expert is not consulted
    return {"expert": e, "context": context.get(pid) if context is not None else None}


def synthetic_trajectory(label: int, expert_top: list[str], lead: int, k: int = 3) -> str:
    """Only for --smoke tests: a format-valid trajectory built from the expert concepts."""
    feats = ((expert_top or ["depressive symptoms", "anxiety", "psychosocial stressors"]) + [c for c in CONCEPT_NAMES if c not in (expert_top or [])])[:5]
    qa = "\n".join(f"<Subquestion {i+1}> Is there evidence of {feats[i]}? </Subquestion {i+1}>\n<Answer {i+1}> The timeline documents {feats[i]} recently. </Answer {i+1}>" for i in range(k))
    return (f"{qa}\n<Final subquestion> {FINAL_SUBQ.format(lead=lead)} </Final subquestion>\n<Important Features> {', '.join(feats)} </Important Features>\n"
            f"<Final answer> {'Yes' if label == 1 else 'No'} </Final answer>")
