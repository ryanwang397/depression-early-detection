"""Clinical-note text processing: L1 cleaning, L2 section parsing/condensing, L3 evidence extraction.

L1 clean_text      : de-id placeholder normalisation, Epic mojibake, unfilled SmartText, footers.
L2 split_sections  : segment a (single-line) Epic note into labelled section spans;
   condense        : keep only clinically narrative sections, drop stock template sentences.
L3 extract_evidence: regex + context-window extraction of depression-relevant evidence with status
                     (present / negated / history / family / uncertain / template).
All functions are pure and deterministic; regex tables are module constants so the exact rule set
is versioned with the code (RULES_VERSION).
"""
from __future__ import annotations
import re, html
from dataclasses import dataclass, asdict

RULES_VERSION = "notes_v1.8_2026-09-07"  # v1.8: do not strip "Psychiatric: Negative" when followed by "for <symptom>" (left dangling "for confusion" fragments)

# ============================================================================ L1
_MOJIBAKE = [("GÇó", " • "), ("GÇá", " "), ("GÇ£", '"'), ("GÇ¥", '"'), ("GÇÖ", "'"), ("GÇÿ", "'"), ("GÇô", "-"),
             ("GÇö", "-"), ("GÇª", "..."), ("Güä", "½"), ("Gé¼", "€"), ("-¦", "°"), ("GÇó", " • ")]
_MOJI_ANY = re.compile(r"G[Çü][^\sA-Za-z0-9]?")
_PLACEHOLDER = re.compile(r"\[\*\*([A-Z_]+)\*\*\]")
_PH_MAP = {"DATE": "<DATE>", "NAME": "<NAME>", "AGE": "<AGE>", "ID": "<ID>", "CONTACT_PHONE": "<PHONE>", "OTHER": "<OTHER>"}
_SMART_DEFAULT = re.compile(r'\{[^{}]{0,300}?::\s*"{1,2}(.*?)"{1,2}\s*\}', re.S)
_SMART_UNFILLED = re.compile(r"\{[^{}]{1,120}?:\d{3,}\}")
_STARS = re.compile(r"\*{3,}")
_FOOTERS = [
    re.compile(r"Interested in learning more about our referring physician portal.*$", re.I | re.S),
    re.compile(r"Please note before visiting that our campuses are tobacco-free.*$", re.I | re.S),
    re.compile(r"For more information, please visit us online at www\.\S+.*$", re.I | re.S),
    re.compile(r"Electronically signed by\b.*$", re.I | re.S),
    re.compile(r"This (?:message|document) was (?:electronically )?signed.*$", re.I | re.S),
    re.compile(r"CONFIDENTIALITY NOTICE.*$", re.I | re.S),
]
_MYCHART_HEAD = re.compile(r"^\s*From:\s*<NAME>.{0,120}?Sent:\s*<DATE>[^A-Za-z]{0,20}(?:AM|PM)?\s*[A-Z]{2,4}?\s*(?:Subject:)?", re.S)
_ONBEHALF = re.compile(r"This message is being sent by <NAME> on behalf of <NAME>\.?", re.I)
_WS = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    t = raw or ""
    if "&" in t:
        t = html.unescape(t)
    for a, b in _MOJIBAKE:
        t = t.replace(a, b)
    t = _MOJI_ANY.sub(" ", t)
    t = _PLACEHOLDER.sub(lambda m: _PH_MAP.get(m.group(1), "<LOC>" if m.group(1).startswith("LOCATION") else f"<{m.group(1)}>"), t)
    t = _SMART_DEFAULT.sub(lambda m: m.group(1), t)
    t = _SMART_UNFILLED.sub(" ___ ", t)
    t = _STARS.sub(" ___ ", t)
    for rx in _FOOTERS:
        t = rx.sub("", t)
    t = _MYCHART_HEAD.sub("", t)
    t = _ONBEHALF.sub("", t)
    t = t.replace('""', '"')
    return _WS.sub(" ", t).strip()


# ============================================================================ L2
# (group, regex). Matched at any position of the single-line note. Order = priority for ties.
SECTION_PATTERNS: list[tuple[str, str]] = [
    ("admin", r"\b(?:Addended by|Attestation|Dictated by|Document Reviewed|Document Revised|Document Released|Signed by|I personally (?:saw|examined)|cc:|CC:)"),
    ("assessment_plan", r"\b(?:Assessment and Plan|Assessment & Plan|Assessment/Plan|ASSESSMENT AND PLAN|ASSESSMENT/PLAN|ASSESSMENT & PLAN|A/P:|A&P:|Assessment:|ASSESSMENT:|Plan:(?!\s*(?:MC\b|MEDICAID|MEDICARE|HMO|PPO|[A-Z][A-Z ]{2,}/))|PLAN:|Impression:|IMPRESSION:|ED Course|ED COURSE|Medical Decision Making|MDM:|Clinical Impression|Diagnoses this visit|Final diagnos[ei]s|Discharge Diagnos[ei]s|Disposition:|DISPOSITION:|ED Disposition|Follow[- ]?up:|FOLLOW[- ]?UP:|Recommendations?:|RECOMMENDATIONS?:|Problem List|Patient Active Problem List|Ongoing Problem|Diagnosis:|DIAGNOSIS:)"),
    ("psych_mh", r"\b(?:Psychiatric:|PSYCHIATRIC:|Psychiatric/Behavioral|Psych:|PSYCH:|Mental Status(?: Exam)?|MENTAL STATUS|Mood:|Affect:|PHQ-?[29]|Patient Health Questionnaire|GAD-?7|Suicide Risk|Safety Assessment|Columbia|C-SSRS|Behavioral Health|Past Psychiatric History|PAST PSYCHIATRIC HISTORY|Psychiatric History|Psychiatric ROS|Stressors\s*/\s*Depression|Depression screen(?:ing)?|Depression Screen)"),
    ("social", r"\b(?:Social History|SOCIAL HISTORY|Social Hx|SOCIAL HX|HEADDSSS|HEEADSSS|HEADSSS|HEADSS|Home:|HOME:|Education:|EDUCATION:|School:|SCHOOL:|Activities:|ACTIVITIES:|Drugs:|DRUGS:|Sexuality|Safety:|SAFETY:|Substance (?:use|abuse)|Tobacco (?:use|abuse):|Alcohol use:|Living situation|Family Situation|Social History Narrative|Social History Main Topics)"),
    ("hpi", r"\b(?:History of Present Illness|HISTORY OF PRESENT ILLNESS|HPI:|HPI\b|Interval History|INTERVAL HISTORY|Interim History|Subjective:|SUBJECTIVE:?|Chief Complaint|CHIEF COMPLAINT|Reason for Visit|REASON FOR VISIT|Reason for Consult|Presenting Complaint|Hospital Course|HOSPITAL COURSE|Interval History/Concerns|Concerns:|History:|HISTORY:|Nutrition History|Narrative:)"),
    ("ros", r"\b(?:Review of Systems|REVIEW OF SYSTEMS|ROS:|ROS\b)"),
    ("pmh_fh", r"\b(?:Past Medical History|PAST MEDICAL HISTORY|Past Surgical History|PAST SURGICAL HISTORY|Family History|FAMILY HISTORY|Medical History:|Surgical History:|Birth History|Developmental History|Immunizations?:|IMMUNIZATIONS?:|Immunization History|Allerg(?:y|ies):|ALLERG(?:Y|IES):|No Known Allergies|PMH:|PSH:|FH:|FHx:)"),
    ("meds", r"\b(?:Current Outpatient Medications|Current Outpatient Prescriptions|Current Medications|CURRENT MEDICATIONS|Home Medications|HOME MEDICATIONS|Medications:|MEDICATIONS:|Discharge Medication|Medication List|Medication Sig|Scheduled Meds|PRN Meds|Continuous Infusions|Patient's Medications|New Prescriptions|Previous Medications|Modified Medications|Discontinued Medications)"),
    ("exam_vitals", r"\b(?:Physical Exam(?:ination)?:?|PHYSICAL EXAM(?:INATION)?:?|Objective:|OBJECTIVE:?|Vitals:|VITALS:|Vital Signs|VITAL SIGNS|Visit Vitals|BP Readings|Wt Readings|Ht Readings|Growth parameters|Exam:|EXAM:|Blood pressure \d|Pulse \d|Temp \d|Ht \d|Wt \d|BMI \d)"),
    ("_sub", r"\b(?:Constitutional:|General:|HENT:|Head:|Eyes:|Ears:|Nose:|Mouth/Throat:|Neck:|Cardiovascular:|Pulmonary/Chest:|Respiratory:|Chest:|Abdominal:|Abdomen:|Gastrointestinal:|Musculoskeletal:|Skin:|Neurological:|Genitourinary:|Endocrine:|Hematological[^:]{0,15}:|Allergic/Immunologic[^:]{0,5}:|Lymphadenopathy:|Extremities:)"),
    ("labs_data", r"\b(?:Labs?:|LABS?:|Lab Results|Results for|Recent Results|Imaging:|IMAGING:|Radiology:|Data Review|DATA REVIEW|Diagnostic(?:s| Results):|Component Value|EKG:|ECG:|Microbiology|Pathology:)"),
    ("education_template", r"\b(?:Patient Instructions|Patient Education|Anticipatory Guidance|ANTICIPATORY GUIDANCE|Return precautions|Discharge Instructions|Health Maintenance|Screening:|Well Child|WELL CHILD|Developmental Milestones|Milestones:|Counseled on|Counseling provided)"),
]
_SEC_RE = [(g, re.compile(p)) for g, p in SECTION_PATTERNS]
NARRATIVE_GROUPS = ("preamble", "hpi", "social", "psych_mh", "assessment_plan", "pmh_fh", "narrative")
PREAMBLE_MAX_CHARS = 1500

# stock sentences that carry no patient-specific information (removed from L2, counted in L3)
_NOT_ON_FILE = re.compile(r"(?:[A-Za-z][A-Za-z /'()-]{1,60}?:\s*Not on file\.?|[A-Za-z][A-Za-z /'()-]{1,60}?\s+Not Asked\b\.?|\bNot on file\b\.?|\bNot Asked\b\.?)")
TEMPLATE_NORMAL = re.compile(
    r"(?:(?:He|She|Patient|Pt) has a normal mood and affect\.?|normal mood and affect|Mood and affect (?:are )?normal\.?|Affect (?:is )?normal\.?|(?:The patient|He|She) is not nervous/anxious\.?|Nursing note and vitals reviewed\.?|Vitals reviewed\.?|(?:His|Her|Their) behavior is normal\.?|"
    r"Judgment and thought content normal\.?|(?:no|not in) acute distress|No distress\.?|well[- ]appearing|well developed,? (?:well nourished|no distress)|"
    r"(?:Alert|Awake) and oriented(?: x ?3| to person, place,? and time)?|Negative for depression|Negative for (?:suicidal ideation|self-injury)|"
    r"mood, memory, affect and judgment are normal|Psychiatric: Negative(?!\s+for)\.?|Psychiatric/Behavioral: Negative(?!\s+for)\.?|"
    r"(?:Denies|denies) (?:SI|HI|SI/HI|suicidal ideation|homicidal ideation)(?:,? (?:HI|homicidal ideation))?\.?|"
    r"There were no vitals filed for this visit\.?|There is no height or weight on file to calculate BMI\.?|"
    r"No current facility-administered medications (?:on file|for this visit)[^.]*\.?|No Known Allergies\.?)", re.I)


@dataclass
class Section:
    group: str
    start: int
    end: int
    header: str


def split_sections(text: str) -> list[Section]:
    hits = []
    for g, rx in _SEC_RE:
        for m in rx.finditer(text):
            hits.append((m.start(), m.end(), g, m.group(0)))
    hits.sort(key=lambda h: (h[0], -h[1]))
    # drop overlapping hits (keep first at a given position)
    secs: list[Section] = []
    last_end = -1
    cur_g, cur_s, cur_h = "preamble", 0, ""
    major_ctx = "exam_vitals"   # last enclosing organ-system context: ros or exam_vitals
    for s, e, g, h in hits:
        if s < last_end:
            continue
        if g in ("ros", "exam_vitals"):
            major_ctx = g
        if g == "_sub":  # organ-system sub-header inherits the enclosing ROS / exam context
            g = major_ctx
            if g == cur_g:
                last_end = e
                continue
        if s > cur_s or cur_h == "":
            secs.append(Section(cur_g, cur_s, s, cur_h))
        cur_g, cur_s, cur_h = g, s, h
        last_end = e
    secs.append(Section(cur_g, cur_s, len(text), cur_h))
    # remove empty spans, merge consecutive same-group spans
    out: list[Section] = []
    for sec in secs:
        if sec.end <= sec.start:
            continue
        if out and out[-1].group == sec.group:
            out[-1].end = sec.end
        else:
            out.append(sec)
    if len(out) <= 1 or sum(1 for s in out if s.group != "preamble") < 2:
        return [Section("narrative", 0, len(text), "")]
    return out


def condense(text: str, sections: list[Section], keep=NARRATIVE_GROUPS, drop_templates: bool = True) -> tuple[str, dict]:
    parts = []; kept = 0; dropped = 0; n_templates = 0
    for s in sections:
        seg = text[s.start:s.end]
        if s.group not in keep:
            dropped += len(seg); continue
        if s.group == "preamble" and len(seg) > PREAMBLE_MAX_CHARS:
            seg = seg[:PREAMBLE_MAX_CHARS] + " …"
        if drop_templates:
            seg, n = TEMPLATE_NORMAL.subn(" ", seg); n_templates += n
            seg = _NOT_ON_FILE.sub(" ", seg)
        seg = _WS.sub(" ", seg).strip()
        if len(seg) < 3:
            continue
        parts.append(f"[{s.group.upper()}] {seg}")
        kept += len(seg)
    return " ".join(parts), {"chars_in": len(text), "chars_kept": kept, "chars_dropped": dropped, "templates_removed": n_templates,
                             "sections": [s.group for s in sections]}


# ============================================================================ L3
_NEG = re.compile(r"\b(?:no|not|denies|denied|deny|denying|negative for|without|w/o|absence of|free of|none|never|nor|does not|doesn't|did not|didn't|has not|hasn't)\b", re.I)
_BUT = re.compile(r"\b(?:but|however|although|except)\b", re.I)
_FAMILY = re.compile(r"\b(?:family history|family hx|fhx|fh:|mother|mom|father|dad|parent|parents|sibling|brother|sister|grandmother|grandfather|grandma|grandpa|aunt|uncle|cousin|maternal|paternal|mother's|father's|mom's|dad's|son|daughter|wife|husband)\b", re.I)
_HISTORY = re.compile(r"\b(?:history of|hx of|h/o|past history|past medical history|pmh|previous|previously|prior|remote|in the past|resolved|remitted|formerly|years ago|as a child|childhood|prev dx)\b", re.I)
_UNCERTAIN = re.compile(r"\b(?:rule out|r/o|possible|possibly|suspected|suspect|concern for|concerns for|question of|questionable|screen(?:ing)? for|evaluat(?:e|ion) for|referred for|likely|may have|\?)", re.I)
_QUESTION = re.compile(r"\b(?:have you|do you|did you|has (?:he|she|your child|the patient)|does (?:he|she|your child|the patient)|are you|were you|is (?:he|she|the patient)|how often|over the (?:past|last) (?:two|2) weeks|in the past (?:year|month|2 weeks|two weeks)|little interest or pleasure|feeling down, depressed)\b", re.I)
_NONDEP = re.compile(r"\b(?:ST|st segment|skull|fontanel\w*|respiratory|CNS|bone marrow|myocardial|reflex\w*|mental status is not|drive)\b", re.I)

# category -> (regex, default_status_rule)
EVIDENCE_PATTERNS: dict[str, re.Pattern] = {
    "dep_dx": re.compile(r"\b(?:major depressive disorder|major depression|MDD\b(?! ?\d)|depressive disorder|depressive episode|dysthymi\w*|persistent depressive disorder|adjustment disorder[^.;]{0,60}depress\w*|clinical depression|postpartum depression|depression,? (?:unspecified|recurrent|mild|moderate|severe)|(?:recurrent|severe|moderate|mild|chronic) depression|F3[23]\.\w+)", re.I),
    "dep_assert": re.compile(r"\b(?:history of|hx of|h/o|diagnosed with|diagnosis of|dx of|dx:|treated for|being treated for|struggles with|suffers from|suffering from|has been (?:depressed|having depression)|endorses|reports|experiencing|positive for|\+)\s+(?:current\s+|active\s+|recurrent\s+|severe\s+|moderate\s+|mild\s+|chronic\s+|ongoing\s+|significant\s+|anxiety and\s+|anxiety/|some\s+)?depress(?:ion|ive symptoms?|ive disorder|ed mood)\b|\bdepression\s+(?:history|diagnosis|dx|and anxiety)\b|\b(?:feels?|feeling|felt)\s+(?:very\s+|really\s+|more\s+)?depressed\b", re.I),
    "dep_mention": re.compile(r"\bdepress(?:ion|ed|ive)\b", re.I),
    "suicidality": re.compile(r"\b(?:suicidal ideation|suicidal thoughts?|suicidal|suicide attempt|attempted suicide|suicide plan|(?:wants?|wanting|wish(?:es|ed)?) to (?:die|be dead|kill (?:him|her|them)self)|kill (?:him|her|them)self|end (?:his|her|their) life|(?:passive|active) SI\b|SI/HI|SI and HI|SI or HI|\bSI\b(?=[^A-Za-z]{0,3}(?:HI|ideation|plan)))", re.I),
    "self_harm": re.compile(r"\b(?:self[- ]harm\w*|self[- ]injur\w*|cutting (?:him|her|them)self|cuts? on (?:his|her|their) (?:arms?|wrists?|thighs?|legs?)|cutting behavio\w*|NSSI|self[- ]mutilat\w*|burn(?:s|ing) (?:him|her|them)self)", re.I),
    "sx_mood": re.compile(r"\b(?:sad(?:ness)?|tearful|crying (?:spells|episodes|a lot|more)|cries (?:easily|a lot)|hopeless\w*|worthless\w*|guilt(?:y)? feelings|low mood|down mood|feels? down|feeling down|irritab\w*|anger outbursts|mood swings|anhedonia|loss of interest|lost interest|no longer enjoys|not enjoying|lack of motivation|unmotivated|withdrawn|isolat(?:ing|ed|es) (?:him|her|them)self|social withdrawal|self[- ]esteem)\b", re.I),
    "sx_neuroveg": re.compile(r"\b(?:insomnia|trouble (?:falling|staying) asleep|difficulty (?:falling|staying) asleep|can't sleep|cannot sleep|not sleeping|sleeping (?:too much|more|less|poorly)|hypersomnia|poor sleep|sleep (?:problems?|difficult\w*|disturbance)|nightmares?|(?:decreased|poor|increased|loss of) appetite|not eating|overeating|weight (?:loss|gain)|fatigue\w*|low energy|tired all the time|always tired|(?:poor|decreased|difficulty) concentrat\w*|trouble focusing|can't focus|psychomotor)\b", re.I),
    "sx_somatic": re.compile(r"\b(?:headaches? (?:daily|every day|frequent\w*)|stomach ?aches?|abdominal pain (?:daily|frequent\w*|chronic)|chronic pain|somatic complaints?)\b", re.I),
    "stressor": re.compile(r"\b(?:bullied|bullying|bullies|divorce\w*|separat(?:ed|ion) (?:from|of) (?:his|her|their|the) (?:parents?|mother|father)|parents? (?:are|were|got) (?:separated|divorced)|DCF|foster care|foster home|CPS\b|child protective|(?:physical|sexual|emotional|verbal|substance) abuse|abused|history of abuse|neglected|neglect\b|molest\w*|sexual(?:ly)? assault\w*|rape\w*|domestic violence|(?:death|passing|loss) of (?:his|her|their|a) (?:mother|father|mom|dad|grandmother|grandfather|brother|sister|friend|parent)|died|passed away|grief|grieving|incarcerat\w*|jail|homeless\w*|shelter|changed schools|moved schools|new school|recently moved|breakup|broke up|boyfriend|girlfriend|relationship problems|peer (?:problems|conflict)|fights? with|argu(?:es|ing) with|conflict with (?:his|her|their) (?:mother|father|parents|mom|dad|family)|stress(?:ed|ors|ful)? (?:at|about|with|from)|financial (?:stress|problems|difficult\w*))\b", re.I),
    "school": re.compile(r"\b(?:failing (?:grades|classes|school)|grades (?:dropped|declining|falling|are (?:poor|bad|dropping))|drop(?:ped)? out|suspended|expelled|truan\w*|skipping school|missing school|school (?:refusal|avoidance|problems|difficulties)|IEP\b|504 plan|special education|ESE\b|repeat(?:ed|ing) (?:a |the )?grade|held back|learning disabilit\w*|struggling (?:in|at|with) school|behind in school|homeschool\w*|virtual school)\b", re.I),
    "treatment": re.compile(r"\b(?:counsel(?:ing|or)|therapist|psychotherapy|talk therapy|CBT\b|DBT\b|psychiatr\w*|psycholog\w*|behavioral health|mental health (?:referral|services|counselor|provider|treatment|therapy)|Baker Act\w*|inpatient psych\w*|psychiatric (?:hospitalization|admission|hospital|unit)|crisis (?:stabilization|unit)|referr(?:ed|al) to (?:psych\w*|counsel\w*|behavioral health|therapy)|therapy referral|on (?:an? )?(?:SSRI|antidepressant)|antidepressant\w*|SSRI\b)", re.I),
    "psych_med": re.compile(r"\b(?:sertraline|zoloft|fluoxetine|prozac|escitalopram|lexapro|citalopram|celexa|paroxetine|paxil|fluvoxamine|luvox|venlafaxine|effexor|desvenlafaxine|pristiq|duloxetine|cymbalta|bupropion|wellbutrin|mirtazapine|remeron|vilazodone|viibryd|vortioxetine|trintellix|trazodone|desyrel|amitriptyline|elavil|nortriptyline|imipramine|doxepin|buspirone|buspar|hydroxyzine|aripiprazole|abilify|risperidone|risperdal|quetiapine|seroquel|olanzapine|zyprexa|lamotrigine|lamictal|lithium|guanfacine|intuniv|clonidine|methylphenidate|ritalin|concerta|adderall|amphetamine|atomoxetine|strattera|melatonin)\b", re.I),
    "fam_mh": re.compile(r"\b(?:depress\w*|bipolar|schizophreni\w*|anxiety|suicid\w*|substance (?:use|abuse)|alcoholi\w*|drug (?:use|abuse)|mental (?:illness|health (?:issues|problems|history))|psychiatric (?:illness|history|problems))\b", re.I),
    "anxiety": re.compile(r"\b(?:anxiety|anxious|panic attacks?|GAD\b|generalized anxiety|social anxiety|worr(?:ies|ied|y) (?:a lot|excessively|constantly|about everything)|nervous(?:ness)?)\b", re.I),
}
_PHQ9 = re.compile(r"(?:PHQ[- ]?9(?:[- ]?(?:M|A|modified))?|Patient Health Questionnaire[- ]?9(?:[- ]?(?:M|A))?)[^0-9]{0,60}?(\d{1,2})(?:\s*/\s*27)?", re.I)
_PHQ9_SLASH = re.compile(r"PHQ[- ]?9\w*[^0-9]{0,40}?(\d{1,2})\s*/\s*27", re.I)
_PHQ2 = re.compile(r"(?:PHQ[- ]?2|Patient Health Questionnaire[- ]?2)[^0-9]{0,60}?(\d{1,2})(?:\s*/\s*6)?", re.I)
_GAD7 = re.compile(r"(?:GAD[- ]?7|Generalized Anxiety Disorder[- ]?7)[^0-9]{0,60}?(\d{1,2})(?:\s*/\s*21)?", re.I)
_HAS_DEP = re.compile(r"Does this patient have depression:?\s*(Yes|No)", re.I)
_DEP_SCREEN_RESULT = re.compile(r"depression screen(?:ing)?[^.]{0,40}?\b(positive|negative|abnormal|normal)\b", re.I)

CTX_BEFORE = 45  # chars of left context inspected for negation / history / family cues

_MDD_BAD_LEFT = re.compile(r"(?:provider|physician|attending|nurse|dr\.?|md,|resident|desk|since|forms? (?:on|to)|mouthwash|\bqty)\s*:?\s*$", re.I)
_MDD_BAD_RIGHT = re.compile(r"^\s*(?::\s*\d|of \d|desk|out\b|next|is reached|mL|mg)", re.I)
_MDD_GOOD_LEFT = re.compile(r"(?:dx|diagnos\w*|history|hx|h/o|with|carries|symptoms of|for|of|and|,|\(|:|-|\d\.)\s*$", re.I)


def mdd_is_depression(text: str, s: int, e: int) -> bool:
    """Bare 'MDD' is ambiguous (provider credential, maximum daily dose). Accept only with diagnosis-like context."""
    left = text[max(0, s - 40):s]; right = text[e:e + 20]
    if _MDD_BAD_LEFT.search(left) or _MDD_BAD_RIGHT.match(right):
        return False
    return bool(_MDD_GOOD_LEFT.search(left)) or bool(re.match(r"^\s*(?:\(|,|;|\.|with|and|recurrent|single|episode|moderate|severe|mild|unspecified|in|treatment)", right, re.I))


@dataclass
class Evidence:
    category: str
    status: str      # present | negated | history | family | uncertain | template
    span: str
    section: str
    pos: int


def _status(text: str, s: int, e: int, section: str) -> str:
    left = text[max(0, s - CTX_BEFORE):s]
    left_clause = re.split(r"[.;:]\s", left)[-1] if left else ""
    if section == "education_template":
        return "template"
    fam = list(_FAMILY.finditer(left_clause))
    if fam:
        between = left_clause[fam[-1].end():]
        # 'Mother reports patient has ...' -> the relative is the reporter, not the experiencer
        if not re.search(r"\b(?:patient|pt|child|son|daughter|he|she|his|her|him|reports?|reported|states?|stated|says?|said|notes?|noted|endorses|denies|concerned|brought|per)\b", between, re.I):
            return "family"
    elif _FAMILY.search(text[e:e + 25]) and re.search(r"\b(?:in|of)\b", text[e:e + 6]):
        return "family"
    if _NEG.search(left_clause) and not _BUT.search(left_clause[_NEG.search(left_clause).end():]):
        return "negated"
    if re.match(r"\s*[:=-]?\s*(?:no\b|none\b|denies|denied|negative|absent|nil\b|not present|0\b|n/a)", text[e:e + 16], re.I):
        return "negated"
    if _HISTORY.search(left_clause):
        return "history"
    if _UNCERTAIN.search(left_clause) or "?" in text[e:e + 3]:
        return "uncertain"
    return "present"


def extract_evidence(text: str, sections: list[Section] | None = None) -> tuple[list[Evidence], dict]:
    sections = sections or split_sections(text)
    sec_of = lambda pos: next((s.group for s in sections if s.start <= pos < s.end), "narrative")
    ev: list[Evidence] = []
    for cat, rx in EVIDENCE_PATTERNS.items():
        for m in rx.finditer(text):
            sec = sec_of(m.start())
            if cat in ("dep_dx", "dep_mention", "dep_assert") and _NONDEP.search(text[max(0, m.start() - 25):m.end() + 25]):
                continue
            if cat == "dep_dx" and re.fullmatch(r"MDD", m.group(0)) and not mdd_is_depression(text, m.start(), m.end()):
                continue
            if cat in ("dep_assert", "dep_mention", "sx_mood", "sx_neuroveg", "suicidality", "self_harm", "anxiety"):
                left = text[max(0, m.start() - 70):m.start()]; right = text[m.end():m.end() + 60]
                if _QUESTION.search(left) or re.match(r"[^.;]{0,60}\?", right):
                    ev.append(Evidence(cat, "template", text[max(0, m.start() - 30):m.end() + 30].strip(), sec, m.start()))
                    continue
            if cat == "fam_mh":
                # only meaningful inside family-history context
                if not (sec == "pmh_fh" and _FAMILY.search(text[max(0, m.start() - 80):m.start()])) and not re.search(r"family history[^.]{0,80}$", text[max(0, m.start() - 100):m.start()], re.I):
                    continue
                st = "family"
            elif cat == "psych_med":
                st = "negated" if _NEG.search(text[max(0, m.start() - 25):m.start()]) else ("history" if re.search(r"\b(?:discontinued|stopped|previously|prior|was on|used to)\b", text[max(0, m.start() - 40):m.start()], re.I) else "present")
                if sec == "meds":
                    st = "present"
            else:
                st = _status(text, m.start(), m.end(), sec)
                if cat == "dep_assert" and st == "present" and _HISTORY.match(m.group(0)):
                    st = "history"
            if sec in ("exam_vitals", "ros") and cat in ("sx_mood", "sx_neuroveg", "anxiety") and st == "negated":
                st = "template"
            ev.append(Evidence(cat, st, text[max(0, m.start() - 30):m.end() + 30].strip(), sec, m.start()))
    # screening instruments
    scores = {}
    for name, rx in [("phq9", _PHQ9), ("phq2", _PHQ2), ("gad7", _GAD7)]:
        vals = [int(m.group(1)) for m in rx.finditer(text)]
        vals = [v for v in vals if v <= {"phq9": 27, "phq2": 6, "gad7": 21}[name]]
        if vals:
            scores[name] = max(vals)
            ev.append(Evidence(f"{name}_score", "present", f"{name}={max(vals)}", "psych_mh", 0))
    m = _HAS_DEP.search(text)
    if m:
        scores["has_depression_field"] = m.group(1).lower()
        ev.append(Evidence("has_depression_field", "present" if m.group(1).lower() == "yes" else "negated", m.group(0), "psych_mh", m.start()))
    m = _DEP_SCREEN_RESULT.search(text)
    if m:
        scores["dep_screen_result"] = m.group(1).lower()
    n_templates = len(TEMPLATE_NORMAL.findall(text))
    feats = summarize_evidence(ev, scores, n_templates)
    return ev, feats


def summarize_evidence(ev: list[Evidence], scores: dict, n_templates: int) -> dict:
    f: dict = {"n_template_phrases": n_templates}
    for cat in list(EVIDENCE_PATTERNS) :
        for st in ("present", "negated", "history", "family", "uncertain", "template"):
            n = sum(1 for e in ev if e.category == cat and e.status == st)
            if n:
                f[f"{cat}__{st}"] = n
    f.update({f"score_{k}": v for k, v in scores.items()})
    f["dep_asserted"] = int(any(e.category in ("dep_dx", "dep_assert") and e.status in ("present", "history") for e in ev) or scores.get("has_depression_field") == "yes")
    f["dep_current_asserted"] = int(any(e.category in ("dep_dx", "dep_assert") and e.status == "present" for e in ev) or scores.get("has_depression_field") == "yes")
    f["si_present"] = int(any(e.category in ("suicidality", "self_harm") and e.status == "present" for e in ev))
    f["si_denied"] = int(any(e.category == "suicidality" and e.status in ("negated", "template") for e in ev))
    f["phq9_positive"] = int(scores.get("phq9", -1) >= 10)
    f["phq2_positive"] = int(scores.get("phq2", -1) >= 3)
    f["mh_relevance"] = sum(1 for e in ev if e.status == "present" and e.category in ("dep_dx", "dep_assert", "dep_mention", "suicidality", "self_harm", "sx_mood", "sx_neuroveg", "stressor", "school", "treatment", "anxiety"))
    return f


def process_note(raw: str) -> dict:
    """Full L1->L2->L3 pipeline for one note. Returns dict with clean text, condensed text, evidence, features."""
    clean = clean_text(raw)
    secs = split_sections(clean)
    condensed, l2info = condense(clean, secs)
    ev, feats = extract_evidence(clean, secs)
    return {"clean": clean, "condensed": condensed, "l2": l2info, "evidence": [asdict(e) for e in ev], "features": feats}
