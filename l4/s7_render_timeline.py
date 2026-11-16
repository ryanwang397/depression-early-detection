#!/usr/bin/env python3
"""Step 7: render each (patient, lead) record as a compact plain-text EVIDENCE TIMELINE for LLM-only predictors.

  python s7_render_timeline.py --lead 12 --splits train,dev,test [--limit N] [--stats-only]
Output: outputs/l4/timeline/lead{L}_{split}.jsonl  {person_id, label, lead_months, cutoff_date, sex, age, n_notes, text, n_chars}

Sections (all derived from the dataset JSONL + the LLM ledger, nothing post-cutoff):
  PATIENT / UTILIZATION            demographics, record span, visit counts by window, ED, specialties, BMI trajectory
  DIAGNOSES / MEDICATIONS          structured codes grouped by name with first date and recurrence count (capped)
  EVIDENCE FROM NOTES              grounded ledger items (status != template), grouped by date, MH categories first; older items summarised
  RECENT NOTE SUMMARIES            one-sentence LLM summaries of the most recent relevant notes
  MOST RECENT RELEVANT NOTE        excerpt of the L2 text of the latest relevant note
Budget: ~12k characters (~3k tokens) per patient; caps below keep p95 under ~16k characters."""
from __future__ import annotations
import argparse, json, re, time
from collections import defaultdict, Counter
import numpy as np, pandas as pd, pyarrow.parquet as pq
from l4_common import OUT, DATASET, get_logger, write_summary

log = get_logger("s7")
TL_DIR = OUT / "timeline"
MH_CATS = ["depression_dx_statement", "suicidality", "self_harm", "screening", "dep_symptom", "anxiety", "psych_med", "treatment", "stressor", "family_mh_history", "other_psych_dx", "eating_body_image"]
OTHER_CATS = ["school_behavior", "substance", "somatic", "protective", "repro_derm"]
MAX_DX = 40; MAX_RX = 25; MAX_EVIDENCE = 70; MAX_SUMMARIES = 8; EXCERPT_CHARS = 1500; SPAN_CHARS = 90; RECENT_YEARS = 2.0; N_EXCERPTS = 1


def fmt_days(d):  # days before cutoff -> compact relative time
    return f"{d}d ago" if d < 60 else (f"{d//30}mo ago" if d < 730 else f"{d/365:.1f}y ago")


def header(r):
    enc = r["encounters"]; real = [e for e in enc if e["visit_type"] not in ("none", "telephone", "unknown")]
    def nvis(lo, hi): return sum(1 for e in real if lo <= e["days_before_cutoff"] < hi)
    ed365 = sum(1 for e in real if e["days_before_cutoff"] <= 365 and "ed" in e["visit_type"]); inp365 = sum(1 for e in real if e["days_before_cutoff"] <= 365 and "inpatient" in e["visit_type"])
    specs = Counter(n["specialty"] for e in enc if e["days_before_cutoff"] <= 365 for n in e["notes"] if n.get("specialty"))
    last = min((e["days_before_cutoff"] for e in real), default=r["history_days"])
    bmi = sorted([(e["days_before_cutoff"], m["value"]) for e in enc for m in e["measurements"] if m["name"] == "BMI" and m.get("value") is not None])
    bmi_s = ""
    if bmi:
        older = [v for d, v in bmi if 365 <= d <= 730]; bmi_s = f" BMI {bmi[0][1]:.1f}" + (f" (1y change {bmi[0][1]-older[0]:+.1f})" if older else "")
    n_notes = sum(len(e["notes"]) for e in enc)
    L = [f"PATIENT: {r['sex'].lower() if r.get('sex') else 'unknown sex'}, age {r['age_at_cutoff']:.0f} at prediction time. Record spans {r['history_days']/365:.1f} years with {len(enc)} encounter-days and {n_notes} notes.",
         f"UTILIZATION: in-person/telehealth visits in last 90/180/365 days: {nvis(0,91)}/{nvis(0,181)}/{nvis(0,366)} (previous year {nvis(366,731)}); ED visits last year {ed365}; inpatient last year {inp365}; "
         f"days since last visit {last}; specialties seen last year: {', '.join(k for k, _ in specs.most_common(5)) or 'none recorded'}.{bmi_s}"]
    return L, n_notes


def structured(r):
    first = {}; cnt = Counter(); rx_first = {}
    for e in r["encounters"]:
        d = e["days_before_cutoff"]
        for c in e["conditions"]:
            nm = (c.get("name") or c.get("icd") or "")[:60]; cnt[nm] += 1
            if nm not in first or d < first[nm][0]: pass
            first.setdefault(nm, (d, c.get("icd", "")))
            if d < first[nm][0]: first[nm] = (d, c.get("icd", ""))
        for m in e["medications"]:
            nm = (m.get("name") or "").split()[0].lower() if m.get("name") else ""
            if nm and (nm not in rx_first or d < rx_first[nm][0]): rx_first[nm] = (d, m.get("class"))
    # order: most recent first-occurrence... show chronological (oldest first) but cap by recency
    items = sorted(first.items(), key=lambda kv: kv[1][0])            # by days_before (most recent first)
    keep = items[:MAX_DX]; dropped = len(items) - len(keep)
    lines = ["DIAGNOSES (structured; most recent first; xN = recorded N times):"] if items else ["DIAGNOSES: none recorded."]
    lines += [f"  {fmt_days(d):>9}: {nm}{' x'+str(cnt[nm]) if cnt[nm] > 1 else ''}" for nm, (d, icd) in keep]
    if dropped: lines.append(f"  (+{dropped} older diagnoses omitted)")
    rx = sorted(rx_first.items(), key=lambda kv: kv[1][0])[:MAX_RX]
    if rx: lines.append("MEDICATIONS (first recorded): " + "; ".join(f"{nm} ({fmt_days(d)}{', '+cls if cls else ''})" for nm, (d, cls) in rx))
    return lines


def evidence(g: pd.DataFrame, cutoff):
    """g: this person's grounded, non-template ledger rows with note_date <= cutoff."""
    if g is None or len(g) == 0:
        return ["EVIDENCE FROM NOTES: no mental-health, psychosocial, sleep, somatic or substance evidence extracted."]
    g = g.assign(days=(cutoff - g.note_date.values.astype("datetime64[D]")).astype("timedelta64[D]").astype(int))
    g = g.drop_duplicates(["note_date", "type", "status", "span"]).sort_values(["days"])
    recent = g[g.days <= RECENT_YEARS * 365]; old = g[g.days > RECENT_YEARS * 365]
    pri = recent[recent.category.isin(MH_CATS)]; sec = recent[~recent.category.isin(MH_CATS)]
    keep = pd.concat([pri, sec]).sort_values("days").head(MAX_EVIDENCE) if len(pri) + len(sec) > MAX_EVIDENCE else recent
    if len(keep) > MAX_EVIDENCE: keep = pd.concat([pri.head(MAX_EVIDENCE), sec.head(max(0, MAX_EVIDENCE - len(pri)))]).sort_values("days")
    lines = ["EVIDENCE FROM NOTES (LLM-extracted, verbatim quotes; [status]; most recent first):"]
    for d, h in keep.groupby("days", sort=True):
        parts = [f"{it.type} [{it.status}{'' if it.experiencer == 'patient' else '/'+it.experiencer}] \"{it.span[:SPAN_CHARS]}\"" for it in h.itertuples()]
        lines.append(f"  {fmt_days(int(d)):>9}: " + "; ".join(parts))
    n_omit = len(recent) - len(keep)
    if n_omit > 0: lines.append(f"  (+{n_omit} further recent items omitted)")
    if len(old):
        yrs = old.assign(y=(old.days // 365)).groupby(["y", "category", "status"]).size()
        summ = defaultdict(list)
        for (y, cat, st), n in yrs.items(): summ[y].append(f"{cat} {st} x{n}")
        lines.append("  Older evidence (counts): " + " | ".join(f"{y}-{y+1}y ago: " + ", ".join(v[:10]) for y, v in sorted(summ.items())))
    return lines


def note_summaries(r, ns_person: pd.DataFrame | None, cutoff):
    lines = []
    if ns_person is not None and len(ns_person):
        rel = ns_person[(ns_person.note_relevant == True) & ns_person.summary.notna()].sort_values("note_date", ascending=False).head(MAX_SUMMARIES)
        if len(rel):
            lines.append("RECENT NOTE SUMMARIES (LLM one-line summaries of the latest relevant notes):")
            for it in rel.itertuples():
                d = (cutoff - np.datetime64(it.note_date, "D")).astype(int)
                lines.append(f"  {fmt_days(int(d)):>9} {it.note_type or ''}: {it.summary.strip()[:220]}")
    # excerpts of the N most recent relevant notes (from the JSONL text); N_EXCERPTS=1 reproduces the original single excerpt
    rel_ids = set(ns_person[ns_person.note_relevant == True].note_id) if ns_person is not None and len(ns_person) else set()
    found = []; seen = set()
    for e in reversed(r["encounters"]):
        for n in e["notes"]:
            if (n["note_id"] in rel_ids or (not rel_ids and n["relevance"] > 0)) and n["note_id"] not in seen:
                found.append((e["days_before_cutoff"], n)); seen.add(n["note_id"])
                if len(found) >= N_EXCERPTS: break
        if len(found) >= N_EXCERPTS: break
    for k, (d, n) in enumerate(found):
        txt = re.sub(r"\s+", " ", n["text"])[:EXCERPT_CHARS]
        label = "MOST RECENT RELEVANT NOTE" if k == 0 else f"EARLIER RELEVANT NOTE {k+1}"
        lines.append(f"{label} ({fmt_days(d)}, {n['type']}{' / '+n['specialty'] if n.get('specialty') else ''}), excerpt: \"{txt}\"")
    return lines


def render(r, g, ns_person):
    cutoff = np.datetime64(r["cutoff_date"], "D")
    H, n_notes = header(r)
    parts = H + structured(r) + evidence(g, cutoff) + note_summaries(r, ns_person, cutoff)
    return "\n".join(parts), n_notes


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lead", type=int, default=12); ap.add_argument("--splits", default="train,dev,test"); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-excerpts", type=int, default=1); ap.add_argument("--excerpt-chars", type=int, default=1500); ap.add_argument("--suffix", default="", help="output file suffix, e.g. _x3 -> lead{L}_{split}_x3.jsonl"); a = ap.parse_args()
    global N_EXCERPTS, EXCERPT_CHARS; N_EXCERPTS = a.n_excerpts; EXCERPT_CHARS = a.excerpt_chars
    t0 = time.time(); TL_DIR.mkdir(parents=True, exist_ok=True)
    led = pd.read_parquet(OUT / "ledger.parquet", columns=["person_id", "note_date", "category", "type", "status", "experiencer", "span", "grounding"])
    led = led[(led.grounding != "none") & (led.status != "template")].drop(columns="grounding"); led["note_date"] = pd.to_datetime(led.note_date)
    ns = pd.read_parquet(OUT / "note_summary.parquet", columns=["person_id", "note_id", "note_date", "note_relevant", "summary"]); ns["note_date"] = pd.to_datetime(ns.note_date)
    qi = pd.read_parquet(OUT / "note_queue_index.parquet", columns=["note_id", "note_type"]); ns = ns.merge(qi, on="note_id", how="left")
    led_by = {k: v for k, v in led.groupby("person_id")}; ns_by = {k: v for k, v in ns.groupby("person_id")}
    log.info(f"ledger {len(led)} rows / {len(led_by)} persons; {time.time()-t0:.0f}s")
    stats = {}
    for split in a.splits.split(","):
        out = TL_DIR / f"lead{a.lead}_{split}{a.suffix}.jsonl"; k = 0; lens = []
        with open(DATASET / f"lead_{a.lead}mo" / f"{split}.jsonl") as fh, open(out, "w") as fo:
            for line in fh:
                r = json.loads(line); pid = r["person_id"]; cutoff = pd.Timestamp(r["cutoff_date"])
                g = led_by.get(pid); g = g[g.note_date <= cutoff] if g is not None else None
                n = ns_by.get(pid); n = n[n.note_date <= cutoff] if n is not None else None
                text, n_notes = render(r, g, n)
                fo.write(json.dumps({"person_id": pid, "label": r["label"], "lead_months": a.lead, "cutoff_date": r["cutoff_date"], "sex": r.get("sex"), "age": r["age_at_cutoff"], "n_notes": n_notes, "n_chars": len(text), "text": text}, ensure_ascii=True) + "\n")
                lens.append(len(text)); k += 1
                if a.limit and k >= a.limit: break
        q = np.percentile(lens, [50, 90, 95, 99, 100]); stats[split] = {"n": k, "chars_p50": q[0], "chars_p90": q[1], "chars_p95": q[2], "chars_p99": q[3], "chars_max": q[4]}
        log.info(f"lead {a.lead} {split}: {k} rows; chars p50/p90/p95/p99/max = {q.astype(int).tolist()}; {time.time()-t0:.0f}s")
    write_summary(TL_DIR / f"s7_lead{a.lead}{a.suffix}_summary.json", {"lead": a.lead, "splits": stats, "caps": {"MAX_DX": MAX_DX, "MAX_RX": MAX_RX, "MAX_EVIDENCE": MAX_EVIDENCE, "MAX_SUMMARIES": MAX_SUMMARIES, "EXCERPT_CHARS": EXCERPT_CHARS, "N_EXCERPTS": N_EXCERPTS}, "seconds": round(time.time() - t0)})


if __name__ == "__main__":
    main()
