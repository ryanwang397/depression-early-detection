"""Results table (lead 3): AUROC / AP / F1 / precision / recall per method on the test split.

Threshold: the F1-optimal threshold on the DEV predictions of the same run (exp_common.best_f1_threshold), applied to TEST.

  python evaluation/make_table.py                # runs under results/
  python evaluation/make_table.py --boot 1000    # plus 95% CIs from patient-level bootstrap resamples of the test split
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "experiments"))
from exp_common import metrics, best_f1_threshold, f1_at_threshold  # noqa: E402

METHODS = [  # (row label, run directory under results/)
    ("Clinical ModernBERT SFT", "lead3_mbert_modernbert_full_notes_ehr_L8192"),
    ("Llama-3.1-8B zero-shot (XML)", "lead3_zsxml_llama-3.1-8b_zeroshot"),
    ("Llama-3.1-8B LoRA-SFT (XML, two-stage)", "lead3_xml2_llama-3.1-8b_lora"),
    ("Ours: evidence timeline + self-distilled reasoning + case-aware expert", "lead3_ours_final_llama-3.1-8b"),
]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--boot", type=int, default=0, help="bootstrap replicates for 95%% CIs (0 = off)")
    ap.add_argument("--out", default=str(ROOT / "results" / "table_lead3.csv")); a = ap.parse_args()
    rows = []
    for label, run in METHODS:
        d = Path(a.results) / run
        if not (d / "predictions_test.csv").exists(): print(f"[missing] {label}: {d}"); continue
        dv = pd.read_csv(d / "predictions_dev.csv"); te = pd.read_csv(d / "predictions_test.csv"); thr = best_f1_threshold(dv.label.values, dv.prob.values)
        m = metrics(te.label.values, te.prob.values, n_boot=a.boot); f = f1_at_threshold(te.label.values, te.prob.values, thr)
        rows.append({"method": label, "n": len(te), "pos": int(te.label.sum()), "AUROC": round(m["auroc"], 4), "AP": round(m["auprc"], 4), "F1": round(f["f1"], 4),
                     "Pre": round(f["precision"], 3), "Recall": round(f["recall"], 3), "dev_threshold": round(float(thr), 4),
                     **({"AUROC_ci95": m["auroc_ci95"], "AP_ci95": m["auprc_ci95"]} if a.boot else {})})
    if not rows: return
    df = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
    print(df.to_string(index=False)); Path(a.out).parent.mkdir(parents=True, exist_ok=True); df.to_csv(a.out, index=False); print(f"\nwritten {a.out}")


if __name__ == "__main__":
    main()
