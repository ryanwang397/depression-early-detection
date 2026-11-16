#!/usr/bin/env python3
"""Step 3: per-note guided-JSON extraction with MedGemma-27B-text-it via vLLM (resumable, shardable).

  python s2_extract_vllm.py --tier 1 --shard 0 --n-shards 4 [--limit N] [--out-tag full]
  python s2_extract_vllm.py --queue-parquet outputs/l4/pilot/sample.parquet --out-tag pilot   (explicit note list)

Output: outputs/l4/extractions/<out_tag>_tier<T>_shard<i>of<n>.jsonl, one line per note:
  {note_id, person_id, note_date, tier, raw: <parsed json or null>, error, prompt_tokens, out_tokens, finish}
Resume: note_ids already present in the output file are skipped."""
from __future__ import annotations
import sys
import argparse, json, os, re, time
from pathlib import Path
import pandas as pd, pyarrow.parquet as pq
from l4_common import QUEUE_DIR, EXTRACT_DIR, OUT, model_dir, note_shard, load_done_ids, get_logger, write_summary
from schema import JSON_SCHEMA, SCHEMA_VERSION
from prompt import build_messages, PROMPT_VERSION

log = get_logger("s2")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def salvage_json(txt: str) -> dict | None:
    """Recover complete items from a length-truncated guided-JSON output: keep every balanced {...} object inside
    "items": [...] and mark the record as truncated (note_relevant/summary unknown)."""
    i = txt.find('"items"')
    if i < 0:
        return None
    items = []; depth = 0; start = None; in_str = False; esc = False
    for j in range(txt.find("[", i) + 1, len(txt)):
        ch = txt[j]
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"': in_str = True
        elif ch == "{":
            if depth == 0: start = j
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try: items.append(json.loads(txt[start:j + 1]))
                except Exception: pass
                start = None
        elif ch == "]" and depth == 0:
            break
    m = re.search(r'"note_relevant"\s*:\s*(true|false)', txt)
    return {"note_relevant": (m.group(1) == "true") if m else None, "items": items, "summary": None, "truncated": True} if items else None


def load_queue(a) -> pd.DataFrame:
    cols = ["note_id", "person_id", "note_date", "note_type", "author_type", "author_specialty", "tier", "text", "mh_relevance"]
    if a.queue_parquet:
        df = pd.read_parquet(a.queue_parquet); df = df[[c for c in cols if c in df.columns]]
    else:
        frames = []
        for p in sorted(QUEUE_DIR.glob("part_*.parquet")):
            t = pq.read_table(p, columns=cols, filters=[("tier", "=", a.tier)] if a.tier else None).to_pandas()
            if a.n_shards > 1:
                t = t[[note_shard(n, a.n_shards) == a.shard for n in t.note_id]]
            frames.append(t)
        df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["person_id", "note_date"]).reset_index(drop=True)
    if a.limit:
        df = df.head(a.limit)
    return df


class Extractor:
    def __init__(self, a):
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        self.md = model_dir(); log.info(f"loading {self.md}")
        kw = {"guided_decoding_backend": a.guided_backend} if a.guided_backend else {}
        if a.fp8: kw["quantization"] = "fp8"
        self.llm = LLM(model=self.md, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
                       tensor_parallel_size=1, enable_prefix_caching=True, seed=0, max_num_seqs=a.max_num_seqs, **kw)
        self.tok = self.llm.get_tokenizer()
        self.sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, guided_decoding=None if a.no_guided else GuidedDecodingParams(json=JSON_SCHEMA))
        self.text_budget = a.max_model_len - a.max_tokens - a.prompt_overhead_tokens
        log.info(f"model loaded; text budget {self.text_budget} tokens")

    def prompt(self, row) -> tuple[str, int]:
        ids = self.tok.encode(row.text, add_special_tokens=False)
        text = row.text if len(ids) <= self.text_budget else self.tok.decode(ids[: self.text_budget]) + " [TRUNCATED]"
        msgs = build_messages(row.note_type, row.author_specialty, row.author_type, text)
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), len(ids)

    def run_batch(self, rows: list) -> list[dict]:
        prompts = [self.prompt(r) for r in rows]
        outs = self.llm.generate([p for p, _ in prompts], self.sp, use_tqdm=False)
        res = []
        for r, (_, ntok), o in zip(rows, prompts, outs):
            txt = o.outputs[0].text; err = None; raw = None
            try:
                raw = json.loads(txt)
            except Exception as e:
                raw = salvage_json(txt); err = f"json:{type(e).__name__}" + (":salvaged" if raw else "")
            res.append({"note_id": int(r.note_id), "person_id": int(r.person_id), "note_date": str(r.note_date)[:10], "tier": int(r.tier),
                        "raw": raw, "error": err, "text_tokens": int(ntok), "prompt_tokens": len(o.prompt_token_ids), "out_tokens": len(o.outputs[0].token_ids),
                        "finish": o.outputs[0].finish_reason})
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=1); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--queue-parquet", default=None); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--out-tag", default="full")
    ap.add_argument("--out", default=None); ap.add_argument("--batch", type=int, default=512); ap.add_argument("--max-model-len", type=int, default=5632)
    ap.add_argument("--max-tokens", type=int, default=1500); ap.add_argument("--prompt-overhead-tokens", type=int, default=1900)
    ap.add_argument("--gpu-mem", type=float, default=0.85); ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--fp8", action="store_true", help="online fp8 weight quantization")
    ap.add_argument("--no-guided", action="store_true", help="A/B: plain JSON generation (prompt-only) instead of xgrammar-guided decoding")
    ap.add_argument("--guided-backend", default=None, help="vLLM guided_decoding_backend override (e.g. xgrammar, guidance, outlines)")
    a = ap.parse_args()
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(a.out) if a.out else EXTRACT_DIR / f"{a.out_tag}_tier{a.tier}_shard{a.shard}of{a.n_shards}.jsonl"
    linked = [str(p) for p in EXTRACT_DIR.glob("*.jsonl") if p.is_symlink()]
    if linked: sys.exit(f"refusing to extract: {len(linked)} extraction files in {EXTRACT_DIR} are links to archived outputs (resume mode appends to them). Remove the links to re-extract from scratch.")
    df = load_queue(a); done = load_done_ids(sorted(EXTRACT_DIR.glob(f"{a.out_tag}_tier{a.tier}_shard*.jsonl")) or [out]); todo = df[~df.note_id.isin(done)]   # resume across all shard files of this tag (nested re-sharding safe)
    log.info(f"queue {len(df)} notes, done {len(done)}, todo {len(todo)} -> {out}")
    if len(todo) == 0:
        return
    ex = Extractor(a); t0 = time.time(); n = 0; ptok = 0; otok = 0; nerr = 0
    with open(out, "a") as f:
        for i in range(0, len(todo), a.batch):
            rows = list(todo.iloc[i:i + a.batch].itertuples(index=False))
            res = ex.run_batch(rows)
            for r in res:
                f.write(json.dumps(r, ensure_ascii=True) + "\n"); n += 1; ptok += r["prompt_tokens"]; otok += r["out_tokens"]; nerr += r["error"] is not None
            f.flush(); el = time.time() - t0
            log.info(f"{n}/{len(todo)} notes | {n/el:.2f} notes/s | prompt {ptok/el:.0f} tok/s out {otok/el:.0f} tok/s | errors {nerr} | eta {(len(todo)-n)/(n/el)/3600:.1f} h")
    write_summary(out.with_suffix(".summary.json"), {"out": str(out), "n": n, "errors": nerr, "seconds": round(time.time() - t0), "prompt_tokens": ptok, "out_tokens": otok,
                                                     "schema_version": SCHEMA_VERSION, "prompt_version": PROMPT_VERSION, "model": ex.md, "args": vars(a)})


if __name__ == "__main__":
    main()
