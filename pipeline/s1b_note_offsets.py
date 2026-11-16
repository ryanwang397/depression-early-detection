#!/usr/bin/env python3
"""Stage 1b: build a byte-offset index over the 45 clinical-note CSV files.

Each record in the note CSVs starts at a line beginning with  "<digits>","  .
Records may span several physical lines (quoted field with embedded newlines); a
continuation line never starts with that pattern. We record for every note:
  linkage_note_id, file_idx, offset (bytes), length (bytes), n_lines
Random access later: seek(offset), read(length), parse with the csv module.

Usage:
  python s1b_note_offsets.py --file-idx 0 [--limit-records N]      # one file -> outputs/index/note_offsets/part_00.parquet
  python s1b_note_offsets.py --merge                                # concatenate parts -> outputs/index/note_offsets.parquet
"""
from __future__ import annotations
import argparse, re, time, glob
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq
from common import refuse_archived, load_config, out_dir, get_logger, write_summary

log = get_logger("s1b")
START = re.compile(rb'^"(\d+)","')


def note_files(cfg) -> list[Path]:
    files = sorted(Path(cfg["paths"]["notes_dir"]).glob(cfg["paths"].get("notes_glob", "*_CLINICAL_NOTE_*.csv")),
                   key=lambda p: int(re.search(r"_(\d+)\.csv$", p.name).group(1)))
    return files


def scan_file(path: Path, file_idx: int, limit: int | None = None):
    ids, offs, lens, nls = [], [], [], []
    with open(path, "rb", buffering=16 << 20) as f:
        header = f.readline()
        assert header.startswith(b'"DEID_LINKAGE_NOTE_ID"'), header[:80]
        pos = len(header)
        cur_id = None; cur_off = 0; cur_len = 0; cur_lines = 0
        for line in f:
            m = START.match(line)
            if m:
                if cur_id is not None:
                    ids.append(cur_id); offs.append(cur_off); lens.append(cur_len); nls.append(cur_lines)
                    if limit and len(ids) >= limit:
                        cur_id = None
                        break
                cur_id = int(m.group(1)); cur_off = pos; cur_len = 0; cur_lines = 0
            cur_len += len(line); cur_lines += 1
            pos += len(line)
        if cur_id is not None:
            ids.append(cur_id); offs.append(cur_off); lens.append(cur_len); nls.append(cur_lines)
    return pa.table({"linkage_note_id": pa.array(ids, pa.int64()), "file_idx": pa.array([file_idx] * len(ids), pa.int16()),
                     "offset": pa.array(offs, pa.int64()), "length": pa.array(lens, pa.int64()), "n_lines": pa.array(nls, pa.int32())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file-idx", type=int, default=None, help="index into sorted note file list (0-based)")
    ap.add_argument("--limit-records", type=int, default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out-subdir", default="index")
    args = ap.parse_args()
    cfg = load_config(); refuse_archived(cfg, args.out_subdir)
    odir = out_dir(cfg, args.out_subdir, "note_offsets")
    files = note_files(cfg)
    if args.merge:
        parts = sorted(glob.glob(str(odir / "part_*.parquet")))
        tbl = pa.concat_tables([pq.read_table(p) for p in parts])
        out = odir.parent / "note_offsets.parquet"
        pq.write_table(tbl, out, compression="zstd")
        n_unique = pa.compute.count_distinct(tbl.column("linkage_note_id")).as_py()
        info = {"parts": len(parts), "rows": tbl.num_rows, "unique_ids": n_unique, "files": [str(f) for f in files],
                "multi_line_records": pa.compute.sum(pa.compute.greater(tbl.column("n_lines"), 1)).as_py()}
        write_summary(odir.parent / "s1b_note_offsets_summary.json", info); log.info(info)
        return
    path = files[args.file_idx]
    t0 = time.time()
    tbl = scan_file(path, args.file_idx, args.limit_records)
    pq.write_table(tbl, odir / f"part_{args.file_idx:02d}.parquet", compression="zstd")
    log.info({"file": str(path), "file_idx": args.file_idx, "records": tbl.num_rows, "seconds": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
