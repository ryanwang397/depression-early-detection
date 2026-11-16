"""Shared helpers for the depression early-detection dataset pipeline.

Design rules:
  * source data is read-only; everything derived goes to config.paths.out_dir
  * every stage streams large CSVs with pyarrow (never loads a full table into pandas)
  * every stage writes a <stage>_summary.json next to its outputs
"""
from __future__ import annotations
import json, os, re, sys, time, hashlib, logging
from pathlib import Path
import yaml
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    od = Path(cfg["paths"]["out_dir"])
    if not od.is_absolute():   # keep the repository relocatable: relative out_dir = relative to the repo root, not the CWD
        cfg["paths"]["out_dir"] = str(ROOT / od)
    return cfg


def refuse_archived(cfg: dict, sub: str) -> None:
    """Stages that WRITE outputs/<sub> must never run while that directory is a link to the archived intermediates
    (FROM_SOURCE=0 links outputs/index and outputs/notes to precomputed intermediates; writing through the link would alter them)."""
    p = Path(cfg["paths"]["out_dir"]) / sub
    if p.is_symlink():
        sys.exit(f"refusing to write: {p} is a link to archived outputs ({os.path.realpath(p)}). "
                 f"Remove the link (rm {p}) to rebuild this stage from the source data, or keep the link and skip this stage.")


def out_dir(cfg: dict, *sub: str) -> Path:
    p = Path(cfg["paths"]["out_dir"]).joinpath(*sub)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        lg.addHandler(h)
    lg.setLevel(logging.INFO)
    return lg


def write_summary(path: Path, payload: dict) -> None:
    payload = dict(payload)
    payload.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


# ----------------------------------------------------------------------------- CSV streaming
def stream_csv(path: str | Path, columns: list[str], column_types: dict | None = None,
               block_size: int = 64 << 20, limit_rows: int | None = None):
    """Yield pyarrow RecordBatches with only `columns`. All columns default to string
    unless overridden in column_types (pyarrow types). Handles quoted multi-line fields."""
    column_types = column_types or {}
    types = {c: column_types.get(c, pa.string()) for c in columns}
    reader = pacsv.open_csv(
        str(path),
        read_options=pacsv.ReadOptions(block_size=block_size, use_threads=True),
        parse_options=pacsv.ParseOptions(newlines_in_values=True),
        convert_options=pacsv.ConvertOptions(include_columns=columns, column_types=types,
                                             strings_can_be_null=True, null_values=["", "NULL", "null", "?"]),
    )
    n = 0
    for batch in reader:
        if limit_rows is not None and n >= limit_rows:
            break
        if limit_rows is not None and n + batch.num_rows > limit_rows:
            batch = batch.slice(0, limit_rows - n)
        n += batch.num_rows
        yield batch


class ParquetSink:
    """Append record batches / tables to one parquet file with a fixed schema."""

    def __init__(self, path: str | Path, schema: pa.Schema | None = None, compression: str = "zstd"):
        self.path = Path(path)
        self.schema = schema
        self.compression = compression
        self._w = None
        self.rows = 0

    def write(self, tbl: pa.Table | pa.RecordBatch) -> None:
        if isinstance(tbl, pa.RecordBatch):
            tbl = pa.Table.from_batches([tbl])
        if tbl.num_rows == 0:
            return
        if self._w is None:
            self.schema = self.schema or tbl.schema
            self._w = pq.ParquetWriter(str(self.path), self.schema, compression=self.compression)
        if not tbl.schema.equals(self.schema):
            tbl = tbl.select(self.schema.names).cast(self.schema)
        self._w.write_table(tbl)
        self.rows += tbl.num_rows

    def close(self) -> None:
        if self._w is not None:
            self._w.close()
            self._w = None


# ----------------------------------------------------------------------------- column conversions
def to_int64(arr: pa.Array | pa.ChunkedArray) -> pa.Array:
    """'1234567890.0' / '1234567890' / null -> int64 (via float64; ids < 2^53)."""
    if pa.types.is_string(arr.type):
        arr = pc.cast(arr, pa.float64(), safe=False)
    if pa.types.is_floating(arr.type):
        arr = pc.cast(pc.round(arr), pa.int64(), safe=False)
    return arr


def to_date32(arr: pa.Array | pa.ChunkedArray) -> pa.Array:
    """'2018-04-18' or '2018-04-18 09:48:00' -> date32; unparseable -> null."""
    if pa.types.is_date32(arr.type):
        return arr
    if pa.types.is_timestamp(arr.type):
        return pc.cast(arr, pa.date32())
    s = pc.utf8_slice_codeunits(arr, 0, 10)
    ok = pc.match_substring_regex(s, r"^\d{4}-\d{2}-\d{2}$")
    s = pc.if_else(ok, s, pa.scalar(None, pa.string()))
    ts = pc.strptime(s, format="%Y-%m-%d", unit="s", error_is_null=True)
    return pc.cast(ts, pa.date32())


def norm_str(arr: pa.Array | pa.ChunkedArray) -> pa.Array:
    return pc.utf8_trim_whitespace(arr)


def stable_hash(seed: int, *parts) -> int:
    payload = "|".join([str(seed), *map(str, parts)]).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def icd_prefix_match(values: pa.Array | pa.ChunkedArray, prefixes: list[str]) -> pa.Array:
    """Boolean mask: condition_source_value starts with any prefix (exact prefix, dot kept)."""
    if not prefixes:
        return pa.array([False] * len(values))
    pat = "^(" + "|".join(re.escape(p) for p in prefixes) + ")"
    return pc.fill_null(pc.match_substring_regex(values, pat), False)


# ----------------------------------------------------------------------------- note text random access
import csv as _csv, io as _io

_NOTE_FILES: list[Path] | None = None


def note_file_list(cfg) -> list[Path]:
    global _NOTE_FILES
    if _NOTE_FILES is None:
        _NOTE_FILES = sorted(Path(cfg["paths"]["notes_dir"]).glob(cfg["paths"].get("notes_glob", "*_CLINICAL_NOTE_*.csv")),
                             key=lambda p: int(re.search(r"_(\d+)\.csv$", p.name).group(1)))
    return _NOTE_FILES


def parse_note_record(raw: bytes) -> tuple[int, str]:
    text = raw.decode("utf-8", errors="replace")
    _csv.field_size_limit(1 << 31)
    row = next(_csv.reader(_io.StringIO(text)))
    return int(row[0]), row[1]


def fetch_note_texts(cfg, requests: list[tuple[int, int, int, int]]) -> dict[int, str]:
    """requests: (linkage_note_id, file_idx, offset, length). Returns {linkage_note_id: text}.
    Reads are sorted by (file, offset) so each file is opened once and read forward."""
    files = note_file_list(cfg)
    out: dict[int, str] = {}
    for fi in sorted({r[1] for r in requests}):
        reqs = sorted((r for r in requests if r[1] == fi), key=lambda r: r[2])
        with open(files[fi], "rb", buffering=4 << 20) as f:
            for nid, _, off, ln in reqs:
                f.seek(off)
                rid, txt = parse_note_record(f.read(ln))
                if rid != nid:
                    raise ValueError(f"offset index mismatch: expected {nid} got {rid} in {files[fi]} @ {off}")
                out[nid] = txt
    return out
