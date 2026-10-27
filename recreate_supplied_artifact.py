#!/usr/bin/env python3
"""Recreate the supplied label-inverted PhiUSIIL artifact from the canonical UCI CSV.

Verified transformation:
- drop the canonical-only FILENAME column;
- invert the binary label (new label = 1 - canonical label);
- retain row order and all other fields.

For the UCI release checked on 27 October 2026, this deterministic operation
reproduces the supplied CSV byte-for-byte. CSV formatting can differ with other
pandas versions, so SHA-256 equality is checked when --expected-supplied is given.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", required=True, help="Official UCI PhiUSIIL CSV")
    ap.add_argument("--out", required=True, help="Path for recreated supplied artifact")
    ap.add_argument("--expected-supplied", help="Optional path to existing supplied CSV for byte/hash comparison")
    args = ap.parse_args()
    source, out = Path(args.canonical), Path(args.out)
    df = pd.read_csv(source)
    url_col = next((c for c in df.columns if c.lower() == "url"), None)
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if url_col is None or label_col is None:
        raise ValueError("Canonical file must contain URL and label columns.")
    labels = pd.to_numeric(df[label_col], errors="raise").astype(int)
    if set(labels.unique()) != {0, 1}:
        raise ValueError(f"Expected binary labels {{0,1}}, found {sorted(labels.unique())}")
    if len(df) != 235795:
        raise ValueError(f"Unexpected canonical row count {len(df)}; inspect source release.")
    counts = labels.value_counts().to_dict()
    if counts != {1: 134850, 0: 100945}:
        raise ValueError(f"Unexpected canonical label counts: {counts}")
    df[label_col] = 1 - labels
    if "FILENAME" in df.columns:
        df = df.drop(columns=["FILENAME"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    check = pd.read_csv(out)
    if not check[url_col].astype(str).equals(df[url_col].astype(str)):
        raise RuntimeError("URL sequence changed during output writing.")
    output_hash = sha256(out)
    expected = Path(args.expected_supplied) if args.expected_supplied else None
    expected_hash = sha256(expected) if expected else None
    report = {
        "canonical_file": source.name,
        "canonical_sha256": sha256(source),
        "recreated_file": out.name,
        "recreated_sha256": output_hash,
        "rows": int(len(check)),
        "columns": int(len(check.columns)),
        "label_counts_after_inversion": {str(int(k)): int(v) for k, v in check[label_col].value_counts().sort_index().items()},
        "operation": "drop FILENAME if present; replace label with 1 - label; preserve all other values and row order",
        "expected_supplied_file": expected.name if expected else None,
        "expected_supplied_sha256": expected_hash,
        "recreated_SHA256_matches_expected_supplied": bool(expected_hash == output_hash) if expected else None,
        "interpretation": (
            "Byte-for-byte match to the supplied artifact confirmed."
            if expected and expected_hash == output_hash
            else "No expected supplied file given; recreate output is written and its SHA-256 recorded."
            if not expected
            else "Byte-for-byte checksum mismatch; inspect pandas/CSV serialization and compare row values with verify_canonical.py."
        )
    }
    (out.parent / (out.stem + "_recreation_report.json")).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
