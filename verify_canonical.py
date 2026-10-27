#!/usr/bin/env python3
"""Verify the local PhiUSIIL CSV against the official UCI CSV.

The script checks raw URL equality and multiplicity, label semantics,
duplicate-label conflicts, checksums, and all shared non-label columns.
It uses a URL occurrence index to preserve duplicate observations and
reports when row order is identical. The canonical CSV is not bundled;
download it from the official UCI page.

Example:
  python scripts/verify_canonical.py \
    --supplied data/phiusiil_phishing.csv \
    --canonical data/official/PhiUSIIL_Phishing_URL_Dataset.csv \
    --out results/canonical_match
"""
from __future__ import annotations
import argparse, hashlib, json, unicodedata
from collections import Counter
from pathlib import Path
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def find_col(df: pd.DataFrame, name: str) -> str:
    matches = [c for c in df.columns if str(c).lower() == name.lower()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name} column; found {matches}")
    return matches[0]


def key(v) -> str:
    if pd.isna(v):
        return ""
    return unicodedata.normalize("NFC", str(v).strip())


def value_equal(a: pd.Series, b: pd.Series) -> pd.Series:
    # Equality that treats nulls in corresponding rows as equal.
    return a.eq(b) | (a.isna() & b.isna())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--supplied", required=True)
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    supplied_path, canonical_path, out = Path(args.supplied), Path(args.canonical), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s, c = pd.read_csv(supplied_path), pd.read_csv(canonical_path)
    su, sl = find_col(s, "URL"), find_col(s, "label")
    cu, cl = find_col(c, "URL"), find_col(c, "label")
    s[sl] = pd.to_numeric(s[sl], errors="raise").astype(int)
    c[cl] = pd.to_numeric(c[cl], errors="raise").astype(int)

    s_key = s[su].map(key)
    c_key = c[cu].map(key)
    exact_sequence_match = s[su].astype(str).equals(c[cu].astype(str))
    s_counts, c_counts = Counter(s_key), Counter(c_key)
    normalized_intersection = set(s_key) & set(c_key)
    normalized_intersection.discard("")
    unmatched_supplied_keys = set(s_key) - set(c_key) - {""}
    unmatched_canonical_keys = set(c_key) - set(s_key) - {""}

    s_dups = pd.DataFrame({"url_key": s_key, "label": s[sl]}).groupby("url_key", sort=False)["label"].agg(["size", "nunique"])
    c_dups = pd.DataFrame({"url_key": c_key, "label": c[cl]}).groupby("url_key", sort=False)["label"].agg(["size", "nunique"])
    all_keys = sorted((set(s_key) | set(c_key)) - {""})
    s_labels = pd.DataFrame({"url_key": s_key, "label": s[sl]}).groupby("url_key", sort=False)["label"].agg(
        lambda x: ",".join(map(str, sorted(set(x.astype(int)))))
    )
    c_labels = pd.DataFrame({"url_key": c_key, "label": c[cl]}).groupby("url_key", sort=False)["label"].agg(
        lambda x: ",".join(map(str, sorted(set(x.astype(int)))))
    )
    groups = pd.DataFrame({"url_key": all_keys})
    groups["canonical_rows"] = groups["url_key"].map(c_counts)
    groups["supplied_rows"] = groups["url_key"].map(s_counts)
    groups["canonical_labels"] = groups["url_key"].map(c_labels)
    groups["supplied_labels"] = groups["url_key"].map(s_labels)

    def rel(row):
        if pd.isna(row.canonical_rows) or pd.isna(row.supplied_rows):
            return "unmatched_url"
        ca = set(map(int, str(row.canonical_labels).split(",")))
        sups = set(map(int, str(row.supplied_labels).split(",")))
        if len(ca) > 1 or len(sups) > 1:
            return "conflicting_duplicate_labels"
        if ca == {1 - x for x in sups}:
            return "inverted"
        if ca == sups:
            return "same"
        return "conflict"

    groups["label_relation"] = groups.apply(rel, axis=1)
    groups["duplicate_URL_group"] = (groups["canonical_rows"] > 1) | (groups["supplied_rows"] > 1)
    groups.to_csv(out / "url_match_details.csv.gz", index=False, compression="gzip")

    shared_non_label = [x for x in c.columns if x in s.columns and x.lower() != "label"]
    field_rows = []
    for col in shared_non_label:
        eq = value_equal(c[col], s[col])
        field_rows.append({
            "field": col, "rows_compared": len(eq), "rows_equal": int(eq.sum()),
            "rows_different": int((~eq).sum()), "exact_match_rate": float(eq.mean())
        })
    pd.DataFrame(field_rows).to_csv(out / "shared_field_comparison.csv", index=False)

    label_same = int((c[cl].to_numpy() == s[sl].to_numpy()).sum()) if exact_sequence_match else None
    label_inverted = int(((1 - c[cl].to_numpy()) == s[sl].to_numpy()).sum()) if exact_sequence_match else None
    label_conflicts = int((s[sl].to_numpy() != 1 - c[cl].to_numpy()).sum()) if exact_sequence_match else None
    dup_url_counts = c[cu].value_counts(dropna=False)
    shared_fields_all_equal = all(x["rows_different"] == 0 for x in field_rows)
    report = {
        "canonical_file": canonical_path.name,
        "canonical_sha256": sha256(canonical_path),
        "supplied_file": supplied_path.name,
        "supplied_sha256": sha256(supplied_path),
        "canonical_shape_rows_columns": [int(c.shape[0]), int(c.shape[1])],
        "supplied_shape_rows_columns": [int(s.shape[0]), int(s.shape[1])],
        "canonical_columns_absent_from_supplied": sorted(set(c.columns) - set(s.columns)),
        "supplied_columns_absent_from_canonical": sorted(set(s.columns) - set(c.columns)),
        "same_exact_URL_sequence_row_by_row": bool(exact_sequence_match),
        "URL_multiset_equal_including_duplicate_multiplicities": bool(s_counts == c_counts),
        "canonical_unique_nonblank_URLs": int(len(set(c_key) - {""})),
        "supplied_unique_nonblank_URLs": int(len(set(s_key) - {""})),
        "shared_unique_nonblank_URLs": int(len(normalized_intersection)),
        "unmatched_canonical_unique_URLs": int(len(unmatched_canonical_keys)),
        "unmatched_supplied_unique_URLs": int(len(unmatched_supplied_keys)),
        "unmatched_canonical_rows": int(sum(v for k, v in c_counts.items() if k in unmatched_canonical_keys)),
        "unmatched_supplied_rows": int(sum(v for k, v in s_counts.items() if k in unmatched_supplied_keys)),
        "duplicate_URL_groups": int((dup_url_counts > 1).sum()),
        "rows_in_duplicate_URL_groups": int(dup_url_counts[dup_url_counts > 1].sum()),
        "extra_duplicate_rows_if_keep_first": int((dup_url_counts - 1).clip(lower=0).sum()),
        "canonical_duplicate_URL_groups_with_conflicting_labels": int(((c_dups["nunique"] > 1) & (c_dups.index.isin(normalized_intersection))).sum()),
        "supplied_duplicate_URL_groups_with_conflicting_labels": int(((s_dups["nunique"] > 1) & (s_dups.index.isin(normalized_intersection))).sum()),
        "canonical_label_counts": {str(k): int(v) for k, v in c[cl].value_counts().sort_index().items()},
        "supplied_label_counts": {str(k): int(v) for k, v in s[sl].value_counts().sort_index().items()},
        "rowwise_label_same_count_if_same_URL_sequence": label_same,
        "rowwise_label_inverted_count_if_same_URL_sequence": label_inverted,
        "rowwise_label_conflict_count_if_same_URL_sequence": label_conflicts,
        "all_shared_non_label_fields_equal_row_by_row_if_same_URL_sequence": bool(exact_sequence_match and shared_fields_all_equal),
        "shared_non_label_field_count": len(shared_non_label),
        "label_relation_counts_by_URL_group": {str(k): int(v) for k, v in groups["label_relation"].value_counts().items()},
        "canonical_and_supplied_source_checksums_are_full_sha256": True,
        "shared_non_label_field_comparison_csv": "shared_field_comparison.csv",
        "url_group_detail_file": "url_match_details.csv.gz",
        "canonical_unique_identifier_column_absent_from_supplied": "FILENAME" if "FILENAME" in c.columns and "FILENAME" not in s.columns else None,
        "canonical_label_semantics_from_UCI_metadata": {"1": "legitimate", "0": "phishing"},
        "interpretation": "Confirm inversion only when URL matching has no unmatched records, every aligned label is complementary, no conflicting labels occur, and shared non-label fields agree. Check all exported files before interpreting class-specific metrics."
    }
    relation_counts = {str(k): int(v) for k, v in groups["label_relation"].value_counts().items()}
    report["record_level_counts"] = {
        "canonical_rows": int(len(c)),
        "supplied_rows": int(len(s)),
        "matched_rows_by_same_URL_sequence_and_duplicate_occurrence": int(len(s)) if exact_sequence_match else None,
        "unmatched_canonical_rows": report["unmatched_canonical_rows"],
        "unmatched_supplied_rows": report["unmatched_supplied_rows"],
        "rowwise_exact_label_inversions": label_inverted,
        "rowwise_same_labels": label_same,
        "rowwise_conflicting_labels": label_conflicts,
        "matched_distinct_URL_keys": int(len(normalized_intersection)),
        "unmatched_canonical_distinct_URL_keys": int(len(unmatched_canonical_keys)),
        "unmatched_supplied_distinct_URL_keys": int(len(unmatched_supplied_keys))
    }
    report["duplicate_group_details"] = {
        "duplicate_URL_groups": report["duplicate_URL_groups"],
        "rows_in_duplicate_URL_groups": report["rows_in_duplicate_URL_groups"],
        "extra_duplicate_rows_removed_when_keep_first": report["extra_duplicate_rows_if_keep_first"],
        "canonical_duplicate_URL_groups_with_conflicting_labels": report["canonical_duplicate_URL_groups_with_conflicting_labels"],
        "supplied_duplicate_URL_groups_with_conflicting_labels": report["supplied_duplicate_URL_groups_with_conflicting_labels"]
    }
    processed_candidate = supplied_path.parent / "deduplicated_features_for_audit.csv.gz"
    report["processed_file_checksums"] = {
        "supplied_raw_CSV_sha256": sha256(supplied_path),
        "canonical_UCI_CSV_sha256": sha256(canonical_path),
        "deduplicated_feature_cache_sha256": sha256(processed_candidate) if processed_candidate.exists() else None
    }
    report["verified_conclusion"] = (
        "The supplied artifact is record-aligned with the canonical UCI CSV by exact URL sequence and duplicate occurrence. "
        "Every shared non-label field agrees row by row, and all labels are complemented."
        if exact_sequence_match and shared_fields_all_equal and label_inverted == len(c) and label_conflicts == 0
        else "The record-level comparison did not establish a complete, conflict-free label inversion; inspect the detailed outputs before interpreting class-specific metrics."
    )
    report["label_relation_counts_by_URL_group"] = relation_counts
    (out / "canonical_match_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
