# Canonical dataset verification report

**Comparison date:** 27 October 2026  
**Status:** Complete; supplied label mapping verified.

## Files compared

| Artifact | Rows | Columns | SHA-256 |
|---|---:|---:|---|
| Official UCI: `PhiUSIIL_Phishing_URL_Dataset.csv` | 235,795 | 56 | `a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6` |
| Supplied: `phiusiil_phishing.csv` | 235,795 | 55 | `78636bd2bf9166f45b6a31232ef461a2814ef38e53f287c56deaa994d2d43fa9` |

## Record-level results

| Check | Result |
|---|---:|
| Exact URL sequence identical row by row | Yes |
| URL multiset and duplicate multiplicities identical | Yes |
| Matched rows | 235,795 / 235,795 |
| Shared distinct URL keys | 235,370 |
| Unmatched distinct URLs in either file | 0 |
| Shared non-label columns | 54 |
| Shared non-label columns identical row by row | 54 / 54 |
| Canonical-only column | `FILENAME` |
| Rows with inverted labels | 235,795 / 235,795 |
| Rows with same labels or label conflicts | 0 |
| Exact-URL duplicate groups | 425 |
| Rows belonging to those duplicate groups | 850 |
| Extra duplicate rows removed by keeping the first occurrence | 425 |
| Duplicate groups with conflicting labels (either file) | 0 |

The canonical UCI dictionary assigns label `1` to legitimate and label `0` to phishing. The supplied artifact's label is exactly the complement of the canonical label for every row, so its verified mapping is **label `0` = legitimate and label `1` = phishing**. All 54 shared fields other than the label match exactly for every aligned row. The canonical file includes a `FILENAME` column that is absent from the supplied CSV.

## Consequence for the experiments

The existing experiments were trained and evaluated with supplied label `1` as the positive class. Because the comparison verifies that this label corresponds to canonical phishing rows, the existing reported precision, recall, F1, FPR/FNR and SHAP probabilities can be interpreted as phishing-positive results for this corpus. No model rerun is required solely to correct the label interpretation. The exact duplicate handling and feature audit remain as reported in the manuscript.

The deterministic operation of dropping `FILENAME`, replacing `label` by `1 - label`, and retaining the remaining values/order recreates the supplied CSV byte-for-byte: the reconstructed SHA-256 equals the supplied-file SHA-256. The historical script/person that originally performed the operation is not identifiable from the data alone.

## Reproducible materials

The full report is in `results/canonical_match/canonical_match_report.json`. `shared_field_comparison.csv` records equality for all shared columns; `url_match_details.csv.gz` is a duplicate-aware per-URL report. The optimized `scripts/verify_canonical.py` can reproduce these comparisons when the canonical CSV is downloaded from the [official UCI page](https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset).

SHA-256 for the deduplicated feature cache: `16b96ba75b8c5598929952eead08007643515a9c2ca5fc3b50f3803622513988`.
