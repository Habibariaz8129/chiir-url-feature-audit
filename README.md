# CHIIR 2027 URL-feature audit — reproducibility package

**Status: canonical label mapping verified; repository staging package.** This repository staging version omits the local input CSV and regenerated feature cache; download the official UCI release and reconstruct the supplied artifact using the steps below. Its reconstructed supplied-file SHA-256 should be `78636bd2bf9166f45b6a31232ef461a2814ef38e53f287c56deaa994d2d43fa9`. The supplied CSV was compared directly with the canonical UCI CSV. All 235,795 URL-ordered records match; all 235,370 distinct URLs are shared with duplicate multiplicities preserved; 54 shared non-label fields agree row by row; and every supplied label is the complement of the canonical label. The only canonical column absent from the supplied file is `FILENAME`.

The canonical UCI data dictionary defines label `1` as legitimate and label `0` as phishing. The direct comparison confirms that the supplied artifact uses the inverse coding: supplied label `1` = phishing and supplied label `0` = legitimate. There are 425 duplicate URL groups (850 rows involved), no conflicting duplicate-label groups, and no unmatched URLs. The canonical source SHA-256 is `a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6`. Applying the simple transformation of dropping `FILENAME` and inverting `label` recreates the supplied CSV byte-for-byte, with the same SHA-256. The transformation script and its reconstruction report are included. The deduplicated feature cache and every processed result are checksummed in `SHA256SUMS.txt`; the detailed source comparison is in `results/canonical_match/`.

## Double-blind review and sharing

This package is prepared for double-blind review. Before citing it in a submission, publish the files under a neutral repository name, create an anonymous mirror with <https://anonymous.4open.science/>, configure author-specific redaction terms if any are present in text files, and inspect the mirrored files in a logged-out/private browser window. Set the mirror's expiry action to remove the mirror rather than redirecting to the identifiable source repository. Cite only the tested anonymous-mirror URL in the review manuscript and response letter. The public source repository and anonymous mirror are separate; creating this archive does not publish either one.

## Official source and provenance verification

Canonical dataset page: <https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset>

The UCI dataset page identifies the release as PhiUSIIL Phishing URL (Website), by Arvind Prasad and Shalini Chandra (2024), licensed CC BY 4.0. Cite the dataset and its associated paper when redistributing or reusing it.

Download `PhiUSIIL_Phishing_URL_Dataset.csv` from the official UCI page and place it at `data/official/PhiUSIIL_Phishing_URL_Dataset.csv`. First reconstruct the original supplied artifact, then verify it. The verifier matches URL sequence and duplicate multiplicity, counts matched/unmatched rows and label relations, compares shared non-label columns row by row, and records file checksums. The completed comparison report and per-URL details are included in this repository.

```bash
mkdir -p data/official
python scripts/recreate_supplied_artifact.py \
  --canonical data/official/PhiUSIIL_Phishing_URL_Dataset.csv \
  --out data/phiusiil_phishing.csv
python scripts/verify_canonical.py \
  --supplied data/phiusiil_phishing.csv \
  --canonical data/official/PhiUSIIL_Phishing_URL_Dataset.csv \
  --out results/canonical_match
```

The recorded run produced 235,795 row matches, 235,370 shared distinct URLs, zero unmatched URLs, an exact label inversion in every row, zero conflicts, and exact agreement in all 54 shared non-label columns. Review `results/canonical_match/canonical_match_report.json`, `shared_field_comparison.csv`, and the compressed `url_match_details.csv.gz`. If rerunning on another copy of the source, compare the generated report with these checksums and counts.

To reconstruct the label-inverted local CSV from the canonical file:
```bash
python scripts/recreate_supplied_artifact.py \
  --canonical data/official/PhiUSIIL_Phishing_URL_Dataset.csv \
  --out data/phiusiil_phishing_recreated.csv \
  --expected-supplied data/phiusiil_phishing.csv
```
For the checked source, the reconstructed file hash is identical to the supplied CSV hash. Model results already use supplied label `1` as positive, so the verified mapping means they are phishing-positive; no model rerun is needed solely due to this label inversion.

## Environment

The recorded environment was Python 3.13.5, pandas 2.2.3, NumPy 2.3.5, scikit-learn 1.8.0, XGBoost 3.1.3, SHAP 0.50.0, `idna` for IDNA/UTS46 processing, and system `libpsl` 0.21.2 (+libidn2/2.3.7) for registrable-domain grouping. `libpsl.so.5` is required for the registrable-domain split; the recorded runs were made on Linux. See `requirements.txt` and `results/software_versions.json`.

## Reproduce feature extraction and audit

```bash
python -m pip install -r requirements.txt
python scripts/extract_features.py --csv data/phiusiil_phishing.csv --out data/recomputed
```

This exact-URL deduplicates the CSV (keeping the first occurrence), extracts six lexical features directly from the URL without retrieving any webpage, and writes a compressed feature cache, agreement metrics, representative examples, substring-signature results, normalization examples, and a source/environment report.

The full reproducibility archive includes `data/deduplicated_features_for_audit.csv.gz`, but this repository staging version omits it to keep the repository lightweight. Generate the cache by running the extraction command below.

## Reproduce model, split, noise and SHAP results

```bash
python scripts/run_experiments.py \
  --features data/recomputed/deduplicated_features_for_audit.csv.gz \
  --out results/reproduced \
  --bootstrap-resamples 1000
```

Outputs include fixed-threshold confusion counts and metrics, 1,000-row-bootstrap confidence intervals for accuracy/F1, all saved test labels/predictions/probabilities, a seeded validation-only Gaussian-noise sweep, split manifests with partition class counts and group-overlap checks, SHAP values plus probability reconstruction checks, and a metric-recalculation check confirming the saved threshold metrics. Model settings are pinned in the script and included in `results/split_manifest.json`.

### Split and model details

- Random URL: stratified split with test fraction 0.20 and seed 42; validation is 0.25 of remaining rows with seed 43, yielding 60/20/20.
- Exact hostname: `GroupShuffleSplit`, 20% of hostname groups for test (seed 42), then 25% of remaining groups for validation (seed 43).
- Registrable domain: same group-split procedure using `libpsl` built-in Public Suffix List. Exact-hostname separation can leave sibling subdomains on opposite sides; registrable-domain separation prevents that.
- Logistic Regression: `C=0.01`, `max_iter=1000`; Decision Tree: `max_depth=3`; Random Forest: `n_estimators=20`, `max_depth=5`; XGBoost: 20 estimators, depth 2, learning rate 0.03, subsample/column sample 0.6, `reg_alpha=2`, `reg_lambda=2`, histogram tree method; all random-state values 42 and model thread count 1 where supported.
- Noise: validation AP for `sigma={0,0.1,0.3,0.5}` on the random split only; one deterministic draw per non-zero setting/model. All four select `sigma=0`. Grouped holdouts use the resulting no-noise baseline and do not retune noise.
- SHAP: `TreeExplainer`, `feature_perturbation=interventional`, `model_output=probability`; 100 training examples sampled with seed 4242 and 250 test examples with seed 4243 after the recorded standardization step. Maximum absolute additivity error is approximately `2.96e-8`.

## Key interpretation cautions

1. The source-label inversion is verified record by record. Metrics use the supplied artifact's label 1 as phishing-positive.
2. The `https://www.` feature-count pattern is an empirical signature consistent with counting a shortened URL slice; without the original feature-construction source code, the historical cause is not proven.
3. Exact-hostname and registrable-domain tests are holdouts from the same dataset, not independent or temporal evaluations.
4. Gaussian noise in standardized feature space is not equivalent to a valid URL mutation and does not establish adversarial robustness.
5. SHAP values are model attributions, not causal rules; correlated URL-length and character-count features can share attribution.
6. No interface, user study, or human benefit was evaluated. The interaction wording in the manuscript is a design hypothesis only.

## Data and generated files

- `data/phiusiil_phishing.csv`: generated locally from the official UCI CSV by `scripts/recreate_supplied_artifact.py` (not committed).
- `data/deduplicated_features_for_audit.csv.gz`: regenerated locally by `scripts/extract_features.py` (not committed in this lightweight staging version).
- `scripts/`: URL normalization/features, extraction/audit, canonical comparison, and experiment runner.
- `results/`: current reported outputs, with test label/prediction/probability exports under `results/predictions/`.
- `SHA256SUMS.txt`: per-file checksums for package files, generated during packaging.
