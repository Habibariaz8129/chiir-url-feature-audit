[CHIIR_2027_Repository_README.md](https://github.com/user-attachments/files/34016000/CHIIR_2027_Repository_README.md)
# CHIIR 2027 URL-feature audit: reproducibility guide

This repository accompanies the short paper **“Auditing a Pre-Fetch URL-Lexical Baseline for Phishing Classification: Feature Availability, Evaluation, and Explainability.”** It provides the feature-extraction code, dataset-provenance checks, experiment runner, saved test predictions, metrics, split manifest, noise-selection outputs, and SHAP reconstruction artefacts.

## Research scope

The study compares four fixed classifiers using six explicitly defined URL-derived lexical features. It audits those features against the similarly named stored PhiUSIIL columns, evaluates random-URL, exact-hostname and registrable-domain holdouts, selects Gaussian feature noise using validation average precision, and reports Random Forest SHAP attributions. No webpage is fetched during feature extraction. No user study, deployed warning, temporal evaluation, or independent external-dataset evaluation was conducted; the proposed pre-navigation warning is a design hypothesis only.

## Dataset and label provenance

Canonical dataset: [UCI PhiUSIIL Phishing URL (Website) Dataset](https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset).

The canonical CSV contains 235,795 records and 56 columns. The supplied working artefact contains the same 235,795 URL-ordered records and 55 columns. The record-level comparison found 235,370 distinct URLs, no unmatched URLs, 425 duplicate URL groups, exact agreement across all 54 shared non-label columns, and an inverted binary label in every record. The only canonical column absent from the supplied artefact is `FILENAME`. The canonical data dictionary uses label `0` for phishing and `1` for legitimate; the supplied artefact uses the inverse mapping, so supplied label `1` is phishing-positive. The direct comparison and recreation reports are included in the result files.

The canonical CSV is not included in this public repository. Obtain it from the official UCI page above and save it locally as `data/official/PhiUSIIL_Phishing_URL_Dataset.csv`. Do not redistribute the canonical CSV unnecessarily; follow the dataset licence and attribution requirements.

## Environment

The recorded run used Python 3.13.5, pandas 2.2.3, NumPy 2.3.5, scikit-learn 1.8.0, XGBoost 3.1.3, SHAP 0.50.0 and idna 3.17. Install the pinned Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

The exact-hostname and registrable-domain group split code relies on the system `libpsl` library. The recorded experiment environment was Linux; ensure `libpsl` is installed and discoverable before running all three split protocols. Versions are recorded in `software_versions.json` among the published artefacts.

## Reproduction instructions

The files in this anonymous mirror are currently listed at the repository root. The commands below therefore use root-level Python script names rather than `scripts/` paths. Run commands from the downloaded/cloned repository root.

### 1. Create the data directory and download the canonical CSV

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path "data/official" | Out-Null
```

On Linux/macOS:

```bash
mkdir -p data/official
```

Download `PhiUSIIL_Phishing_URL_Dataset.csv` from the official UCI page and place it in `data/official/`.

### 2. Recreate the supplied label-inverted artefact and verify it

```bash
python recreate_supplied_artifact.py --canonical data/official/PhiUSIIL_Phishing_URL_Dataset.csv --out data/phiusiil_phishing.csv
python verify_canonical.py --supplied data/phiusiil_phishing.csv --canonical data/official/PhiUSIIL_Phishing_URL_Dataset.csv --out results/canonical_match
```

The reconstruction performs two documented operations: drop `FILENAME` if present and replace `label` with `1 - label`, preserving row order and the remaining fields. The supplied-file SHA-256 recorded in the verification report is `78636bd2bf9166f45b6a31232ef461a2814ef38e53f287c56deaa994d2d43fa9`. CSV serialization can vary across environments; consult the row-level and shared-field comparison reports if a newly written file has a different byte hash.

### 3. Recompute URL-only features and feature-audit outputs

```bash
python extract_features.py --csv data/phiusiil_phishing.csv --out data/recomputed
```

This step removes exact duplicate URLs, keeping the first occurrence, and writes `data/recomputed/deduplicated_features_for_audit.csv.gz` plus feature-agreement metrics, representative URL examples, normalization examples, and an environment/source-audit JSON file.

### 4. Re-run the experiments

```bash
python run_experiments.py --features data/recomputed/deduplicated_features_for_audit.csv.gz --out results/reproduced --bootstrap-resamples 1000
```

The experiment runner generates metrics for all four classifiers on the random-URL, exact-hostname and registrable-domain splits, saved test labels/predictions/probabilities, 1,000-resample confidence intervals, validation-only noise sweep outputs, split manifests, and SHAP values with additivity-reconstruction checks. The stored result files in this repository are the outputs associated with the recorded run; rerunning is a separate operation and may require a compatible Linux environment for `libpsl`.

## Model and split protocol

- **Random URL split:** stratified 80% train/validation pool and 20% test set (seed 42); the pool is split 75% core training / 25% validation (seed 43), yielding approximately 60/20/20 overall.
- **Exact-hostname holdout:** group split by parsed ASCII hostname; it prevents the same hostname appearing across partitions but may allow sibling subdomains of a registrable domain to cross partitions.
- **Registrable-domain holdout:** group split by registrable domain using `libpsl`; this is stricter than exact-hostname separation but remains within the same dataset and is not independent external or temporal validation.
- **Models:** Logistic Regression (`C=0.01`, `max_iter=1000`); Decision Tree (`max_depth=3`); Random Forest (`20` trees, `max_depth=5`); XGBoost (`20` estimators, depth `2`, learning rate `0.03`, subsample and column sample `0.6`, `reg_alpha=2`, `reg_lambda=2`). Random-state settings and thresholds are recorded in the runner and split manifest.
- **Noise:** standardised feature-space Gaussian noise at sigma values 0, 0.1, 0.3 and 0.5 is assessed using validation AP on the random split only. One seeded draw is used for each non-zero level/model; this does not establish robustness to valid URL transformations.
- **SHAP:** Random Forest `TreeExplainer`, interventional feature perturbation and probability output. Background sample: 100 training examples (seed 4242); explained sample: 250 test examples (seed 4243). Feature attributions are model explanations, not causal rules; correlated length and character-count features may share attribution.

## Published artefacts

The repository includes `run_experiments.py`, `extract_features.py`, `url_features.py`, `recreate_supplied_artifact.py` and `verify_canonical.py`; `requirements.txt`; saved metrics and per-model test-prediction CSVs; `split_manifest.json`; feature-agreement and representative-example files; normalization examples; noise-selection outputs; and SHAP values and reconstruction checks. The canonical source comparison includes the per-URL report and shared-column comparison files. A separate human-readable summary is provided in `CHIIR_2027_Canonical_Dataset_Verification_Report.md`.

## Interpretation and limitations

1. The supplied file's binary label inversion has been verified against the canonical UCI records.
2. The observed `https://www.` substring pattern is consistent with a shortened-string counting signature, but the original historical feature-construction code is unavailable, so its cause is not proven.
3. Random and grouped holdouts are from the same corpus; they do not establish external transfer or temporal generalisation.
4. Feature-space Gaussian jitter is not a valid-URL mutation test and does not establish adversarial robustness.
5. SHAP values are not causal explanations or security rules.
6. The proposed pre-navigation warning interface has not been implemented or evaluated with participants; no human benefit is claimed.

## Anonymous review link

Use this anonymized mirror during double-blind review: <https://anonymous.4open.science/r/chiir-url-feature-audit-96DC/>. Do not cite the identifiable source repository in the manuscript or reviewer response.
