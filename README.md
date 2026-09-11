# TabPFN vs Random Forest vs XGBoost on Gut Microbiome Data

MSc thesis project comparing **TabPFN**, **Random Forest**, and **XGBoost** on predicting host **age**, **BMI**, and **sex** from gut microbiome composition, with a focus on how each model's performance changes with training sample size — and whether observed results are statistically significant, not just numerically different from chance.

---

## Repository structure

```
.
├── schirmer5foldcv+sig.R             # MAIN SCRIPT: 5-fold CV, learning curves,
│                                      # significance testing, cut-point classification
├── sanitycheck_fake+real.R           # validation: synthetic ground-truth recovery check
├── permutation.R                     # validation: permutation test on synthetic data
│                                      # (chance-distribution histogram vs. real R^2)
├── real_data_sanity_check.R          # validation: real-data extreme-groups check
│                                      # (produces the AUC = 0.688 result cited in the thesis)
├── real_permutation_check.R          # significance testing: continuous + binned outcomes,
│                                      # all three models
├── train_classifier.py               # TabPFN classifier helper (Sex)
├── train_regression_latest.py        # regression helper (Age, BMI)
├── metadata_Schirmer_2016_500FG_human_core_wide.tsv   # sample metadata
├── metaphlan4_Schirmer_2016_500FG_2026-06-25.tsv      # taxonomic abundance table
├── LICENSE
└── README.md
```

Everything the scripts need sits in the same folder — no separate `data/` subfolder required.

---

## Dataset

Schirmer 2016 500FG cohort, sourced from the [MetaLog database](https://microbiome.github.io/).

- 471 samples, Netherlands, healthy adults (18-75 years)
- After filtering to adults with complete age/BMI/sex data: **456 samples** (~364 train / ~92 test per fold)

**Publication:** Schirmer M et al. (2016) *Linking the Human Gut Microbiome to Inflammatory Cytokine Production Capacity.* Cell.

---

## Validation

Two independent checks confirm that near-zero R² for Age/BMI prediction reflects genuine absence of signal, not a pipeline bug — this is validated *before* trusting the main results:

- **`sanitycheck_fake+real.R`** — plants a known signal into fully synthetic compositional data and checks whether Random Forest, XGBoost, and TabPFN can recover it. Confirms the pipeline detects real signal when one is deliberately present.
- **`permutation.R`** — formally tests this synthetic result via a permutation test: shuffles the synthetic data's labels 200 times to build a chance-only R² distribution, then compares the real R² against it (p = 0.005), with an accompanying histogram visualizing the result.
- **`real_data_sanity_check.R`** — an extreme-groups comparison (youngest vs. oldest age quartile) using real, unmodified data, providing a real-data positive control independent of the synthetic check.

---

## Significance testing

**`real_permutation_check.R`** — permutation-based significance testing on the real results: shuffles outcome labels to build a chance-only null distribution, applied to both continuous outcomes (Age, BMI R²) and binned outcomes (age decade, WHO BMI class), across all three models.

---

## Setup

**R packages:** `randomForest`, `xgboost`, `mia`, `TreeSummarizedExperiment`, `reticulate`, `dplyr`, `tidyr`, `ggplot2`, `patchwork`

**Python:** virtual environment at `./venv311` (Python 3.11), with `tabpfn==0.1.9`, `torch==2.1.0`, `scikit-learn==1.1.3`, `numpy<2`

**Optional environment variables** (default to the current directory if unset): `THESIS_DATA_DIR`, `THESIS_VENV_PATH`, `THESIS_SCRIPT_DIR`

---

## How to run

1. **`schirmer5foldcv+sig.R`** — main analysis: learning curves, 5-fold CV, significance testing, and cut-point classification across all 3 models × 3 tasks (Age, BMI, Sex)
2. **`sanitycheck_fake+real.R`**, **`permutation.R`** and **`real_data_sanity_check.R`** — pipeline validation (ground-truth recovery checks and significance confirmation)
3. **`real_permutation_check.R`** — full significance testing on the real results

---

## License

MIT — see [LICENSE](https://github.com/Ashlin-daniel/thesis-tabpfn-rf-xgboost/blob/main/LICENSE).
