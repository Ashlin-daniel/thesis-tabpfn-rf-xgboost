# thesis-tabpfn-rf-xgboost

**Applications of TabPFN in microbiome research in R**

TabPFN vs Random Forest vs XGBoost on Gut Microbiome Data

MSc thesis project comparing TabPFN, Random Forest, and XGBoost on predicting host Age, BMI, and Sex from gut microbiome composition, with a focus on how each model's performance changes with training sample size.

## Repository structure

```
.
├── schirmer_5foldCV.R              # main analysis: 5-fold CV, learning curves by sample size
├── schirmer_5foldCV+sig.R          # analysis: 5-fold CV, learning curves, significance testing, class significance
├── sanitycheck_fake+real.R         # validation: fully synthetic + real-X semi-synthetic ground-truth check
├── sanitycheck_tabpfn.R            # validation: standalone TabPFN planted signal ground-truth check
├── real_data_sanity_check.R        # validation: real data ground-truth check
├── real_permutation_check.R        # validation: permutation significance test (continuous + binned)
├── train_classifier.py             # TabPFN classifier helper (Sex)
├── train_regression_latest.py      # RF regression helper (Age, BMI)
├── metadata_Schirmer_2016_500FG_human_core_wide.tsv   # sample metadata
├── metaphlan4_Schirmer_2016_500FG_2026-06-25.tsv      # taxonomic abundance table
├── LICENSE
└── README.md
```

Everything the script needs sits in the same folder — no separate `data/` subfolder required.


## Dataset

Schirmer_2016_500FG cohort, sourced from the MetaLog database.

- 471 samples, Netherlands, healthy adults (18–75 years)
- Sex distribution: ~200 male / 265 female
- After filtering to adults with complete age/BMI/sex: 456 samples (364 train / 92 test)

Publication: Schirmer et al., *Linking the Human Gut Microbiome to Inflammatory Cytokine Production Capacity*, Cell (2016).

## Validation

Two checks address whether a near-zero R² for Age/BMI reflects genuine biological signal (or its absence) rather than a pipeline bug:

- **`sanitycheck_fake+real.R`** — plants a known signal (fully synthetic data, and separately a known-strength signal injected into the real feature matrix) and checks whether RF, XGBoost, and TabPFN can recover it. Establishes the pipeline's detection floor across a range of effect sizes.
- **`real_permutation_check.R`** — permutation significance testing on the real data: shuffles labels to build a chance-only null distribution, for both continuous R² (Age, BMI) and binned versions (age decade, WHO BMI class).

## Setup

- **R packages:** `randomForest`, `xgboost`, `caret`, `reticulate`, `dplyr`, `tidyr`, `ggplot2`, `patchwork`
- **Python:** virtualenv at `./venv311` (Python 3.11), with `tabpfn==0.1.9`, `torch==2.1.0`, `scikit-learn==1.1.3`, `numpy<2`

Optional environment variables (default to the current directory if unset):
`THESIS_DATA_DIR`, `THESIS_VENV_PATH`, `THESIS_SCRIPT_DIR`

## How to run

1. `schirmer_5foldCV.R` — main analysis: learning curves, 5-fold CV, all 3 models × 3 tasks (Age, BMI, Sex)
2. `sanitycheck_fake+real.R` — pipeline validation (ground-truth recovery check)
3. `real_permutation_check.R` — significance testing on the real results

## License

MIT — see [LICENSE](LICENSE).
