# thesis-tabpfn-rf-xgboost
Applications of TabFPN in microbiome research in R 
TabPFN vs Random Forest vs XGBoost on Gut Microbiome Data

MSc thesis project comparing TabPFN, Random Forest, and XGBoost on predicting
host Age, BMI, and Sex from gut microbiome composition, with a focus on how
each model's performance changes with training sample size.

Repository structure

.
├── schirmer_code.R                                          # main analysis script
├── train_classifier.py                                      # TabPFN classifier helper (Sex)
├── train_regression_latest.py                                # RF regression helper (Age, BMI)
├── metadata_Schirmer_2016_500FG_human_core_wide.tsv           # sample metadata
├── metaphlan4_Schirmer_2016_500FG_2026-06-25.tsv              # taxonomic abundance table
├── LICENSE
└── README.md

Everything the script needs sits in the same folder — no separate data/
subfolder required.

Dataset

Schirmer_2016_500FG cohort, sourced from the MetaLog database.


471 samples, Netherlands, healthy adults (18-75 years)
Sex distribution: ~200 male / 265 female
After filtering to adults with complete age/BMI/sex: 456 samples
(364 train / 92 test)
Publication: Schirmer et al., Linking the Human Gut Microbiome to
Inflammatory Cytokine Production Capacity,
Cell (2016).
