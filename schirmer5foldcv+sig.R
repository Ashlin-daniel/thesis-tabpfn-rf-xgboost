library(reticulate)
library(dplyr)
library(tidyr)
library(xgboost)
library(randomForest)
library(glmnet)
library(ggplot2)
library(patchwork)

# dependencies for the mia-based CLR step
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install(c("mia", "TreeSummarizedExperiment"), update = FALSE, ask = FALSE)
library(mia)
library(TreeSummarizedExperiment)

data_dir   <- Sys.getenv("THESIS_DATA_DIR",   unset = ".")
venv_path  <- Sys.getenv("THESIS_VENV_PATH",  unset = "./venv311")
script_dir <- Sys.getenv("THESIS_SCRIPT_DIR", unset = ".")

meta_file  <- file.path(data_dir, "metadata_Schirmer_2016_500FG_human_core_wide.tsv")
abund_file <- file.path(data_dir, "metaphlan4_Schirmer_2016_500FG_2026-06-25.tsv")

use_virtualenv(venv_path)

py_run_string("import warnings; warnings.filterwarnings('ignore')")
source_python(file.path(script_dir, "train_classifier.py"))
source_python(file.path(script_dir, "train_regression_latest.py"))

# importing tabpfn directly for the quantile-binning regression
tabpfn_pkg <- import("tabpfn")

# 1. loading and merging data

meta  <- read.delim(meta_file)
abund <- read.delim(abund_file)
abund <- abund[grepl("s__[^|]+$", abund$clade_name), ]# restricting to species level taxonomic rank

abund_wide <- abund |>
  pivot_wider(names_from = clade_name, values_from = rel_abund, values_fill = 0)

merged <- dplyr::inner_join(meta, abund_wide, by = "sample_alias") |>
  filter(age_years >= 18, !is.na(age_years), !is.na(bmi), sex %in% c("male", "female"))



# 2. feature matrix
# capping at 100 taxa so TabPFN can accept it

taxa_cols <- setdiff(colnames(merged), colnames(meta))
X_full    <- merged[, taxa_cols]

prevalent <- X_full[, colMeans(X_full > 0) >= 0.10]
top_taxa  <- names(sort(apply(prevalent, 2, var), decreasing = TRUE))[1:min(100, ncol(prevalent))]
X_raw     <- prevalent[, top_taxa]
# CLR (centered log-ratio) transform via mia/TreeSummarizedExperiment (OMA
# framework, Borman et al. 2026). Pseudocount is computed explicitly (half the smallest
# non-zero abundance) and passed directly, rather than using mia's
# automatic pseudocount = TRUE default, to ensure results remain consistent
# with the pipeline's original validated preprocessing.

my_pseudocount <- min(X_raw[X_raw > 0], na.rm = TRUE) / 2

clr_transform <- function(X) {
  X <- as.matrix(X); X[X == 0] <- my_pseudocount
  log_X <- log(X); log_X - rowMeans(log_X)
}
X <- as.data.frame(clr_transform(X_raw))

y_age <- merged$age_years
y_bmi <- merged$bmi
y_sex <- merged$sex
y_sex_bin <- as.integer(y_sex == "female")


# 3. helper functions

r2_score <- function(pred, truth) {
  1 - sum((pred - truth)^2) / sum((truth - mean(truth))^2)
}
mae_score <- function(pred, truth) mean(abs(pred - truth))
# manual AUC
auc_score <- function(probs, actual_binary) {
  n1 <- sum(actual_binary == 1); n0 <- sum(actual_binary == 0)
  r  <- rank(probs)
  (sum(r[actual_binary == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# one XGBoost helper for both regression and classification.

run_xgb <- function(X_train, y_train, X_test, y_test, classification = FALSE) {
  y_fit <- if (classification) factor(y_train) else y_train  # observed levels only,
  model <- xgboost(x = as.matrix(X_train), y = y_fit,
                   nrounds = 100, max_depth = 3, learning_rate = 0.1,
                   subsample = 0.8, colsample_bytree = 0.8, verbosity = 0)
  pred <- predict(model, as.matrix(X_test))
  if (classification) auc_score(pred, y_test) else r2_score(pred, y_test)
}
run_xgb_with_mae <- function(X_train, y_train, X_test, y_test) {
  model <- xgboost(x = as.matrix(X_train), y = y_train,
                   nrounds = 100, max_depth = 3, learning_rate = 0.1,
                   subsample = 0.8, colsample_bytree = 0.8, verbosity = 0)
  pred <- predict(model, as.matrix(X_test))
  list(R2 = r2_score(pred, y_test), MAE = mae_score(pred, y_test))
}

# Random Forest classifier (for Sex)

run_rf_clf <- function(X_train, y_train_fac, X_test, y_test_bin) {
  model <- randomForest(x = X_train, y = y_train_fac, ntree = 500)
  probs <- predict(model, X_test, type = "prob")[, "female"]
  auc_score(probs, y_test_bin)
}

# mean baseline: predicts the training mean for every test point, regardless
# of features : an R^2 = 0 by definition, gives a reference point for the
# other models' R^2 values
run_mean_baseline <- function(X_train, y_train, X_test, y_test) {
  pred <- rep(mean(y_train), length(y_test))
  r2_score(pred, y_test)
}

# elastic net baseline: a regularised linear model on the CLR data, as a
# simpler point of comparison against the tree-based/TabPFN models
run_enet <- function(X_train, y_train, X_test, y_test) {
  fit  <- cv.glmnet(as.matrix(X_train), y_train, alpha = 0.5)
  pred <- as.vector(predict(fit, as.matrix(X_test), s = "lambda.min"))
  r2_score(pred, y_test)
}

run_enet_with_mae <- function(X_train, y_train, X_test, y_test) {
  fit  <- cv.glmnet(as.matrix(X_train), y_train, alpha = 0.5)
  pred <- as.vector(predict(fit, as.matrix(X_test), s = "lambda.min"))
  list(R2 = r2_score(pred, y_test), MAE = mae_score(pred, y_test))
}

# TabPFN regression via quantile binning:
#  1. cutting y_train into ~n_bins quantile buckets
#  2. fitting TabPFNClassifier to predict bucket membership
#  3. reconstructing a continuous prediction as the probability-weighted average of each bucket's mean y-value 
# (not a true continuous output — TabPFN 0.1.9 has no native regressor)
run_tabpfn_reg <- function(X_train, y_train, X_test, y_test, n_bins = 10) {
  n_bins <- max(3, min(n_bins, floor(length(y_train) / 5)))
  edges  <- unique(quantile(y_train, probs = seq(0, 1, length.out = n_bins + 1), na.rm = TRUE))
  if (length(edges) < 3) return(NA)
  
  bin_train <- cut(y_train, breaks = edges, include.lowest = TRUE, labels = FALSE)
  bin_means <- tapply(y_train, bin_train, mean)
  
  clf <- tabpfn_pkg$TabPFNClassifier()
  clf$fit(as.matrix(X_train), as.integer(bin_train - 1))
  
  probs   <- clf$predict_proba(as.matrix(X_test))
  classes <- as.integer(clf$classes_) + 1
  pred    <- as.vector(probs %*% bin_means[classes])
  r2_score(pred, y_test)
}

run_tabpfn_reg_with_mae <- function(X_train, y_train, X_test, y_test, n_bins = 10) {
  n_bins <- max(3, min(n_bins, floor(length(y_train) / 5)))
  edges  <- unique(quantile(y_train, probs = seq(0, 1, length.out = n_bins + 1), na.rm = TRUE))
  if (length(edges) < 3) return(list(R2 = NA, MAE = NA))
  bin_train <- cut(y_train, breaks = edges, include.lowest = TRUE, labels = FALSE)
  bin_means <- tapply(y_train, bin_train, mean)
  clf <- tabpfn_pkg$TabPFNClassifier()
  clf$fit(as.matrix(X_train), as.integer(bin_train - 1))
  probs   <- clf$predict_proba(as.matrix(X_test))
  classes <- as.integer(clf$classes_) + 1
  pred    <- as.vector(probs %*% bin_means[classes])
  list(R2 = r2_score(pred, y_test), MAE = mae_score(pred, y_test))
}

# TabPFN classifier for Sex, called directly on the SAME X_sub/X_test as RF and XGBoost 
run_tabpfn_clf_direct <- function(X_train, y_train_bin, X_test, y_test_bin) {
  clf <- tabpfn_pkg$TabPFNClassifier()
  clf$fit(as.matrix(X_train), as.integer(y_train_bin))
  probs <- clf$predict_proba(as.matrix(X_test))[, 2]
  auc_score(probs, y_test_bin)
}

# 4. Outer loop: 5-fold CV. Inner loop: sample-size sweep

set.seed(42)
k <- 5
fold_id <- sample(rep(1:k, length.out = nrow(X)))

# Sample sizes based on the smallest fold's training set size
min_train_size <- min(nrow(X) - table(fold_id))
# learning curve loop
sample_sizes <- unique(pmin(c(50, 100, 150, 200, 250, 300, min_train_size), min_train_size))

raw_results <- data.frame()

for (fold in 1:k) {
  cat("=== Fold", fold, "of", k, "===\n")
  
  test_idx  <- which(fold_id == fold)
  train_idx <- which(fold_id != fold)
  
  X_train <- X[train_idx, ]; X_test <- X[test_idx, ]
  y_age_train <- y_age[train_idx]; y_age_test <- y_age[test_idx]
  y_bmi_train <- y_bmi[train_idx]; y_bmi_test <- y_bmi[test_idx]
  y_sex_train <- y_sex[train_idx]; y_sex_test <- y_sex[test_idx]
  
  for (n_size in sample_sizes) {
    cat("  n =", n_size, "...\n")
    set.seed(fold * 1000 + n_size)
    idx_sub <- sample(nrow(X_train), n_size)
    X_sub   <- X_train[idx_sub, ] # Age
    y_sub_age <- y_age_train[idx_sub]
    age_rf_out    <- train_regression(X_sub, y_sub_age, X_test, y_age_test)
    age_enet_out  <- run_enet_with_mae(X_sub, y_sub_age, X_test, y_age_test)
    age_xgb_out   <- run_xgb_with_mae(X_sub, y_sub_age, X_test, y_age_test)
    age_tabpfn_out <- run_tabpfn_reg_with_mae(X_sub, y_sub_age, X_test, y_age_test)
    raw_results <- rbind(raw_results,
                         data.frame(fold = fold, n = n_size, task = "Age", model = "Mean",
                                    value = run_mean_baseline(X_sub, y_sub_age, X_test, y_age_test), mae = mae_score(rep(mean(y_sub_age), length(y_age_test)), y_age_test)),
                         data.frame(fold = fold, n = n_size, task = "Age", model = "ElasticNet",
                                    value = age_enet_out$R2, mae = age_enet_out$MAE),
                         data.frame(fold = fold, n = n_size, task = "Age", model = "RF",
                                    value = age_rf_out$R2, mae = age_rf_out$MAE),
                         data.frame(fold = fold, n = n_size, task = "Age", model = "XGB",
                                    value = age_xgb_out$R2, mae = age_xgb_out$MAE),
                         data.frame(fold = fold, n = n_size, task = "Age", model = "TABPFN",
                                    value = age_tabpfn_out$R2, mae = age_tabpfn_out$MAE))
    
    # BMI 
    y_sub_bmi <- y_bmi_train[idx_sub]
    bmi_rf_out     <- train_regression(X_sub, y_sub_bmi, X_test, y_bmi_test)
    bmi_enet_out   <- run_enet_with_mae(X_sub, y_sub_bmi, X_test, y_bmi_test)
    bmi_xgb_out    <- run_xgb_with_mae(X_sub, y_sub_bmi, X_test, y_bmi_test)
    bmi_tabpfn_out <- run_tabpfn_reg_with_mae(X_sub, y_sub_bmi, X_test, y_bmi_test)
    raw_results <- rbind(raw_results,
                         data.frame(fold = fold, n = n_size, task = "BMI", model = "Mean",
                                    value = run_mean_baseline(X_sub, y_sub_bmi, X_test, y_bmi_test), mae = mae_score(rep(mean(y_sub_bmi), length(y_bmi_test)), y_bmi_test)),
                         data.frame(fold = fold, n = n_size, task = "BMI", model = "ElasticNet",
                                    value = bmi_enet_out$R2, mae = bmi_enet_out$MAE),
                         data.frame(fold = fold, n = n_size, task = "BMI", model = "RF",
                                    value = bmi_rf_out$R2, mae = bmi_rf_out$MAE),
                         data.frame(fold = fold, n = n_size, task = "BMI", model = "XGB",
                                    value = bmi_xgb_out$R2, mae = bmi_xgb_out$MAE),
                         data.frame(fold = fold, n = n_size, task = "BMI", model = "TABPFN",
                                    value = bmi_tabpfn_out$R2, mae = bmi_tabpfn_out$MAE))
    
    # Sex 
    y_sub_sex      <- y_sex_train[idx_sub]
    y_sub_sex_fac  <- factor(y_sub_sex, levels = c("male", "female"))
    y_sub_sex_bin  <- as.integer(y_sub_sex == "female")
    y_test_sex_bin <- as.integer(y_sex_test == "female")
    
    raw_results <- rbind(raw_results,
                         data.frame(fold = fold, n = n_size, task = "Sex", model = "RF",
                                    value = run_rf_clf(X_sub, y_sub_sex_fac, X_test, y_test_sex_bin), mae = NA),
                         data.frame(fold = fold, n = n_size, task = "Sex", model = "XGB",
                                    value = run_xgb(X_sub, y_sub_sex_bin, X_test, y_test_sex_bin, classification = TRUE), mae = NA))
    
  
    tryCatch({
      tabpfn_sex_auc <- run_tabpfn_clf_direct(X_sub, y_sub_sex_bin, X_test, y_test_sex_bin)
      raw_results <- rbind(raw_results,
                           data.frame(fold = fold, n = n_size, task = "Sex", model = "TABPFN", value = tabpfn_sex_auc, mae = NA))
    }, error = function(e) {
      cat("    Sex/TabPFN skipped - fold", fold, "n =", n_size, "-", conditionMessage(e), "\n")
    })
  }
}
#  5. calculating the aggregate: mean +/- SD 

lc_summary <- raw_results |>
  filter(!is.na(value)) |>
  group_by(n, task, model) |>
  summarise(mean_value = mean(value), sd_value = sd(value),
            mean_mae = mean(mae, na.rm = TRUE), .groups = "drop")

print(lc_summary)

#  5b. SIGNIFICANCE TESTING (permutation test, CV-consistent)

run_cv_permutation_test <- function(y, model_fn, n_perm = 40) {
  real_scores <- numeric(k)
  for (fold in 1:k) {
    te <- which(fold_id == fold); tr <- which(fold_id != fold)
    real_scores[fold] <- model_fn(X[tr, ], y[tr], X[te, ], y[te])
  }
  real_value <- mean(real_scores, na.rm = TRUE)
  
  perm_values <- numeric(n_perm)
  for (i in 1:n_perm) {
    y_shuffled <- sample(y)
    fold_scores <- numeric(k)
    for (fold in 1:k) {
      te <- which(fold_id == fold); tr <- which(fold_id != fold)
      fold_scores[fold] <- model_fn(X[tr, ], y_shuffled[tr], X[te, ], y_shuffled[te])
    }
    perm_values[i] <- mean(fold_scores, na.rm = TRUE)
  }
  
  p_value <- (sum(perm_values >= real_value) + 1) / (n_perm + 1)
  list(real_value = real_value, perm_mean = mean(perm_values), p_value = p_value)
}

xgb_reg_fn <- function(X_train, y_train, X_test, y_test) run_xgb(X_train, y_train, X_test, y_test)
xgb_clf_fn <- function(X_train, y_train, X_test, y_test) run_xgb(X_train, y_train, X_test, y_test, classification = TRUE)

rf_reg_fn  <- function(X_train, y_train, X_test, y_test) train_regression(X_train, y_train, X_test, y_test)$R2
rf_clf_fn  <- function(X_train, y_train, X_test, y_test) {
  y_train_fac <- factor(y_train)   # only classes actually present in this draw,
  # avoids the same empty-class crash fixed earlier
  model <- randomForest(x = X_train, y = y_train_fac, ntree = 500)
  probs_matrix <- predict(model, X_test, type = "prob")
  if (!"1" %in% colnames(probs_matrix)) {
    probs <- rep(0.5, nrow(X_test))
  } else {
    probs <- probs_matrix[, "1"]
  }
  auc_score(probs, y_test)
}

tabpfn_reg_fn <- function(X_train, y_train, X_test, y_test) run_tabpfn_reg(X_train, y_train, X_test, y_test)

tabpfn_clf_fn <- run_tabpfn_clf_direct

cat("\nRunning CV-consistent significance tests ..\n")
sig_results_full <- list()
sig_results_full$age_rf     <- run_cv_permutation_test(y_age, rf_reg_fn, n_perm = 40)
sig_results_full$age_xgb    <- run_cv_permutation_test(y_age, xgb_reg_fn, n_perm = 40)
sig_results_full$age_tabpfn <- run_cv_permutation_test(y_age, tabpfn_reg_fn, n_perm = 15)
sig_results_full$bmi_rf     <- run_cv_permutation_test(y_bmi, rf_reg_fn, n_perm = 40)
sig_results_full$bmi_xgb    <- run_cv_permutation_test(y_bmi, xgb_reg_fn, n_perm = 40)
sig_results_full$bmi_tabpfn <- run_cv_permutation_test(y_bmi, tabpfn_reg_fn, n_perm = 15)
sig_results_full$sex_rf     <- run_cv_permutation_test(y_sex_bin, rf_clf_fn, n_perm = 40)
sig_results_full$sex_xgb    <- run_cv_permutation_test(y_sex_bin, xgb_clf_fn, n_perm = 40)
sig_results_full$sex_tabpfn <- run_cv_permutation_test(y_sex_bin, tabpfn_clf_fn, n_perm = 15)

cat("\n=== FULL SIGNIFICANCE TEST SUMMARY (all 3 models, CV-consistent) ===\n")
for (name in names(sig_results_full)) {
  r <- sig_results_full[[name]]
  cat(sprintf("%-12s real=%.3f  chance_mean=%.3f  p=%.4f  %s\n",
              name, r$real_value, r$perm_mean, r$p_value,
              ifelse(r$p_value < 0.05, "SIGNIFICANT", "not significant")))
}

#  5c. CUT-POINT CLASSIFICATION (CV-consistent)
# FIX: same CV-consistency logic applied here as in 5b above.

age_decade <- cut(merged$age_years,
                  breaks = c(18, 30, 40, 50, 60, 70, Inf),
                  labels = c("18-29", "30-39", "40-49", "50-59", "60-69", "70+"),
                  right = FALSE)

bmi_class <- cut(merged$bmi,
                 breaks = c(-Inf, 18.5, 25, 30, Inf),
                 labels = c("Underweight", "Normal", "Overweight", "Obese"),
                 right = FALSE)

cat("\nAge decade distribution:\n"); print(table(age_decade))
cat("\nBMI class distribution:\n");  print(table(bmi_class))

run_cv_multiclass_perm_test <- function(labels, model_fn, n_perm = 20) {
  labels <- droplevels(as.factor(labels))
  y_int  <- as.integer(labels) - 1
  n_class <- length(levels(labels))
  chance_level <- 1 / n_class
  
  real_scores <- numeric(k)
  for (fold in 1:k) {
    te <- which(fold_id == fold); tr <- which(fold_id != fold)
    real_scores[fold] <- model_fn(X[tr, ], y_int[tr], X[te, ], y_int[te], n_class)
  }
  real_acc <- mean(real_scores, na.rm = TRUE)
  
  perm_acc <- numeric(n_perm)
  for (i in 1:n_perm) {
    y_shuffled <- sample(y_int)
    fold_scores <- numeric(k)
    for (fold in 1:k) {
      te <- which(fold_id == fold); tr <- which(fold_id != fold)
      fold_scores[fold] <- model_fn(X[tr, ], y_shuffled[tr], X[te, ], y_shuffled[te], n_class)
    }
    perm_acc[i] <- mean(fold_scores, na.rm = TRUE)
  }
  
  p_value <- (sum(perm_acc >= real_acc) + 1) / (n_perm + 1)
  list(real_acc = real_acc, chance_level = chance_level,
       perm_mean = mean(perm_acc), p_value = p_value, n_class = n_class)
}

## Model wrappers, each returning accuracy given train/test int labels

xgb_multi_fn <- function(X_train, y_train, X_test, y_test, n_class) {
  dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
  dtest  <- xgb.DMatrix(data = as.matrix(X_test))
  model <- xgb.train(
    params = list(objective = "multi:softmax", num_class = n_class, max_depth = 3, eta = 0.1),
    data = dtrain, nrounds = 100, verbose = 0
  )
  pred <- predict(model, dtest)
  mean(pred == y_test)
}

rf_multi_fn <- function(X_train, y_train, X_test, y_test, n_class) {
  y_train_fac <- factor(y_train)   # only classes actually present in this training draw
  model <- randomForest(x = X_train, y = y_train_fac, ntree = 500)
  pred_int <- as.integer(as.character(predict(model, X_test)))
  mean(pred_int == y_test)
}

tabpfn_multi_fn <- function(X_train, y_train, X_test, y_test, n_class) {
  clf <- tabpfn_pkg$TabPFNClassifier()
  clf$fit(as.matrix(X_train), as.integer(y_train))
  pred <- as.integer(clf$predict(as.matrix(X_test)))
  mean(pred == y_test)
}

## Running all 6 combinations (2 cut-point tasks x 3 models), CV-consistent

cutpoint_results_full <- list()

cat("Age decade: RF, XGBoost...\n")
cutpoint_results_full$age_decade_rf  <- run_cv_multiclass_perm_test(age_decade, rf_multi_fn, n_perm = 20)
cutpoint_results_full$age_decade_xgb <- run_cv_multiclass_perm_test(age_decade, xgb_multi_fn, n_perm = 20)
cat("Age decade: TabPFN (reduced n_perm)...\n")
cutpoint_results_full$age_decade_tabpfn <- run_cv_multiclass_perm_test(age_decade, tabpfn_multi_fn, n_perm = 10)

cat("BMI class: RF, XGBoost...\n")
cutpoint_results_full$bmi_class_rf  <- run_cv_multiclass_perm_test(bmi_class, rf_multi_fn, n_perm = 20)
cutpoint_results_full$bmi_class_xgb <- run_cv_multiclass_perm_test(bmi_class, xgb_multi_fn, n_perm = 20)
cat("BMI class: TabPFN (reduced n_perm)...\n")
cutpoint_results_full$bmi_class_tabpfn <- run_cv_multiclass_perm_test(bmi_class, tabpfn_multi_fn, n_perm = 10)

cat("\n=== CUT-POINT CLASSIFICATION: ALL 3 MODELS (CV-consistent) ===\n")
for (name in names(cutpoint_results_full)) {
  r <- cutpoint_results_full[[name]]
  cat(sprintf("%-18s acc=%.3f  naive_chance=%.3f  perm_chance=%.3f  p=%.4f  %s\n",
              name, r$real_acc, r$chance_level, r$perm_mean, r$p_value,
              ifelse(r$p_value < 0.05, "SIGNIFICANT", "not significant")))
}

# 5d. MULTIPLE-TESTING CORRECTION
# FIX: applies Bonferroni and Benjamini-Hochberg correction across every
# significance test run above (9 from 5b + 6 from 5c = 15 tests total).

all_p_named <- c(
  sapply(sig_results_full, function(r) r$p_value),
  sapply(cutpoint_results_full, function(r) r$p_value)
)

correction_table <- data.frame(
  test = names(all_p_named),
  raw_p = as.numeric(all_p_named)
)
correction_table$bonferroni_p <- p.adjust(correction_table$raw_p, method = "bonferroni")
correction_table$BH_p         <- p.adjust(correction_table$raw_p, method = "BH")
correction_table$bonferroni_sig <- correction_table$bonferroni_p < 0.05
correction_table$BH_sig         <- correction_table$BH_p < 0.05

cat("\n=== MULTIPLE-TESTING CORRECTION (across all 15 tests) ===\n")
print(correction_table)

# 6. plotting the results

reg_data <- lc_summary |> filter(task %in% c("Age", "BMI"))
sex_data <- lc_summary |> filter(task == "Sex")

p_reg <- ggplot(reg_data, aes(x = n, y = mean_value, colour = model, fill = model)) +
  geom_ribbon(aes(ymin = mean_value - sd_value, ymax = mean_value + sd_value),
              alpha = 0.15, colour = NA) +
  geom_line(linewidth = 1) +
  geom_point(size = 2.5) +
  facet_wrap(~task, scales = "free_y") +
  theme_minimal() +
  labs(x = "Training sample size",
       y = expression(R^2 * " (test set; 0 = mean-baseline, negative = worse than mean-baseline)"),
       colour = "Model", fill = "Model") +
  theme(legend.position = "bottom")

p_sex <- ggplot(sex_data, aes(x = n, y = mean_value, colour = model, fill = model)) +
  geom_ribbon(aes(ymin = mean_value - sd_value, ymax = mean_value + sd_value),
              alpha = 0.15, colour = NA) +
  geom_line(linewidth = 1) +
  geom_point(size = 2.5) +
  facet_wrap(~task) +
  theme_minimal() +
  labs(x = "Training sample size",
       y = "AUC (test set; 0.5 = random, 1.0 = perfect)",
       colour = "Model", fill = "Model") +
  theme(legend.position = "bottom")

combined_plot <- (p_reg | p_sex) +
  plot_annotation(
    title = "Learning curves by sample size, cross-validated across 5 folds (mean +/- SD)"
  )

print(combined_plot)

ggsave(file.path(data_dir, "Comparison_model_schirmer_5foldCV_2.jpg"), combined_plot,
       width = 12, height = 6, dpi = 300)
