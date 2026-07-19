
# learning curves by sample size, cross-validated across 5 folds RF, XGBoost, and TabPFN on all 3 tasks (Age, BMI, Sex)
# the outer loop is 5-fold CV - each fold takes a turn as the held-out test set.
#  The inner loop sweeps training sample size within each fold. 
# Because every sample size is run once per fold, the 5 folds themselves provide the repeated estimates (mean +/- SD) .

library(reticulate)
library(dplyr)
library(tidyr)
library(xgboost)
library(randomForest)
library(ggplot2)
library(patchwork)  



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

abund_wide <- abund |>
  pivot_wider(names_from = clade_name, values_from = rel_abund, values_fill = 0)

merged <- inner_join(meta, abund_wide, by = "sample_alias") |>
  filter(age_years >= 18, !is.na(age_years), !is.na(bmi), sex %in% c("male", "female"))

cat("Samples after filtering:", nrow(merged), "\n")

# 2. feature matrix 
# capping at 100 taxa so TabPFN can accept it

taxa_cols <- setdiff(colnames(merged), colnames(meta))
X_full    <- merged[, taxa_cols]

prevalent <- X_full[, colMeans(X_full > 0) >= 0.10]
top_taxa  <- names(sort(apply(prevalent, 2, var), decreasing = TRUE))[1:min(100, ncol(prevalent))]
X_raw     <- prevalent[, top_taxa]

# CLR (centered log-ratio) transform, instead of log1p.

clr_transform <- function(X) {
  X <- as.matrix(X)
  pseudocount <- min(X[X > 0], na.rm = TRUE) / 2
  X[X == 0] <- pseudocount
  log_X <- log(X)
  clr_X <- log_X - rowMeans(log_X)
  as.data.frame(clr_X)
}
X <- clr_transform(X_raw)

y_age <- merged$age_years
y_bmi <- merged$bmi
y_sex <- merged$sex

cat("Feature matrix:", nrow(X), "samples x", ncol(X), "taxa\n")

# 3. helper functions

r2_score <- function(pred, truth) {
  1 - sum((pred - truth)^2) / sum((truth - mean(truth))^2)
}
# manual AUC
auc_score <- function(probs, actual_binary) {
  n1 <- sum(actual_binary == 1); n0 <- sum(actual_binary == 0)
  r  <- rank(probs)
  (sum(r[actual_binary == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# one XGBoost helper for both regression and classification.

run_xgb <- function(X_train, y_train, X_test, y_test, classification = FALSE) {
  y_fit <- if (classification) factor(y_train, levels = c(0, 1)) else y_train
  model <- xgboost(x = as.matrix(X_train), y = y_fit,
                   nrounds = 100, max_depth = 3, learning_rate = 0.1,
                   subsample = 0.8, colsample_bytree = 0.8, verbosity = 0)
  pred <- predict(model, as.matrix(X_test))
  if (classification) auc_score(pred, y_test) else r2_score(pred, y_test)
}

# Random Forest classifier (for Sex)

run_rf_clf <- function(X_train, y_train_fac, X_test, y_test_bin) {
  model <- randomForest(x = X_train, y = y_train_fac, ntree = 500)
  probs <- predict(model, X_test, type = "prob")[, "female"]
  auc_score(probs, y_test_bin)
}

# TabPFN regression via quantile binning:
#  1. cutting y_train into ~n_bins quantile buckets
#  2. fitting TabPFNClassifier to predict bucket membership
#  3. reconstructing a continuous prediction as the probability-weighted average of each bucket's mean y-value (not a true continuous output — TabPFN 0.1.9 has no native regressor

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
    X_sub   <- X_train[idx_sub, ]
    
    # Age
    y_sub_age <- y_age_train[idx_sub]
    raw_results <- rbind(raw_results,
                         data.frame(fold = fold, n = n_size, task = "Age", model = "RF",
                                    value = train_regression(X_sub, y_sub_age, X_test, y_age_test)$R2),
                         data.frame(fold = fold, n = n_size, task = "Age", model = "XGB",
                                    value = run_xgb(X_sub, y_sub_age, X_test, y_age_test)),
                         data.frame(fold = fold, n = n_size, task = "Age", model = "TABPFN",
                                    value = run_tabpfn_reg(X_sub, y_sub_age, X_test, y_age_test)))
    
    # BMI 
    y_sub_bmi <- y_bmi_train[idx_sub]
    raw_results <- rbind(raw_results,
                         data.frame(fold = fold, n = n_size, task = "BMI", model = "RF",
                                    value = train_regression(X_sub, y_sub_bmi, X_test, y_bmi_test)$R2),
                         data.frame(fold = fold, n = n_size, task = "BMI", model = "XGB",
                                    value = run_xgb(X_sub, y_sub_bmi, X_test, y_bmi_test)),
                         data.frame(fold = fold, n = n_size, task = "BMI", model = "TABPFN",
                                    value = run_tabpfn_reg(X_sub, y_sub_bmi, X_test, y_bmi_test)))
    
    # Sex 
    y_sub_sex      <- y_sex_train[idx_sub]
    y_sub_sex_fac  <- factor(y_sub_sex, levels = c("male", "female"))
    y_sub_sex_bin  <- as.integer(y_sub_sex == "female")
    y_test_sex_bin <- as.integer(y_sex_test == "female")
    
    raw_results <- rbind(raw_results,
                         data.frame(fold = fold, n = n_size, task = "Sex", model = "RF",
                                    value = run_rf_clf(X_sub, y_sub_sex_fac, X_test, y_test_sex_bin)),
                         data.frame(fold = fold, n = n_size, task = "Sex", model = "XGB",
                                    value = run_xgb(X_sub, y_sub_sex_bin, X_test, y_test_sex_bin, classification = TRUE)))
    
    X_combined     <- rbind(X_sub, X_test)
    y_combined_sex <- c(y_sub_sex, y_sex_test)
    tryCatch({
      tabpfn_sex <- train_classifier(X_combined, y_combined_sex, pos_label = "female",
                                     test_size = nrow(X_test) / nrow(X_combined))
      raw_results <- rbind(raw_results,
                           data.frame(fold = fold, n = n_size, task = "Sex", model = "TABPFN", value = tabpfn_sex$auc))
    }, error = function(e) {
      cat("    Sex/TabPFN skipped - fold", fold, "n =", n_size, "-", conditionMessage(e), "\n")
    })
  }
}

#  5. calculating the aggregate: mean +/- SD 

lc_summary <- raw_results |>
  filter(!is.na(value)) |>
  group_by(n, task, model) |>
  summarise(mean_value = mean(value), sd_value = sd(value), .groups = "drop")

print(lc_summary)

#  6. plotting the results

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
       y = expression(R^2 * " (test set; 0 = mean-baseline, can be negative)"),
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

ggsave(file.path(data_dir, "Comparison_model_schirmer_5foldCV.jpg"), combined_plot,
       width = 12, height = 6, dpi = 300)