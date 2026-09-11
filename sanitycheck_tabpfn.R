## ============================================================
## SIMPLE STANDALONE TabPFN SANITY CHECK
## ============================================================
## It only tests: "can TabPFN find an obvious, planted age signal?"
## (RF and XGBoost already passed this test, so this fills the gap.)
## ============================================================

library(reticulate)

data_dir   <- Sys.getenv("THESIS_DATA_DIR",   unset = ".")
venv_path  <- Sys.getenv("THESIS_VENV_PATH",  unset = "./venv311")
script_dir <- Sys.getenv("THESIS_SCRIPT_DIR", unset = ".")

use_virtualenv(venv_path)
py_run_string("import warnings; warnings.filterwarnings('ignore')")
tabpfn_pkg <- import("tabpfn")

## ------------------------------------------------------------
## Loading ONLY the synthetic data 
## ------------------------------------------------------------
synthetic_check_data <- read.csv(file.path(data_dir, "synthetic_data_for_tabpfn_check.csv"))
synthetic_check_data <- synthetic_check_data[complete.cases(synthetic_check_data), ]

X_all <- synthetic_check_data[, grep("^taxon_", colnames(synthetic_check_data))]
y_age <- synthetic_check_data$age

# TabPFN accepts at most 100 features. We MUST make sure the taxa that
# actually carry the planted signal survive the cut -- otherwise we'd
# get a false FAIL that has nothing to do with your pipeline.
# These are the taxon numbers printed earlier when the signal was planted:
known_signal_taxa <- paste0("taxon_", c(195, 120, 197, 69, 93, 150, 103, 122, 80, 166))
known_signal_taxa <- intersect(known_signal_taxa, colnames(X_all))  # keep only ones that exist

remaining_slots <- 100 - length(known_signal_taxa)
other_taxa_by_var <- names(sort(apply(X_all[, setdiff(colnames(X_all), known_signal_taxa)], 2, var),
                                decreasing = TRUE))[1:remaining_slots]

top_taxa <- c(known_signal_taxa, other_taxa_by_var)
X <- X_all[, top_taxa]

cat("Rows:", nrow(X), " | Taxa columns (capped at 100):", ncol(X), "\n")
cat("Any NA in y_age?", any(is.na(y_age)), "\n")

## ------------------------------------------------------------
## r2 helper
## ------------------------------------------------------------
r2_score <- function(pred, truth) {
  1 - sum((pred - truth)^2) / sum((truth - mean(truth))^2)
}

## ------------------------------------------------------------
## TabPFN regression via quantile binning 
## ------------------------------------------------------------
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

## ------------------------------------------------------------
## Simple single train/test split (80/20) 
## ------------------------------------------------------------
set.seed(1)
n         <- nrow(X)
test_idx  <- sample(1:n, size = round(0.2 * n))
train_idx <- setdiff(1:n, test_idx)

X_train <- X[train_idx, ]; X_test <- X[test_idx, ]
y_train <- y_age[train_idx]; y_test <- y_age[test_idx]

cat("\nRunning TabPFN on synthetic data (planted age signal)...\n")
r2 <- run_tabpfn_reg(X_train, y_train, X_test, y_test)

cat("\n========================================================\n")
cat(sprintf("TabPFN R^2 on synthetic (known-signal) data: %.3f\n", r2))
if (!is.na(r2) && r2 > 0.3) {
  cat("PASS: TabPFN correctly detects a known, planted signal.\n")
} else {
  cat("FAIL: TabPFN could not find an obvious signal -- worth\n")
  cat("checking the quantile-binning code itself.\n")
}
cat("========================================================\n")
