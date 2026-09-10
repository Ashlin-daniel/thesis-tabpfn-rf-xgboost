
## ============================================================
## PERMUTATION TEST ON THE SYNTHETIC (KNOWN-SIGNAL) DATA
## ============================================================
## PURPOSE: Earlier, we said "R^2 = 0.55-0.69 looks clearly non-zero,
## so PASS." That was a rule-of-thumb eyeball check, not proof.
## This script properly PROVES it, the same way we proved the real
## data's null result -- by comparing against a shuffled-label
## chance distribution.
## ============================================================

library(xgboost)

r2_score <- function(pred, truth) {
  1 - sum((pred - truth)^2) / sum((truth - mean(truth))^2)
}

run_xgb_r2 <- function(X_train, y_train, X_test, y_test) {
  dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
  dtest  <- xgb.DMatrix(data = as.matrix(X_test))
  model <- xgb.train(
    params = list(objective = "reg:squarederror", max_depth = 3, eta = 0.1),
    data = dtrain, nrounds = 100, verbose = 0
  )
  pred <- predict(model, dtest)
  r2_score(pred, y_test)
}

run_permutation_test <- function(X, y, model_fn, n_perm = 200, test_frac = 0.2, seed = 1) {
  set.seed(seed)
  n <- nrow(X)
  test_idx  <- sample(1:n, size = round(test_frac * n))
  train_idx <- setdiff(1:n, test_idx)
  
  real_r2 <- model_fn(X[train_idx, ], y[train_idx], X[test_idx, ], y[test_idx])
  cat(sprintf("Real R^2 (correct labels): %.4f\n", real_r2))
  
  cat(sprintf("Running %d permutations...\n", n_perm))
  perm_r2 <- numeric(n_perm)
  for (i in 1:n_perm) {
    y_shuffled <- sample(y)
    perm_r2[i] <- model_fn(X[train_idx, ], y_shuffled[train_idx],
                           X[test_idx, ], y_shuffled[test_idx])
    if (i %% 20 == 0) cat("  permutation", i, "of", n_perm, "\n")
  }
  
  p_value <- (sum(perm_r2 >= real_r2) + 1) / (n_perm + 1)
  
  cat("\n========================================================\n")
  cat(sprintf("Real R^2:              %.4f\n", real_r2))
  cat(sprintf("Chance R^2 (mean):     %.4f\n", mean(perm_r2)))
  cat(sprintf("Chance R^2 (95th pct): %.4f\n", quantile(perm_r2, 0.95)))
  cat(sprintf("p-value:               %.4f\n", p_value))
  if (p_value < 0.05) {
    cat("PASS CONFIRMED (statistically): synthetic signal is significantly\n")
    cat("above chance -- this formally proves the pipeline detects real signal.\n")
  } else {
    cat("PASS NOT CONFIRMED: even the planted signal wasn't statistically\n")
    cat("significant -- worth re-checking the synthetic data generation.\n")
  }
  cat("========================================================\n")
  
  hist(perm_r2, breaks = 30, col = "grey80",
       main = "Synthetic data: chance R^2 distribution",
       xlab = "R^2", xlim = range(c(perm_r2, real_r2)))
  abline(v = real_r2, col = "red", lwd = 2)
  legend("topright", legend = "Real R^2 (synthetic data)", col = "red", lwd = 2, bty = "n")
  
  list(real_r2 = real_r2, perm_r2 = perm_r2, p_value = p_value)
}

## ------------------------------------------------------------
## RUN IT: load the synthetic data and test
## ------------------------------------------------------------
synthetic_check_data <- read.csv("synthetic_data_for_tabpfn_check.csv")
synthetic_check_data <- synthetic_check_data[complete.cases(synthetic_check_data), ]

X_syn <- synthetic_check_data[, grep("^taxon_", colnames(synthetic_check_data))]
y_syn <- synthetic_check_data$age

result_synthetic <- run_permutation_test(X_syn, y_syn, run_xgb_r2, n_perm = 200)

