## ============================================================
## REAL-DATA SANITY CHECKS (no simulated/fake data at all)
## ============================================================
## Run this AFTER your main script's data loading section, so that
## `X` (CLR-transformed real taxa) and `merged` (real metadata) already
## exist in your R session.
## ============================================================

r2_score <- function(pred, truth) {
  1 - sum((pred - truth)^2) / sum((truth - mean(truth))^2)
}

## ------------------------------------------------------------
## OPTION A: Extreme-group selection
## ------------------------------------------------------------
## Logic: predicting EXACT age is hard. Telling a young person from an
## old person should be much easier -- there's more biological distance
## between them. We keep only the youngest 25% and oldest 25% of your
## REAL people (no invented data), turn this into a simple TWO-GROUP
## task, and check whether the model can at least do the "easy" version.

age_q <- quantile(merged$age_years, probs = c(0.25, 0.75))
extreme_idx <- which(merged$age_years <= age_q[1] | merged$age_years >= age_q[2])

cat("Extreme-group check: keeping", length(extreme_idx), "of", nrow(merged),
    "real people (youngest 25% + oldest 25%)\n")

X_extreme <- X[extreme_idx, ]
# Turn into a yes/no label: is this person in the OLDER extreme group?
y_extreme_group <- as.integer(merged$age_years[extreme_idx] >= age_q[2])

# Simple 80/20 split and an XGBoost classifier as a quick check
set.seed(1)
n <- nrow(X_extreme)
test_idx  <- sample(1:n, size = round(0.2 * n))
train_idx <- setdiff(1:n, test_idx)

library(xgboost)
dtrain <- xgb.DMatrix(data = as.matrix(X_extreme[train_idx, ]), label = y_extreme_group[train_idx])
dtest  <- xgb.DMatrix(data = as.matrix(X_extreme[test_idx, ]))

model <- xgb.train(
  params = list(objective = "binary:logistic", max_depth = 3, eta = 0.1),
  data = dtrain, nrounds = 100, verbose = 0
)

probs <- predict(model, dtest)
truth <- y_extreme_group[test_idx]

# manual AUC (same style as your real pipeline's auc_score function)
auc_score <- function(probs, actual_binary) {
  n1 <- sum(actual_binary == 1); n0 <- sum(actual_binary == 0)
  r  <- rank(probs)
  (sum(r[actual_binary == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

auc <- auc_score(probs, truth)
cat(sprintf("Extreme-group (young vs old) AUC: %.3f  (0.5 = chance, 1.0 = perfect)\n", auc))

## ------------------------------------------------------------
## OPTION B: Predict a value computed directly from the real taxa
## ------------------------------------------------------------
## Logic: Shannon diversity is a standard number calculated directly
## FROM the abundance data itself (how evenly spread a person's gut
## bacteria are). It is not fabricated or planted -- it's arithmetic
## on the real numbers already in X_raw. A working pipeline should be
## able to predict it almost perfectly, since it's derived from the
## same columns being used to predict it.

# X_raw = the real relative-abundance data BEFORE the CLR transform
# (this should already exist from your main script's step 2)
shannon_diversity <- function(x) {
  p <- x / sum(x)   # normalize to proportions summing to 1 (fixes the scale bug)
  p <- p[p > 0]
  -sum(p * log(p))
}
y_shannon <- apply(X_raw, 1, shannon_diversity)

cat("\nShannon diversity check (computed directly from real taxa data)\n")
cat("Range of Shannon diversity values:", round(range(y_shannon), 2), "\n")

set.seed(2)
test_idx2  <- sample(1:nrow(X), size = round(0.2 * nrow(X)))
train_idx2 <- setdiff(1:nrow(X), test_idx2)

dtrain2 <- xgb.DMatrix(data = as.matrix(X[train_idx2, ]), label = y_shannon[train_idx2])
dtest2  <- xgb.DMatrix(data = as.matrix(X[test_idx2, ]))

model2 <- xgb.train(
  params = list(objective = "reg:squarederror", max_depth = 3, eta = 0.1),
  data = dtrain2, nrounds = 100, verbose = 0
)

pred2 <- predict(model2, dtest2)
r2_shannon <- r2_score(pred2, y_shannon[test_idx2])

cat(sprintf("Shannon diversity R^2 (should be very high, e.g. >0.8): %.3f\n", r2_shannon))

if (r2_shannon > 0.8) {
  cat("PASS: pipeline correctly recovers a value computed directly from its own input.\n")
} else {
  cat("FAIL: this should be nearly perfect -- worth checking data alignment/leakage.\n")
}