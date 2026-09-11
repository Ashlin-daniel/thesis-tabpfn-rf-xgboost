## ============================================================
## REAL-DATA SANITY CHECKS (no simulated/fake data at all)
## ============================================================
## Run this AFTER the main script's data loading section, so that
## `X` (CLR-transformed real taxa) and `merged` (real metadata) already
## exist in the R session.
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
