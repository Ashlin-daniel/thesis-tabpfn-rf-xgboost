## ============================================================
## SYNTHETIC DATA SANITY CHECK (FULL: fully-synthetic + semi-synthetic)
## ============================================================
## PURPOSE (plain language):
## PART A (Steps 1-7): build FULLY FAKE microbiome data where WE decide,
## in advance, that age has a real, strong relationship with a handful
## of "taxa". Then run RF + XGBoost and check: can they find the planted
## signal? This tests whether your CODE works at all.
##
## PART B (Step 8): use your REAL feature matrix, but inject a KNOWN
## signal at a chosen strength (target R^2 = 0.7 down to 0.05). This
## tests whether your pipeline can still detect signal once your
## data's real structure (real taxa correlations, sparsity) is involved,
## and finds your "detection floor" - the smallest true effect your
## pipeline can still recover in this exact 471-sample cohort.
## ============================================================

set.seed(42)  # so results are reproducible every time you run this

# install.packages("gtools")     # uncomment if you don't have this package
# install.packages("randomForest")
# install.packages("caret")
# install.packages("xgboost")
library(gtools)
library(randomForest)
library(caret)
library(xgboost)


## ============================================================
## PART A: FULLY SYNTHETIC CHECK (your original steps 1-7)
## ============================================================

## ------------------------------------------------------------
## STEP 1: Simulate fake compositional microbiome data
## ------------------------------------------------------------
## Real microbiome data is "compositional" (percentages that sum to 100%
## per person). We fake that structure here using a Dirichlet distribution,
## which is the standard way to simulate compositional data.

n_samples <- 471   # same sample size as your real Schirmer_2016_500FG data
n_taxa    <- 200   # number of fake "species"

# Random "baseline abundance" for each taxon (some common, some rare)
baseline <- rgamma(n_taxa, shape = 0.5, rate = 1)
baseline <- baseline / sum(baseline)

# Draw each person's microbiome composition from a Dirichlet distribution
# centered on that baseline (this mimics natural person-to-person variation)
alpha <- baseline * 50  # concentration parameter; controls how "spread out" people are
comp_data <- rdirichlet(n_samples, alpha)  # n_samples x n_taxa, each row sums to 1
colnames(comp_data) <- paste0("taxon_", 1:n_taxa)

## ------------------------------------------------------------
## STEP 2: CLR transform (same as your real pipeline)
## ------------------------------------------------------------
## CLR = Centered Log-Ratio. It's the standard way to handle compositional
## data so models don't get confused by the "sums to 100%" constraint.

clr_transform <- function(x, pseudocount = 1e-6) {
  x <- x + pseudocount
  log_x <- log(x)
  sweep(log_x, 1, rowMeans(log_x), "-")
}

clr_data <- clr_transform(comp_data)

## ------------------------------------------------------------
## STEP 3: PLANT a known, true age signal
## ------------------------------------------------------------
## We pick 10 taxa at random and declare: "age depends on these."
## This is the ground truth we are hiding inside the fake data.

true_signal_taxa <- sample(1:n_taxa, 10)
true_weights     <- runif(10, min = -3, max = 3)  # random strength/direction per taxon

cat("Taxa carrying the TRUE age signal (ground truth):\n")
print(true_signal_taxa)

# Build age from those taxa + some random noise (nothing is ever 100% predictable)
signal_part <- clr_data[, true_signal_taxa] %*% true_weights
signal_part <- as.vector(scale(signal_part))  # standardize the signal

noise <- rnorm(n_samples, mean = 0, sd = 1)   # random noise, same scale as signal

# Combine: 70% real signal, 30% noise -> this should be an EASY, strong signal
age_raw <- 0.7 * signal_part + 0.3 * noise
age <- 45 + age_raw * 15          # rescale to look like realistic ages
age <- pmin(pmax(age, 18), 80)    # clip to a plausible human age range

hist(age, main = "Simulated age distribution", xlab = "age")

## ------------------------------------------------------------
## STEP 4: Assemble the final synthetic dataset
## ------------------------------------------------------------
synthetic_df <- as.data.frame(clr_data)
synthetic_df$age <- age

cat("\nSynthetic dataset ready:", nrow(synthetic_df), "samples,",
    n_taxa, "CLR-transformed taxa, 1 age outcome.\n\n")

## ------------------------------------------------------------
## STEP 5: Run Random Forest with 5-fold CV (mirrors your real pipeline)
## ------------------------------------------------------------
set.seed(123)
folds <- createFolds(synthetic_df$age, k = 5)

rf_r2_per_fold <- c()

for (i in seq_along(folds)) {
  test_idx  <- folds[[i]]
  train_idx <- setdiff(1:nrow(synthetic_df), test_idx)
  
  train_data <- synthetic_df[train_idx, ]
  test_data  <- synthetic_df[test_idx, ]
  
  rf_model <- randomForest(age ~ ., data = train_data, ntree = 500)
  preds    <- predict(rf_model, newdata = test_data)
  
  ss_res <- sum((test_data$age - preds)^2)
  ss_tot <- sum((test_data$age - mean(test_data$age))^2)
  r2     <- 1 - (ss_res / ss_tot)
  
  rf_r2_per_fold <- c(rf_r2_per_fold, r2)
  cat(sprintf("Random Forest fold %d R^2: %.3f\n", i, r2))
}

cat(sprintf("\nRandom Forest MEAN R^2 across 5 folds: %.3f\n\n", mean(rf_r2_per_fold)))

## ------------------------------------------------------------
## STEP 6: Run XGBoost with 5-fold CV
## ------------------------------------------------------------
xgb_r2_per_fold <- c()

for (i in seq_along(folds)) {
  test_idx  <- folds[[i]]
  train_idx <- setdiff(1:nrow(synthetic_df), test_idx)
  
  train_x <- as.matrix(synthetic_df[train_idx, 1:n_taxa])
  train_y <- synthetic_df$age[train_idx]
  test_x  <- as.matrix(synthetic_df[test_idx, 1:n_taxa])
  test_y  <- synthetic_df$age[test_idx]
  
  dtrain <- xgb.DMatrix(data = train_x, label = train_y)
  dtest  <- xgb.DMatrix(data = test_x)
  
  # NOTE: using xgb.train() (not xgboost()) because newer xgboost versions
  # changed xgboost() to a different x/y-based interface. xgb.train() still
  # works the old, reliable way with DMatrix objects and a params list.
  xgb_model <- xgb.train(
    params = list(
      objective = "reg:squarederror",
      max_depth = 4,
      eta = 0.1
    ),
    data = dtrain,
    nrounds = 200,
    verbose = 0
  )
  
  preds <- predict(xgb_model, dtest)
  
  ss_res <- sum((test_y - preds)^2)
  ss_tot <- sum((test_y - mean(test_y))^2)
  r2     <- 1 - (ss_res / ss_tot)
  
  xgb_r2_per_fold <- c(xgb_r2_per_fold, r2)
  cat(sprintf("XGBoost fold %d R^2: %.3f\n", i, r2))
}

cat(sprintf("\nXGBoost MEAN R^2 across 5 folds: %.3f\n\n", mean(xgb_r2_per_fold)))

## ------------------------------------------------------------
## STEP 7: Verdict
## ------------------------------------------------------------
cat("========================================================\n")
cat("VERDICT (fully synthetic check)\n")
cat("========================================================\n")
mean_r2 <- mean(c(rf_r2_per_fold, xgb_r2_per_fold))

if (mean_r2 > 0.3) {
  cat("PASS: Your pipeline correctly detects a known, planted signal.\n")
  cat("This means your near-zero R^2 on REAL age/BMI is a genuine\n")
  cat("biological finding (weak signal in real data), not a bug.\n")
} else {
  cat("FAIL: Your pipeline could NOT find an obvious, planted signal.\n")
  cat("This points to a bug somewhere in the modeling code itself\n")
  cat("(not a biological result) -- worth checking data types,\n")
  cat("train/test leakage, or CV fold construction.\n")
}
cat(sprintf("Average R^2 across RF + XGBoost: %.3f\n", mean_r2))

## Save synthetic data for a TabPFN check via your existing pipeline
write.csv(synthetic_df, "synthetic_data_for_tabpfn_check.csv", row.names = FALSE)
cat("\nSaved synthetic_data_for_tabpfn_check.csv -- feed this into your\n")
cat("existing TabPFN pipeline (age as target) as a third sanity check.\n\n")


## ============================================================
## PART B: SEMI-SYNTHETIC CHECK USING YOUR REAL X (Step 8)
## ============================================================
## PURPOSE (plain language):
## Part A used FULLY fake X and y - it proves your code runs, but not
## that it handles YOUR real data's specific structure (real taxa
## correlations, sparsity, whatever quirks your actual 471x100 CLR
## matrix has). This part keeps REAL X and only fakes the age label,
## at a KNOWN, chosen R^2, to find your pipeline's detection floor.
## ============================================================

## ------------------------------------------------------------
## Load your REAL, already CLR-transformed feature matrix
## ------------------------------------------------------------
## Replace this line with however you already load X in your real
## pipeline (I don't have your actual file/object name).
X_real <- read.csv("your_real_clr_data.csv")   # <-- REPLACE THIS LINE
n_real <- nrow(X_real)

## ------------------------------------------------------------
## Inject a known signal into real X at a target R^2
## ------------------------------------------------------------
inject_known_signal <- function(X, target_r2, n_signal_taxa = 10, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  signal_idx <- sample(1:ncol(X), n_signal_taxa)
  weights    <- runif(n_signal_taxa, min = -3, max = 3)
  
  signal <- as.matrix(X[, signal_idx]) %*% weights
  signal <- as.vector(scale(signal))              # standardize to unit variance
  
  # R^2 = var(signal) / (var(signal) + var(noise)); var(signal)=1, so:
  noise_sd <- sqrt(1 * (1 / target_r2 - 1))
  noise    <- rnorm(nrow(X), mean = 0, sd = noise_sd)
  
  as.vector(signal + noise)
}

## ------------------------------------------------------------
## Sweep across effect sizes, RF + XGBoost, same 5-fold CV as Part A
## ------------------------------------------------------------
effect_sizes <- c(0.7, 0.4, 0.2, 0.1, 0.05)
semi_synth_results <- data.frame()

for (r2_target in effect_sizes) {
  
  y_fake  <- inject_known_signal(X_real, target_r2 = r2_target, seed = 42)
  df_fake <- as.data.frame(X_real)
  df_fake$age <- y_fake
  
  folds_fake <- createFolds(df_fake$age, k = 5)
  
  ## Random Forest
  rf_fold_r2 <- c()
  for (i in seq_along(folds_fake)) {
    test_idx  <- folds_fake[[i]]
    train_idx <- setdiff(1:nrow(df_fake), test_idx)
    rf_model  <- randomForest(age ~ ., data = df_fake[train_idx, ], ntree = 500)
    preds     <- predict(rf_model, df_fake[test_idx, ])
    ss_res <- sum((df_fake$age[test_idx] - preds)^2)
    ss_tot <- sum((df_fake$age[test_idx] - mean(df_fake$age[test_idx]))^2)
    rf_fold_r2 <- c(rf_fold_r2, 1 - ss_res / ss_tot)
  }
  rf_achieved <- mean(rf_fold_r2)
  
  ## XGBoost
  xgb_fold_r2 <- c()
  for (i in seq_along(folds_fake)) {
    test_idx  <- folds_fake[[i]]
    train_idx <- setdiff(1:nrow(df_fake), test_idx)
    
    train_x <- as.matrix(df_fake[train_idx, 1:ncol(X_real)])
    train_y <- df_fake$age[train_idx]
    test_x  <- as.matrix(df_fake[test_idx, 1:ncol(X_real)])
    test_y  <- df_fake$age[test_idx]
    
    dtrain <- xgb.DMatrix(data = train_x, label = train_y)
    dtest  <- xgb.DMatrix(data = test_x)
    
    xgb_model <- xgb.train(
      params = list(objective = "reg:squarederror", max_depth = 4, eta = 0.1),
      data = dtrain, nrounds = 200, verbose = 0
    )
    preds <- predict(xgb_model, dtest)
    ss_res <- sum((test_y - preds)^2)
    ss_tot <- sum((test_y - mean(test_y))^2)
    xgb_fold_r2 <- c(xgb_fold_r2, 1 - ss_res / ss_tot)
  }
  xgb_achieved <- mean(xgb_fold_r2)
  
  cat(sprintf("Target R^2=%.2f -> RF achieved=%.3f, XGBoost achieved=%.3f\n",
              r2_target, rf_achieved, xgb_achieved))
  
  semi_synth_results <- rbind(semi_synth_results,
                              data.frame(target_r2 = r2_target, model = "rf",  achieved_r2 = rf_achieved),
                              data.frame(target_r2 = r2_target, model = "xgb", achieved_r2 = xgb_achieved))
}

cat("\n")
print(semi_synth_results)
write.csv(semi_synth_results, "semi_synthetic_detection_floor.csv", row.names = FALSE)

## ------------------------------------------------------------
## NOTE ON TabPFN (same pattern as Part A)
## ------------------------------------------------------------
## Pick one target R^2 (e.g. ~0.1-0.2, closest to published values),
## save that df_fake to CSV, and feed it into your EXISTING TabPFN
## pipeline in place of real data - same reasoning as Part A's
## write.csv(synthetic_data_for_tabpfn_check.csv) step.
y_fake_for_tabpfn <- inject_known_signal(X_real, target_r2 = 0.2, seed = 42)
df_fake_tabpfn <- as.data.frame(X_real)
df_fake_tabpfn$age <- y_fake_for_tabpfn
write.csv(df_fake_tabpfn, "semi_synthetic_data_for_tabpfn_check.csv", row.names = FALSE)
cat("\nSaved semi_synthetic_data_for_tabpfn_check.csv (target R^2=0.2) -\n")
cat("feed this into your existing TabPFN pipeline as a third check.\n")