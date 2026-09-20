# ============================================================
# MERGED RERUNS ALL
# ============================================================

cat("===  1: RF/XGBoost reruns (n_perm=400) ===\n")
sig_rerun <- list()
sig_rerun$age_rf  <- run_cv_permutation_test(y_age, rf_reg_fn, n_perm = 400)
cat("age_rf done\n")
sig_rerun$age_xgb <- run_cv_permutation_test(y_age, xgb_reg_fn, n_perm = 400)
cat("age_xgb done\n")
sig_rerun$bmi_rf  <- run_cv_permutation_test(y_bmi, rf_reg_fn, n_perm = 400)
cat("bmi_rf done\n")
sig_rerun$sex_rf  <- run_cv_permutation_test(y_sex_bin, rf_clf_fn, n_perm = 400)
cat("sex_rf done\n")
sig_rerun$sex_xgb <- run_cv_permutation_test(y_sex_bin, xgb_clf_fn, n_perm = 400)
cat("sex_xgb done\n")

cat("\n===  2: TabPFN reruns (n_perm=300) ===\n")
sig_tabpfn_rerun <- list()
sig_tabpfn_rerun$age_tabpfn <- run_cv_permutation_test(y_age, tabpfn_reg_fn, n_perm = 300)
cat("age_tabpfn done\n")
sig_tabpfn_rerun$bmi_tabpfn <- run_cv_permutation_test(y_bmi, tabpfn_reg_fn, n_perm = 300)
cat("bmi_tabpfn done\n")
sig_tabpfn_rerun$sex_tabpfn <- run_cv_permutation_test(y_sex_bin, tabpfn_clf_fn, n_perm = 300)
cat("sex_tabpfn done\n")

cat("\n===  3: merging reruns into sig_results_full ===\n")
sig_results_full$age_rf  <- sig_rerun$age_rf
sig_results_full$age_xgb <- sig_rerun$age_xgb
sig_results_full$bmi_rf  <- sig_rerun$bmi_rf
sig_results_full$sex_rf  <- sig_rerun$sex_rf
sig_results_full$sex_xgb <- sig_rerun$sex_xgb
sig_results_full$age_tabpfn <- sig_tabpfn_rerun$age_tabpfn
sig_results_full$bmi_tabpfn <- sig_tabpfn_rerun$bmi_tabpfn
sig_results_full$sex_tabpfn <- sig_tabpfn_rerun$sex_tabpfn
# bmi_xgb (p=0.244) was already clearly resolved, never hit the floor,
# so it is left as-is from the original run -- no rerun needed

cat("\n===  4: FINAL, fully-resolved correction table ===\n")
all_p_named <- c(
  sapply(sig_results_full, function(r) r$p_value),
  sapply(cutpoint_results_full, function(r) r$p_value)
)
correction_table <- data.frame(test = names(all_p_named), raw_p = as.numeric(all_p_named))
correction_table$bonferroni_p   <- p.adjust(correction_table$raw_p, method = "bonferroni")
correction_table$BH_p           <- p.adjust(correction_table$raw_p, method = "BH")
correction_table$bonferroni_sig <- correction_table$bonferroni_p < 0.05
correction_table$BH_sig         <- correction_table$BH_p < 0.05
print(correction_table)