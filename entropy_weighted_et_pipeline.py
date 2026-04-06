# =========================================================
# ENTROPY-WEIGHTED EXTRA TREES RESEARCH PIPELINE
# Baseline ET vs Entropy-Weighted ET
# 5-Fold Stratified Cross Validation
# =========================================================

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.ensemble import ExtraTreesClassifier

# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv('datasets/heart.csv')



df.rename(columns={             #renaming for better understanding 
    'age': 'Age',
    'sex': 'Sex',
    'cp': 'ChestPainType',
    'trestbps': 'RestingBP',
    'chol': 'Cholesterol',
    'fbs': 'FastingBS',
    'restecg': 'RestingECG',
    'thalach': 'MaxHR',
    'exang': 'ExerciseAngina',
    'oldpeak': 'Oldpeak',
    'slope': 'ST_Slope',
    'ca': 'NumMajorVessels',
    'thal': 'Thalassemia',
    'target': 'HeartDisease'
}, inplace=True)



X_df = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"].values

# =========================================================
# 2. FEATURE ENTROPY COMPUTATION
# =========================================================

def compute_feature_entropy(df):
    entropy_dict = {}
    for col in df.columns:
        probs = df[col].value_counts(normalize=True)
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        entropy_dict[col] = entropy
    return entropy_dict


def apply_entropy_weighting(df, entropy_dict):
    df_weighted = df.copy()
    for col in df.columns:
        weight = 1 / (1 + entropy_dict[col])
        df_weighted[col] = df[col] * weight
    return df_weighted


# =========================================================
# 3. CROSS VALIDATION SETUP
# =========================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

baseline_auc = []
entropy_auc = []

baseline_brier = []
entropy_brier = []

# =========================================================
# 4. CROSS VALIDATION LOOP
# =========================================================

for fold, (train_idx, test_idx) in enumerate(skf.split(X_df, y)):

    print(f"\n========== Fold {fold+1} ==========")

    X_train_df = X_df.iloc[train_idx]
    X_test_df  = X_df.iloc[test_idx]

    y_train = y[train_idx]
    y_test  = y[test_idx]

    # ----------------------------
    # BASELINE ET
    # ----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled  = scaler.transform(X_test_df)

    et_baseline = ExtraTreesClassifier(n_estimators=200, random_state=42)
    et_baseline.fit(X_train_scaled, y_train)

    baseline_proba = et_baseline.predict_proba(X_test_scaled)[:, 1]

    auc_base = roc_auc_score(y_test, baseline_proba)
    brier_base = brier_score_loss(y_test, baseline_proba)

    baseline_auc.append(auc_base)
    baseline_brier.append(brier_base)

    print("Baseline ET AUC:", round(auc_base,4))

    # ----------------------------
    # ENTROPY-WEIGHTED ET
    # ----------------------------

    entropy_dict = compute_feature_entropy(X_train_df)

    X_train_weighted = apply_entropy_weighting(X_train_df, entropy_dict)
    X_test_weighted  = apply_entropy_weighting(X_test_df, entropy_dict)

    scaler_w = StandardScaler()
    X_train_w_scaled = scaler_w.fit_transform(X_train_weighted)
    X_test_w_scaled  = scaler_w.transform(X_test_weighted)

    et_entropy = ExtraTreesClassifier(n_estimators=200, random_state=42)
    et_entropy.fit(X_train_w_scaled, y_train)

    entropy_proba = et_entropy.predict_proba(X_test_w_scaled)[:, 1]

    auc_entropy = roc_auc_score(y_test, entropy_proba)
    brier_entropy = brier_score_loss(y_test, entropy_proba)

    entropy_auc.append(auc_entropy)
    entropy_brier.append(brier_entropy)

    print("Entropy ET AUC:", round(auc_entropy,4))


# =========================================================
# 5. FINAL RESULTS
# =========================================================

print("\n==============================")
print("FINAL CROSS-VALIDATED RESULTS")
print("==============================")

print(f"Baseline ET Mean AUC: {np.mean(baseline_auc):.4f} | Std: {np.std(baseline_auc):.4f}")
print(f"Entropy ET Mean AUC: {np.mean(entropy_auc):.4f} | Std: {np.std(entropy_auc):.4f}")

print("\nBaseline ET Mean Brier:", round(np.mean(baseline_brier),4))
print("Entropy ET Mean Brier:", round(np.mean(entropy_brier),4))

# =========================================================
# 6. STATISTICAL SIGNIFICANCE TEST
# =========================================================

t_stat_auc, p_value_auc = ttest_rel(entropy_auc, baseline_auc)
t_stat_brier, p_value_brier = ttest_rel(entropy_brier, baseline_brier)

print("\nPaired t-test (AUC):")
print("p-value:", round(p_value_auc,6))

print("\nPaired t-test (Brier Score):")
print("p-value:", round(p_value_brier,6))

if p_value_auc < 0.05:
    print("\nAUC improvement is statistically significant.")
else:
    print("\nAUC improvement is NOT statistically significant.")