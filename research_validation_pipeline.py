# =========================================================
# RESEARCH VALIDATION PIPELINE
# ET Baseline vs ET+NB Stacking
# Stratified 5-Fold Cross Validation
# =========================================================

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

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


X = df.drop("HeartDisease", axis=1).values
y = df["HeartDisease"].values

# =========================================================
# 2. SETUP CROSS VALIDATION
# =========================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

et_scores = []
stack_scores = []

# =========================================================
# 3. CROSS VALIDATION LOOP
# =========================================================

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):

    print(f"\n========== Fold {fold+1} ==========")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ------------------------------
    # ET Baseline
    # ------------------------------
    et = ExtraTreesClassifier(n_estimators=200, random_state=42)
    et.fit(X_train, y_train)

    et_test_proba = et.predict_proba(X_test)[:, 1]
    et_auc = roc_auc_score(y_test, et_test_proba)

    et_scores.append(et_auc)

    print("ET AUC:", round(et_auc, 4))

    # ------------------------------
    # ET + NB Stacking
    # ------------------------------
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    # Train probabilities
    et_train_proba = et.predict_proba(X_train)[:, 1]
    nb_train_proba = nb.predict_proba(X_train)[:, 1]

    # Test probabilities
    nb_test_proba = nb.predict_proba(X_test)[:, 1]

    # Meta features
    meta_train = np.column_stack([et_train_proba, nb_train_proba])
    meta_test = np.column_stack([et_test_proba, nb_test_proba])

    # Meta learner
    meta_lr = LogisticRegression(max_iter=1000)
    meta_lr.fit(meta_train, y_train)

    final_test_proba = meta_lr.predict_proba(meta_test)[:, 1]
    stack_auc = roc_auc_score(y_test, final_test_proba)

    stack_scores.append(stack_auc)

    print("ET + NB AUC:", round(stack_auc, 4))

# =========================================================
# 4. FINAL RESULTS
# =========================================================

et_mean = np.mean(et_scores)
et_std = np.std(et_scores)

stack_mean = np.mean(stack_scores)
stack_std = np.std(stack_scores)

print("\n\n==============================")
print("FINAL CROSS-VALIDATED RESULTS")
print("==============================")

print(f"ET Mean AUC: {et_mean:.4f} | Std: {et_std:.4f}")
print(f"ET+NB Mean AUC: {stack_mean:.4f} | Std: {stack_std:.4f}")

# =========================================================
# 5. STATISTICAL SIGNIFICANCE TEST
# =========================================================

t_stat, p_value = ttest_rel(stack_scores, et_scores)

print("\nPaired t-test:")
print("t-statistic:", round(t_stat, 4))
print("p-value:", round(p_value, 6))

if p_value < 0.05:
    print("Result: Improvement is statistically significant (p < 0.05)")
else:
    print("Result: Improvement is NOT statistically significant")