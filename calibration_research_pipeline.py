# =========================================================
# CALIBRATION & RELIABILITY ANALYSIS
# ET vs ET+NB (5-Fold CV)
# =========================================================

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

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

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

et_brier = []
stack_brier = []

all_et_probs = []
all_stack_probs = []
all_true = []

for train_idx, test_idx in skf.split(X, y):

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ET
    et = ExtraTreesClassifier(n_estimators=200, random_state=42)
    et.fit(X_train, y_train)
    et_proba = et.predict_proba(X_test)[:, 1]

    # ET + NB
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    et_train = et.predict_proba(X_train)[:, 1]
    nb_train = nb.predict_proba(X_train)[:, 1]

    et_test = et_proba
    nb_test = nb.predict_proba(X_test)[:, 1]

    meta_train = np.column_stack([et_train, nb_train])
    meta_test = np.column_stack([et_test, nb_test])

    meta_lr = LogisticRegression(max_iter=1000)
    meta_lr.fit(meta_train, y_train)

    stack_proba = meta_lr.predict_proba(meta_test)[:, 1]

    # Brier Scores
    et_brier.append(brier_score_loss(y_test, et_proba))
    stack_brier.append(brier_score_loss(y_test, stack_proba))

    all_et_probs.extend(et_proba)
    all_stack_probs.extend(stack_proba)
    all_true.extend(y_test)

# =========================================================
# RESULTS
# =========================================================

print("ET Mean Brier:", np.mean(et_brier))
print("ET+NB Mean Brier:", np.mean(stack_brier))

# =========================================================
# RELIABILITY DIAGRAM
# =========================================================

et_true, et_pred = calibration_curve(all_true, all_et_probs, n_bins=10)
stack_true, stack_pred = calibration_curve(all_true, all_stack_probs, n_bins=10)

plt.figure(figsize=(6,6))
plt.plot(et_pred, et_true, marker='o', label='ET')
plt.plot(stack_pred, stack_true, marker='o', label='ET+NB')
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Reliability Diagram")
plt.legend()
plt.show()