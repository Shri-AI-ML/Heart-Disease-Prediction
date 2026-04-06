# =========================================================
# ROBUSTNESS ANALYSIS UNDER GAUSSIAN NOISE
# ET vs ET+NB
# =========================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
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

noise_levels = [0.0, 0.05, 0.1, 0.2]

print("\n===== ROBUSTNESS UNDER NOISE =====")

for noise in noise_levels:

    et_scores = []
    stack_scores = []

    for train_idx, test_idx in skf.split(X, y):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Add Gaussian noise
        X_test_noisy = X_test + np.random.normal(0, noise, X_test.shape)

        # ET
        et = ExtraTreesClassifier(n_estimators=200, random_state=42)
        et.fit(X_train, y_train)
        et_proba = et.predict_proba(X_test_noisy)[:, 1]
        et_auc = roc_auc_score(y_test, et_proba)
        et_scores.append(et_auc)

        # ET + NB
        nb = GaussianNB()
        nb.fit(X_train, y_train)

        et_train = et.predict_proba(X_train)[:, 1]
        nb_train = nb.predict_proba(X_train)[:, 1]

        et_test = et_proba
        nb_test = nb.predict_proba(X_test_noisy)[:, 1]

        meta_train = np.column_stack([et_train, nb_train])
        meta_test = np.column_stack([et_test, nb_test])

        meta_lr = LogisticRegression(max_iter=1000)
        meta_lr.fit(meta_train, y_train)

        stack_proba = meta_lr.predict_proba(meta_test)[:, 1]
        stack_auc = roc_auc_score(y_test, stack_proba)
        stack_scores.append(stack_auc)

    print(f"\nNoise Level: {noise}")
    print("ET Mean AUC:", round(np.mean(et_scores),4))
    print("ET+NB Mean AUC:", round(np.mean(stack_scores),4))