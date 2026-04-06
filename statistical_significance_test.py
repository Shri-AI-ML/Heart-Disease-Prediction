# =========================================================
# CROSS-VALIDATION ENSEMBLE EXPERIMENT
# Stratified 5-Fold Validation
# =========================================================
import numpy as np
import pandas as pd
import itertools
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold



from sklearn.calibration import CalibratedClassifierCV

# =========================================================
# 1. DATA
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
# 2. MODELS
# =========================================================
models = {
    "RF": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGB": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4,
                         eval_metric='logloss', random_state=42),
    "LR": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel='rbf', probability=True),
    "ET": ExtraTreesClassifier(n_estimators=200, random_state=42),
    "NB": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# =========================================================
# 3. CROSS VALIDATION
# =========================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

base_results = {name: [] for name in models.keys()}
stack_results = {}

# =========================================================
# 4. TRAIN BASE MODELS WITH CV
# =========================================================

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):

    print(f"\n===== Fold {fold+1} =====")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    fold_probabilities = {}

    # Train base models
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)

        base_results[name].append(auc)
        fold_probabilities[name] = proba

        print(f"{name} AUC: {round(auc,4)}")

    # Store for stacking later
    if fold == 0:
        for combo in itertools.combinations(models.keys(), 2):
            stack_results[combo] = []

    # 2-model stacking
    for combo in itertools.combinations(models.keys(), 2):

        meta_train = []
        meta_test = []

        # Refit on train fold
        for name in combo:
            model = models[name]
            model.fit(X_train, y_train)

            val_proba = model.predict_proba(X_train)[:, 1]
            test_proba = model.predict_proba(X_test)[:, 1]

            meta_train.append(val_proba)
            meta_test.append(test_proba)

        meta_train = np.column_stack(meta_train)
        meta_test  = np.column_stack(meta_test)

        meta_lr = LogisticRegression(max_iter=1000)
        meta_lr.fit(meta_train, y_train)

        final_proba = meta_lr.predict_proba(meta_test)[:, 1]
        auc = roc_auc_score(y_test, final_proba)

        stack_results[combo].append(auc)

# =========================================================
# 5. SUMMARY RESULTS
# =========================================================

print("\n\n===== CROSS-VALIDATED BASE MODEL RESULTS =====")
for name, aucs in base_results.items():
    print(f"{name}: Mean AUC = {np.mean(aucs):.4f} | Std = {np.std(aucs):.4f}")

print("\n===== CROSS-VALIDATED STACKING RESULTS =====")

stack_summary = []

for combo, aucs in stack_results.items():
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    stack_summary.append((combo, mean_auc, std_auc))
    print(f"{combo}: Mean AUC = {mean_auc:.4f} | Std = {std_auc:.4f}")

stack_df = pd.DataFrame(stack_summary, columns=["Models", "Mean_AUC", "Std"])
stack_df = stack_df.sort_values(by="Mean_AUC", ascending=False)

print("\n===== TOP 10 STACKING COMBINATIONS =====")
print(stack_df.head(10))