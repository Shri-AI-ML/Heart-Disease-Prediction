# 🫀 Advanced Heart Disease Prediction & Research Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/Model-Ensembles%20%26%20Stacking-purple)
![Research](https://img.shields.io/badge/Focus-Robustness%20%26%20Calibration-orange)

## 📌 Overview

This project has evolved from a foundational Hybrid Stacking Ensemble into a comprehensive **Machine Learning Research Pipeline** focused on predicting heart disease. The repository now encompasses deep investigations into feature weighting, model calibration, robustness against data noise, and statistical validation of ensemble methods.

Our main goal is to not only achieve high accuracy but to ensure **clinical reliability**, producing well-calibrated probabilities and robust predictions even when patient data is noisy.

---

## 🔬 Key Research Areas

### 1. Entropy-Weighted Feature Ensembles
*File: `entropy_weighted_et_pipeline.py`*
* **Concept:** Computed per-feature entropy to dynamically weight features based on their information content before feeding them into an Extra Trees classifier.
* **Outcome:** Actively mitigates the effect of noisy or low-information variables. Evaluated against a baseline Extra Trees model using Brier Scores, AUC, and paired t-tests.

### 2. Statistical Stacking & Validation
*Files: `statistical_significance_test.py`, `research_validation_pipeline.py`*
* **Concept:** Extensive Stratified 5-Fold Cross-Validation across a diverse set of models (Extra Trees, Random Forest, XGBoost, Naive Bayes, SVM, KNN).
* **Investigation:** Automated evaluation of all possible 2-model stacking combinations using Logistic Regression as the Meta-Learner.
* **Validation:** Rigorous statistical hypothesis testing (paired t-tests) to confirm whether ensemble setups (e.g., ET + NB) provide a significant improvement over single baseline models.

### 3. Model Calibration & Reliability
*File: `calibration_research_pipeline.py`*
* **Concept:** High accuracy means little if a model's predicted probability of 80% doesn't roughly map to 80% true positive rate in reality.
* **Investigation:** Benchmarked the calibration curves and Brier scores between individual base models (like ET) and stacked models (ET + NB).
* **Outcome:** Verified that the stacking approach improves true probability mapping, making it more reliable for clinical decision-support systems.

### 4. Robustness Under Gaussian Noise
*File: `robustness_noise_analysis.py`*
* **Concept:** Medical data is rarely perfect. Missing sensor readings, human error, or differing hospital standardizations create "noisy" datasets.
* **Investigation:** Simulated real-world imperfections by injecting varying levels of Gaussian noise (from 0.0 to 0.2 variance) into the testing data.
* **Outcome:** Analyzed the degradation curve of stacked models vs standalone models, proving the added resilience gained through combining orthogonal models (like Extra Trees + Naive Bayes).

---

## ⚙️ Core Architectures Researched

### Early Architecture
- **Base Learners (Level 0):** Random Forest and XGBoost.
- **Meta Learner (Level 1):** Logistic Regression.

### Progressive Research Architectures
- **Base Learners (Level 0):** Extra Trees (ET) and Naive Bayes (NB). This pairing proved highly effective because ET handles complex non-linear splits while NB handles probabilistic conditional independence, making them exceptionally orthogonal.
- **Meta Learner (Level 1):** Logistic Regression to output the final, smooth probability map.

---

## 📊 Dataset & Preprocessing

* **Dataset:** Contains standard cardiovascular clinical measures, including Age, Sex, Chest Pain Type, Cholesterol, Fasting Blood Sugar, Max HR, etc.
* **Preprocessing Pipeline:**
  * Scaling via `StandardScaler` to normalize feature variance.
  * Entropy Computation for feature importance scaling.
  * Cross-Validation splitting (Stratified 5-Fold) to prevent data leakage and guarantee valid statistical conclusions.

---

## 📂 Project Structure

```bash
├── datasets/                              # Clinical heart disease data
├── entropy_weighted_et_pipeline.py        # Entropy scaling & ET research
├── calibration_research_pipeline.py       # Reliability diagrams & Brier scoring
├── robustness_noise_analysis.py           # Noise injection capability tests
├── statistical_significance_test.py       # Base model CV & pairwise stacking grid-search
├── research_validation_pipeline.py        # Paired t-tests for stacking vs baseline
├── dashboard.py                           # Frontend UI/Dashboard for interaction
├── HUC-EGML.ipynb                         # Additional experimental notebook
├── main.ipynb                             # Core modeling & EDA
├── universal_heart_research_pipeline.ipynb# End-to-end exploratory pipeline
├── requirements.txt                       # Project dependencies
└── README.md                              # This documentation
```

---

## 🛠️ Tech Stack

* **Programming Language:** Python 🐍
* **Machine Learning:** `scikit-learn`, `xgboost`, `scipy` (for statistical testing)
* **Data Manipulation & Viz:** `pandas`, `numpy`, `matplotlib`, `seaborn`

---

## 💻 Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/Shri-AI-ML/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. **Verify Dependencies**
   It's recommended to use a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Run the Research Pipelines**
   Execute any of the individual research scripts to replicate the findings:
   ```bash
   python robustness_noise_analysis.py
   python statistical_significance_test.py
   ```

---

## 🙌 Acknowledgements

* Dataset inspired by the UCI Heart Disease dataset.
* This is an ongoing project designed to bridge the gap between simple benchmark accuracy and robust, clinically reliable machine learning.
