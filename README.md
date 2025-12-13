# 🫀 Heart Disease Prediction using Hybrid Stacking Ensemble

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/Model-Hybrid%20Stacking-purple)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Overview

This project implements a robust **Hybrid Machine Learning Model** to predict the presence of heart disease. Instead of relying on a single algorithm, we used a **Stacking Ensemble** technique that combines multiple powerful classifiers (Random Forest, Logistic Regression, XGBoost) to achieve higher accuracy and better generalization on unseen data.

The model is designed to minimize false negatives, ensuring that critical cases of heart disease are detected effectively.

---

## ⚙️ Model Architecture

We used a **Stacking Classifier** approach:

1.  **Base Learners (Level 0):**
    * **Random Forest:** Captures complex, non-linear patterns.
    * **XGBoost:** Handles gradient boosting for high performance.
    2.  **Meta Learner (Level 1):**
    * **Logistic Regression:** Takes the predictions from base learners and makes the final decision.

---

## 📈 Performance Results

The Hybrid Stacking Model demonstrated superior performance in distinguishing between patients with and without heart disease.

### 🏆 Key Metrics
* **Best Test Accuracy:** 89.13%
* **Best AUC Score:** 0.9429

### 📝 Detailed Classification Report
*(Based on Stacking Ensemble Evaluation)*

| Class                | Precision | Recall   | F1-Score | Support |
| :---                 | :---:     | :---:    | :---:    | :---:   |
| **0 (No Disease)**   | 0.83      |   0.79   |  0.81    |   24    |
| **1 (Disease)**      | **0.77**  | **0.81** | **0.79** |    21   |
| **Overall Accuracy** | | | **0.80** | 45 |

> **Interpretation:**
> * **AUC of ~0.94** indicates the model has excellent capability to distinguish between classes.
> * **Recall of 0.81 for Class 1** shows the model is effective at identifying positive heart disease cases.

---

## 📊 Dataset & Preprocessing

* **Dataset:** Contains medical details like Age, Sex, CP (Chest Pain), Chol (Cholesterol), Thal, etc.
* **Preprocessing Steps:**
    * Handling missing values.
    * **Scaling:** Applied Standard Scaling to normalize features.
    * **Encoding:** Converted categorical variables for ML compatibility.
    * **Splitting:** Data divided into Training and Testing sets to prevent data leakage.

---

## 🛠️ Tech Stack

* **Programming Language:** Python 🐍
* **Libraries:**
    * `scikit-learn` (for Stacking, RF, Logistic Regression)
    * `xgboost` (for Gradient Boosting)
    * `pandas` & `numpy` (for Data Manipulation)
    * `matplotlib` & `seaborn` (for Visualization)

---

## 📂 Project Structure

```bash
├── data/               # Dataset files
├── heart_disease_prediction.ipynb  # Main source code
├── requirements.txt    # Dependencies
└── README.md           # Project Documentation

---

## 🛠️ Tech Stack

* **Language:** Python 🐍
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`
* **Tools:** Jupyter Notebook / Google Colab

---

## 💻 Installation & Usage

1. **Clone the repository**
   ```bash
   git clone [https://github.com/your-username/heart-disease-prediction.git](https://github.com/your-username/heart-disease-prediction.git)


## 🚀 Real-world Use Case

This project demonstrates how ML can assist in **early detection of heart disease**.
Doctors and healthcare professionals can use such predictive models to **support diagnosis, reduce risks, and improve patient outcomes**.

---

## 🛠️ Tech Stack

* **Python** 🐍
* Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`

---

## 📂 Project Structure

```
├── datasets/              
├── main.ipynb
├── dashboard.py       
├── requirements.txt
├── .gitignore  
└── README.md         
```

---


---

## 🙌 Acknowledgements

* Dataset inspired by UCI Heart Disease dataset
* Built as part of my **AI/ML learning journey** 🌟


