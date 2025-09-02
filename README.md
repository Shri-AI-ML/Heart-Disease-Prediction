# 🫀 Heart Disease Prediction using Machine Learning

## 📌 Overview

This project focuses on predicting the presence of heart disease using machine learning algorithms.
We applied multiple classification models, compared their performance, and selected the best-performing ones for accurate predictions.

---

## 📊 Dataset

* The dataset contains medical attributes such as **age, cholesterol, resting BP, max heart rate, chest pain type, etc.**
* Preprocessing steps included:

  * Handling missing values & outliers
  * Encoding categorical features
  * Standardization for models like Logistic Regression & SVM

---

## ⚙️ Methods

1. **Data Preprocessing**

   * Label Encoding categorical variables
   * Outlier detection using IQR
   * Standardization (for LR, SVM, KNN, Naive Bayes)

2. **Train / Validation / Test Split**

   * 60% → Training
   * 20% → Validation
   * 20% → Testing

3. **Machine Learning Models Used**

   * Logistic Regression
   * Support Vector Machine (SVM)
   * Random Forest
   * Decision Tree
   * K-Nearest Neighbors (KNN)
   * Naive Bayes
   * XGBoost

---

## 📈 Results

*Logistic Regression: 0.83  
*Random Forest: 0.99  
*SVM: 0.84  
*Decision Tree: 0.96  
*KNN: 0.85  
*Naive Bayes: 0.80  
*XGBoost: 0.96 


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
├── data/              # Dataset (if included or linked)  
├── notebooks/         # Jupyter Notebooks (EDA + model training)  
├── heart_disease.py   # Main Python script  
├── requirements.txt   # Dependencies  
└── README.md          # Project documentation  
```

---

## 📸 Example Output

```
Logistic Regression: 0.83  
Random Forest: 0.99  
SVM: 0.84  
Decision Tree: 0.96  
KNN: 0.85  
Naive Bayes: 0.80  
XGBoost: 0.96  
```

---

## 🙌 Acknowledgements

* Dataset inspired by UCI Heart Disease dataset
* Built as part of my **AI/ML learning journey** 🌟


