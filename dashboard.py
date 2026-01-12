import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load Model and Scaler 
model = joblib.load('Hybrid_heart_disease_model.joblib')
scaler = joblib.load('scaler.joblib')

# 2. Exact 8 Features Used During Fitting 

features = [
    'Age', 'Sex', 'ChestPainType', 'Cholesterol', 
    'NumMajorVessels', 'Thalassemia', 'ST_Slope', 'ExerciseAngina'
]

st.set_page_config(page_title="Heart Disease AI", page_icon="❤️")
st.title("❤️ Heart Disease Prediction Dashboard")

# 3. User Inputs 
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 100, 50)
    sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    chol = st.number_input("Cholesterol", 100, 600, 200)

with col2:
    vessels = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (1=Normal, 2=Fixed, 3=Reversable)", [1, 2, 3])
    slope = st.selectbox("ST Slope (0-2)", [0, 1, 2])
    exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [1, 0])

# 4. Prediction Logic 
if st.button("Predict Results"):
    # Creating DataFrame with the exact names and order
    input_df = pd.DataFrame([[age, sex, cp, chol, vessels, thal, slope, exang]], 
                            columns=features)
    
    try:
        # Step A: Scaling (Ab 8 vs 8 match ho jayega)
        input_scaled = scaler.transform(input_df)
        
        # Step B: Prediction
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[:, 1]

        st.divider()
        if prediction[0] == 1:
            st.error(f"### ⚠️ High Risk: Heart Disease Detected")
            st.write(f"Risk Probability: **{round(probability[0]*100, 2)}%**")
        else:
            st.success(f"### ✅ Low Risk: Normal")
            st.write(f"Confidence Score: **{round((1-probability[0])*100, 2)}%**")
            
    except Exception as e:
        st.error(f"Error: {e}")