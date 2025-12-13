import streamlit as st
import pandas as pd
import joblib
import time
import plotly.graph_objects as go

# --- Configuration and Mappings ---
FEATURE_MAPPINGS = {
    'Sex': {'Male': 1, 'Female': 0},
    'ChestPainType': {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3},
    'ExerciseAngina': {'Yes': 1, 'No': 0},
    'ST_Slope': {'Up Sloping': 0, 'Flat': 1, 'Down Sloping': 2},
    'Thalassemia': {'NULL': 0, 'Normal Blood Flow': 1, 'Fixed Defect': 2, 'Reversible Defect': 3},
    'NumMajorVessels': {'0': 0, '1': 1, '2': 2, '3': 3}
}

# --- Streamlit App Setup ---
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide"
)

# --- Model Loading (Cached for Performance) ---
@st.cache_resource
def load_model():
    """Load the machine learning model from a file."""
    try:
        model = joblib.load('Hybrid_heart_disease_model.joblib')
        return model
    except FileNotFoundError:
        st.error("Error: The model file 'Hybrid_heart_disease_model.joblib' was not found. Please make sure it is in the same directory.")
        return None

model = load_model()
if model is None:
    st.stop()

# --- Page Header ---
st.title('🫀 Heart Disease Prediction Dashboard')
st.info("A Machine Learning-based Predictor built with Streamlit.")
st.markdown("---")

# --- Initialize session state for resetting the form ---
if 'form_key' not in st.session_state:
    st.session_state.form_key = 0

def reset_form():
    st.session_state.form_key += 1
    st.session_state.prediction = None

# --- Main App Content with Tabs ---
tab1, tab2 = st.tabs(["📊 Prediction", "ℹ About the App"])

with tab1:
    st.subheader("Patient Data Input")
    st.markdown("Fill out the patient's information below to get a *heart disease risk prediction*.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider('Age', min_value=18, max_value=100, value=60)
        sex = st.radio('Sex', list(FEATURE_MAPPINGS['Sex'].keys()))
        resting_bp = st.slider('RestingBP', min_value=80, max_value=200, value=120)
        max_hr = st.slider('Max HR', min_value=60, max_value=220, value=150)
        #oldpeak = st.slider('Oldpeak', min_value=0.0, max_value=6.0, value=1.0, step=0.1)

    with col2:
        cp = st.selectbox('Chest Pain Type', list(FEATURE_MAPPINGS['ChestPainType'].keys()))
        cholesterol = st.slider('Cholesterol', min_value=100, max_value=600, value=200)
        exang = st.radio('Exercise Angina', list(FEATURE_MAPPINGS['ExerciseAngina'].keys()))
        slope = st.selectbox('ST Slope', list(FEATURE_MAPPINGS['ST_Slope'].keys()))
        thal = st.selectbox('Thalassemia', list(FEATURE_MAPPINGS['Thalassemia'].keys()))
        num_major_vessels = st.selectbox('Num Major Vessels', list(FEATURE_MAPPINGS['NumMajorVessels'].keys()))

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button('Predict'):
            st.session_state.prediction_made = True
            try:
                with st.status("Analyzing Patient Data...", expanded=True) as status:
                    st.write("Initializing model...")
                    time.sleep(0.5)
                    st.write("Analyzing patient vitals and health indicators...")
                    time.sleep(1)
                    st.write("Cross-referencing against the dataset...")
                    time.sleep(1.5)
                    st.write("Finalizing prediction...")
                    time.sleep(0.5)


                    user_data = {
                        'Age': age, 'Sex': FEATURE_MAPPINGS['Sex'][sex], 'ChestPainType': FEATURE_MAPPINGS['ChestPainType'][cp],
                        'RestingBP': resting_bp, 'Cholesterol': cholesterol, 'MaxHR': max_hr,
                        'ExerciseAngina': FEATURE_MAPPINGS['ExerciseAngina'][exang], 
                        'ST_Slope': FEATURE_MAPPINGS['ST_Slope'][slope],
                        'NumMajorVessels': FEATURE_MAPPINGS['NumMajorVessels'][num_major_vessels],
                        'Thalassemia': FEATURE_MAPPINGS['Thalassemia'][thal]
                    }
                    model_features = list(model.feature_names_in_)
                    prediction, prediction_proba = model.predict(pd.DataFrame([user_data])[model_features]), model.predict_proba(pd.DataFrame([user_data])[model_features])
                    st.session_state.prediction = prediction[0]
                    st.session_state.confidence = prediction_proba[0][prediction[0]]
                    st.session_state.probas = prediction_proba.flatten()
                    st.session_state.user_data = user_data
                    status.update(label="Prediction Complete!", state="complete", expanded=False)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.session_state.prediction_made = False

    with col_btn2:
        st.button('Reset Form', on_click=reset_form)
    
    if st.session_state.get('prediction_made', False):
        st.divider()
        st.subheader("Prediction Results & Insights")
        
        # Display key metrics
        risk_level = "High Risk" if st.session_state.prediction == 1 else "Low Risk"
        risk_color = "red" if st.session_state.prediction == 1 else "green"
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric(label="Predicted Risk", value=risk_level, delta=f"{st.session_state.confidence*100:.2f}% Confidence")
            
        with col_res2:
            st.metric(label="Patient Age", value=st.session_state.user_data['Age'])
            st.metric(label="Resting BP", value=st.session_state.user_data['RestingBP'])
            st.metric(label="Cholesterol", value=st.session_state.user_data['Cholesterol'])

        # Create interactive visualizations
        st.markdown("---")
        st.subheader("Visualizing the Prediction")
        
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("##### Prediction Probability")
            
            # Create a DataFrame for the line chart
            proba_df = pd.DataFrame(
                {
                    'Category': ['No Heart Disease', 'Heart Disease'],
                    'Probability': st.session_state.probas
                }
            )
            st.line_chart(proba_df.set_index('Category'))
            
            if st.session_state.prediction == 1:
                st.error("⚠ The model predicts a high likelihood of heart disease. Please consult a medical professional.")
            else:
                st.success("✅ The model predicts a low likelihood of heart disease. This is good news!")
        
        with chart_col2:
            st.markdown("##### Key Vitals Comparison")
            # Create a radar chart to compare patient data to a "healthy" baseline
            categories = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']
            healthy_values = [40, 120, 200, 160]
            patient_values = [st.session_state.user_data['Age'], st.session_state.user_data['RestingBP'], st.session_state.user_data['Cholesterol'], st.session_state.user_data['MaxHR']]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=healthy_values,
                theta=categories,
                fill='toself',
                name='Healthy Baseline',
                marker=dict(color='lightgreen')
            ))
            fig.add_trace(go.Scatterpolar(
                r=patient_values,
                theta=categories,
                fill='toself',
                name='Patient Vitals',
                marker=dict(color='skyblue')
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, max(max(healthy_values), max(patient_values)) * 1.2])
                ),
                showlegend=True,
                height=350,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        
st.warning("Disclaimer : This is for educational purposes only and should not be used as a substitute for professional medical advice. Always consult a healthcare professional for diagnosis and treatment.")

with tab2:
    st.header("About This Application")
    st.write("This app uses a machine learning model to predict the likelihood of heart disease based on patient data.")
    st.markdown("""
    *Model:* The model is a pre-trained Joblib file, likely trained on a dataset containing various health metrics and their correlation with heart disease.
    *Features:* The app requires 11 input features, including age, gender, cholesterol, and heart rate, which are common indicators used in cardiac health analysis.
    """)
    with st.expander("View Feature Mappings"):
        for key, value in FEATURE_MAPPINGS.items():
            st.write(f"{key}:** {value}")
            
    st.subheader("Project Team")
    st.markdown("""
    - Narendra Bishnoi            
    - Aaditya Mishra
    - Aryan Raj
    - Shrijal Goswami
        """)