import streamlit as st
import numpy as np
import joblib

# Load models
log_reg = joblib.load("logistic_model.pkl")
dtree = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🩺 Diabetes Predictor")
st.markdown("This app predicts whether a patient is likely to have **diabetes** based on health metrics.")

model_choice = st.sidebar.selectbox("Select Model", ("Logistic Regression", "Decision Tree"))

st.subheader("Enter Patient Details")

age = st.number_input("Age", min_value=1, max_value=120, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
glucose = st.number_input("Glucose Level", min_value=50.0, max_value=300.0, value=120.0)
bp = st.number_input("Blood Pressure", min_value=50.0, max_value=200.0, value=80.0)
family = st.selectbox("Family History of Diabetes?", ["No", "Yes"])

family_num = 1 if family == "Yes" else 0
input_data = np.array([[age, bmi, glucose, bp, family_num]])

if st.button("Predict"):
    if model_choice == "Logistic Regression":
        input_scaled = scaler.transform(input_data)
        prediction = log_reg.predict(input_scaled)[0]
    else:
        prediction = dtree.predict(input_data)[0]

    result = "🩸 Positive (Likely Diabetic)" if prediction == 1 else "✅ Negative (Not Diabetic)"
    st.subheader(f"Prediction Result: {result}")