import streamlit as st
import pandas as pd
import pickle

# Load the trained model
MODEL_PATH = "lung_cancer_model.pkl" 

@st.cache_data
def load_model():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    return model

# Load the model
model = load_model()

# Streamlit UI
st.title("Lung Cancer Prediction App")
st.write("Provide the input features to predict lung cancer.")

# Define input fields based on the model's features
# Modify the feature names based on your dataset
age = st.number_input("Age", min_value=18, max_value=100, value=50)
smoking = st.selectbox("Smoking", ["No", "Yes"])
alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
family_history = st.selectbox("Family History of Cancer", ["No", "Yes"])
chronic_disease = st.selectbox("Chronic Diseases", ["No", "Yes"])
coughing = st.selectbox("Frequent Coughing", ["No", "Yes"])
shortness_of_breath = st.selectbox("Shortness of Breath", ["No", "Yes"])

# Convert categorical inputs into numerical values
input_data = pd.DataFrame({
    "Age": [age],
    "Smoking": [1 if smoking == "Yes" else 0],
    "Alcohol Consumption": [1 if alcohol == "Yes" else 0],
    "Family History": [1 if family_history == "Yes" else 0],
    "Chronic Disease": [1 if chronic_disease == "Yes" else 0],
    "Coughing": [1 if coughing == "Yes" else 0],
    "Shortness of Breath": [1 if shortness_of_breath == "Yes" else 0]
})

# Prediction
if st.button("Predict"): 
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]
    
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error(f"The model predicts a HIGH RISK of lung cancer with {probability[0]*100:.2f}% probability.")
    else:
        st.success(f"The model predicts a LOW RISK of lung cancer with {probability[0]*100:.2f}% probability.")

st.write("Disclaimer: This is an AI-based prediction and not a medical diagnosis. Please consult a doctor for medical advice.")
