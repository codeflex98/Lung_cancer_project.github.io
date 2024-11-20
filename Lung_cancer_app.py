# Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the model (Ensure the file path is correct)
model_path = os.path.join(os.path.dirname(__file__), 'Lung_cancer_model')
model = joblib.load(model_path)


# Create a function for making predictions
def predict_lung_cancer(model, input_data):
    # Input data can be a single row dataframe based on user input
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data) if hasattr(model, 'predict_proba') else None
    return prediction, probability

#def main():
st.title("Lung Cancer Prediction App")
st.write("Enter the following details to predict the likelihood of lung cancer:")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100)
smoking = st.selectbox("Smoking (1 for Yes, 2 for No)", [1, 2])
yellow_fingers = st.selectbox("Yellow Fingers (1 for Yes, 2 for No)", [1, 2])
anxiety = st.selectbox("Anxiety (1 for Yes, 2 for No)", [1, 2])
peer_pressure = st.selectbox("Peer Pressure (1 for Yes, 2 for No)", [1, 2])
chronic_disease = st.selectbox("Chronic Disease (1 for Yes, 2 for No)", [1, 2])
fatigue = st.selectbox("Fatigue (1 for Yes, 2 for No)", [1, 2])
allergy = st.selectbox("Allergy (1 for Yes, 2 for No)", [1, 2])
wheezing = st.selectbox("Wheezing (1 for Yes, 2 for No)", [1, 2])
alcohol_consuming = st.selectbox("Alcohol Consuming (1 for Yes, 2 for No)", [1, 2])
coughing = st.selectbox("Coughing (1 for Yes, 2 for No)", [1, 2])
shortness_of_breath = st.selectbox("Shortness of Breath (1 for Yes, 2 for No)", [1, 2])
swallowing_difficulty = st.selectbox("Swallowing Difficulty (1 for Yes, 2 for No)", [1, 2])
chest_pain = st.selectbox("Chest Pain (1 for Yes, 2 for No)", [1, 2])

# Map gender input
gender = 1 if gender == "Male" else 0

# Create a DataFrame for the input data
input_data = pd.DataFrame({
    'GENDER': [gender],
    'AGE': [age],
    'SMOKING': [smoking],
    'YELLOW_FINGERS': [yellow_fingers],
    'ANXIETY': [anxiety],
    'PEER_PRESSURE': [peer_pressure],
    'CHRONIC DISEASE': [chronic_disease],
    'FATIGUE': [fatigue],
    'ALLERGY': [allergy],
    'WHEEZING': [wheezing],
    'ALCOHOL CONSUMING': [alcohol_consuming],
    'COUGHING': [coughing],
    'SHORTNESS OF BREATH': [shortness_of_breath],
    'SWALLOWING DIFFICULTY': [swallowing_difficulty],
    'CHEST PAIN': [chest_pain]
})

# Predict button
if st.button("Predict Lung Cancer Risk"):
    prediction = model.predict(input_data)[0]
    result = "High Risk of Lung Cancer" if prediction == 1 else "Low Risk of Lung Cancer"
    st.write("Prediction:", result)

if __name__ == '__main__':
    main()
