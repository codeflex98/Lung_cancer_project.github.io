# Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and preprocess the data
@st.cache
def load_data():
    file_path = 'Lung_cancer_model'  # Update to your file path
    data = pd.read_csv(file_path)

    # Encoding categorical columns
    encoder = LabelEncoder()
    data['GENDER'] = encoder.fit_transform(data['GENDER'])
    data['LUNG_CANCER'] = encoder.fit_transform(data['LUNG_CANCER'])

    X = data.drop(columns=['LUNG_CANCER'])
    y = data['LUNG_CANCER']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, encoder

# Load data and train model
X, y, scaler, encoder = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Streamlit App Interface
st.title("Lung Cancer Prediction App")
st.write("Enter the details to predict the likelihood of lung cancer.")

# Collecting input data from the user
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 1, 100, 30)
smoking = st.selectbox("Smoking (0: No, 1: Yes)", [0, 1])
yellow_fingers = st.selectbox("Yellow Fingers (0: No, 1: Yes)", [0, 1])
anxiety = st.selectbox("Anxiety (0: No, 1: Yes)", [0, 1])
peer_pressure = st.selectbox("Peer Pressure (0: No, 1: Yes)", [0, 1])
chronic_disease = st.selectbox("Chronic Disease (0: No, 1: Yes)", [0, 1])
fatigue = st.selectbox("Fatigue (0: No, 1: Yes)", [0, 1])
allergy = st.selectbox("Allergy (0: No, 1: Yes)", [0, 1])
wheezing = st.selectbox("Wheezing (0: No, 1: Yes)", [0, 1])
alcohol_consuming = st.selectbox("Alcohol Consuming (0: No, 1: Yes)", [0, 1])
coughing = st.selectbox("Coughing (0: No, 1: Yes)", [0, 1])
shortness_of_breath = st.selectbox("Shortness of Breath (0: No, 1: Yes)", [0, 1])
swallowing_difficulty = st.selectbox("Swallowing Difficulty (0: No, 1: Yes)", [0, 1])
chest_pain = st.selectbox("Chest Pain (0: No, 1: Yes)", [0, 1])

# Encoding the input
gender_encoded = 1 if gender == "Male" else 0
input_data = np.array([[gender_encoded, age, smoking, yellow_fingers, anxiety, peer_pressure,
                        chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
                        coughing, shortness_of_breath, swallowing_difficulty, chest_pain]])

# Standardizing the input
input_data_scaled = scaler.transform(input_data)

# Making prediction
prediction = model.predict(input_data_scaled)
prediction_proba = model.predict_proba(input_data_scaled)

# Displaying results
if st.button("Predict"):
    result = "Positive for Lung Cancer" if prediction[0] == 1 else "Negative for Lung Cancer"
    st.write(f"Prediction: {result}")
    st.write(f"Prediction Confidence: {prediction_proba[0][prediction[0]]:.2f}")
