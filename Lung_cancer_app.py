import streamlit as st
import joblib
import numpy as np
import os

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'Lung_cancer_model')
model = joblib.load(model_path)

def main():
    st.title("Lung Cancer Prediction")

    # Input fields
    p1 = st.slider("Enter your age", 1, 120, 30)
    p2 = st.selectbox("Smoking Habit", ("Yes", "No"))
    smoking = 1 if p2 == "Yes" else 0
    p3 = st.selectbox("Yellow Fingers", ("Yes", "No"))
    yellow_fingers = 1 if p3 == "Yes" else 0
    p4 = st.selectbox("Anxiety", ("Yes", "No"))
    anxiety = 1 if p4 == "Yes" else 0
    p5 = st.selectbox("Peer Pressure", ("Yes", "No"))
    peer_pressure = 1 if p5 == "Yes" else 0
    p6 = st.selectbox("Chronic Disease", ("Yes", "No"))
    chronic_disease = 1 if p6 == "Yes" else 0
    p7 = st.selectbox("Fatigue", ("Yes", "No"))
    fatigue = 1 if p7 == "Yes" else 0
    p8 = st.selectbox("Allergy", ("Yes", "No"))
    allergy = 1 if p8 == "Yes" else 0
    p9 = st.selectbox("Wheezing", ("Yes", "No"))
    wheezing = 1 if p9 == "Yes" else 0
    p10 = st.selectbox("Alcohol Consumption", ("Yes", "No"))
    alcohol_consumption = 1 if p10 == "Yes" else 0
    p11 = st.selectbox("Coughing", ("Yes", "No"))
    coughing = 1 if p11 == "Yes" else 0
    p12 = st.selectbox("Shortness of Breath", ("Yes", "No"))
    shortness_of_breath = 1 if p12 == "Yes" else 0
    p13 = st.selectbox("Swallowing Difficulty", ("Yes", "No"))
    swallowing_difficulty = 1 if p13 == "Yes" else 0
    p14 = st.selectbox("Chest Pain", ("Yes", "No"))
    chest_pain = 1 if p14 == "Yes" else 0

    # Combine inputs into a single array
    input_data = np.array([[p1, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease,
                            fatigue, allergy, wheezing, alcohol_consumption, coughing,
                            shortness_of_breath, swallowing_difficulty, chest_pain]])

    # Predict button
    if st.button("Predict"):
        try:
            prediction = model.predict(input_data)
            if prediction[0] == 1:
                st.error("The patient is likely to have lung cancer.")
            else:
                st.success("The patient is unlikely to have lung cancer.")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

if __name__ == '__main__':
    main()
