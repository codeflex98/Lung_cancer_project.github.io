import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_filename = 'Lung_cancer_model.pkl'
model = joblib.load(model_filename)

# Title and description
st.title("Lung Cancer Prediction App")
st.write("""
This app predicts the likelihood of a patient having lung cancer based on their inputs. 
Fill out the details below and click on **Predict** to see the result.
""")

# User input form
def user_input_features():
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=1, max_value=120, step=1, value=30)
    smoking = st.slider("Smoking Level (1-2)", min_value=1, max_value=2, value=1)
    yellow_fingers = st.slider("Yellow Fingers (1-2)", min_value=1, max_value=2, value=1)
    anxiety = st.slider("Anxiety Level (1-2)", min_value=1, max_value=2, value=1)
    peer_pressure = st.slider("Peer Pressure Level (1-2)", min_value=1, max_value=2, value=1)
    chronic_disease = st.slider("Chronic Disease Level (1-2)", min_value=1, max_value=2, value=1)
    fatigue = st.slider("Fatigue Level (1-2)", min_value=1, max_value=2, value=1)
    allergy = st.slider("Allergy Level (1-2)", min_value=1, max_value=2, value=1)
    wheezing = st.slider("Wheezing Level (1-2)", min_value=1, max_value=2, value=1)
    alcohol_consuming = st.slider("Alcohol Consuming Level (1-2)", min_value=1, max_value=2, value=1)
    coughing = st.slider("Coughing Level (1-2)", min_value=1, max_value=2, value=1)
    shortness_of_breath = st.slider("Shortness of Breath (1-2)", min_value=1, max_value=2, value=1)
    swallowing_difficulty = st.slider("Swallowing Difficulty Level (1-2)", min_value=1, max_value=2, value=1)
    chest_pain = st.slider("Chest Pain Level (1-2)", min_value=1, max_value=2, value=1)

    # Encode gender as numerical
    gender_encoded = 1 if gender == "Male" else 0

    # Create a dataframe for the input
    data = {
        "GENDER": gender_encoded,
        "AGE": age,
        "SMOKING": smoking,
        "YELLOW_FINGERS": yellow_fingers,
        "ANXIETY": anxiety,
        "PEER_PRESSURE": peer_pressure,
        "CHRONIC DISEASE": chronic_disease,
        "FATIGUE": fatigue,
        "ALLERGY": allergy,
        "WHEEZING": wheezing,
        "ALCOHOL CONSUMING": alcohol_consuming,
        "COUGHING": coughing,
        "SHORTNESS OF BREATH": shortness_of_breath,
        "SWALLOWING DIFFICULTY": swallowing_difficulty,
        "CHEST PAIN": chest_pain,
    }

    return pd.DataFrame(data, index=[0])

# Get user input
input_data = user_input_features()

# Display user input
st.write("### Patient's Input:")
st.write(input_data)

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "Lung Cancer Likely" if prediction[0] == 1 else "Lung Cancer Unlikely"
    st.write("### Prediction Result:")
    st.write(result)

# Footer
st.write("""
---
Developed with ❤️ using Streamlit.
""")
