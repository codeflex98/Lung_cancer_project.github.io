import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'Lung_cancer_model.pkl')
model = joblib.load(model_path)

# Streamlit UI
st.title("Lung Cancer Prediction App")
st.write("Provide the input features to predict lung cancer.")

# Define input fields based on the model's features
def main():
    st.title("Lung Cancer Prediction App")
    # Input fields
    s1 = st.selectbox('GENDER', ("Male", "Female"))
    p1 = 2 if s1 == "Female" else 0
    
    p2 = st.slider("Enter your age", 18, 100)
    
    s3 = st.selectbox('SMOKING', ("No", "Yes"))
    p3 = 1 if s3 == "Yes" else 0
    
    s4 = st.selectbox('YELLOW_FINGERS', ("No", "Yes"))
    p4 = 2 if s4 == "Yes" else 0
    
    s5 = st.selectbox('ANXIETY', ("No", "Yes"))
    p5 = 2 if s5 == "Yes" else 0
    
    s6 = st.selectbox('PEER_PRESSURE', ("No", "Yes"))
    p6 = 2 if s6 == "Yes" else 0
    
    s7 = st.selectbox('CHRONIC DISEASE', ("No", "Yes"))
    p7 = 2 if s7 == "Yes" else 0
    
    s8 = st.selectbox('FATIGUE', ("No", "Yes"))
    p8 = 2 if s8 == "Yes" else 0
    
    s9 = st.selectbox('ALLERGY', ("No", "Yes"))
    p9 = 2 if s9 == "Yes" else 0
    
    s10 = st.selectbox('WHEEZING', ("No", "Yes"))
    p10 = 2 if s10 == "Yes" else 0
    
    s11 = st.selectbox('ALCOHOL CONSUMING', ("No", "Yes"))
    p11 = 2 if s11 == "Yes" else 0
    
    s12 = st.selectbox('COUGHING', ("No", "Yes"))
    p12 = 2 if s12 == "Yes" else 0
    
    s13 = st.selectbox('SHORTNESS OF BREATH ', ("No", "Yes"))
    p13 = 2 if s13 == "Yes" else 0
    
    s14 = st.selectbox('SWALLOWING DIFFICULTY', ("No", "Yes"))
    p14 = 2 if s14 == "Yes" else 0
    
    s15 = st.selectbox('CHEST PAIN', ("No", "Yes"))
    p15 = 2 if s15 == "Yes" else 0
    
    #print(f"Inputs: {p1}, {p2}, {p3}, {p4}, {p5}, {p6}")
    input_data = np.array([[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15]])
    print(f"Input data shape: {input_data.shape}")

    # Predict button
    if st.button("Predict"):
        prediction = model.predict(input_data)

       # Display results
        if prediction == 1:
            st.success(f"The patient has high risk of Lung cancer")
            gif_url_1 = "https://media1.tenor.com/m/n8DB4bmpduIAAAAd/yeah-bwoi-grin.gif"
            st.markdown(f'<img src="{gif_url_1}" width="600" height="400">', unsafe_allow_html=True)
        else:
            st.error(f"The patient has low risk of Lung cancer")
            gif_url_1 = "https://media.tenor.com/9kVFrGqvcwsAAAAM/fraud-troll.gif"
            st.markdown(f'<img src="{gif_url_1}" width="600" height="400">', unsafe_allow_html=True)

if __name__ == '__main__':
    main()

