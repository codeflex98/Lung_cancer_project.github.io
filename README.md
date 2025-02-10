# Lung Cancer Prediction App

## Overview
This repository contains a **Lung Cancer Prediction App** built using **Streamlit**. The app allows users to input specific health and lifestyle-related data to predict their risk of lung cancer using a pre-trained **machine learning model**.

## Features
- User-friendly interface built with Streamlit.
- Takes multiple health indicators as input.
- Utilizes a trained **ML model** to predict the likelihood of lung cancer.
- Provides visual feedback based on the prediction.

## Files in the Repository
- **Lung_cancer_predi_app.py**: Main application file containing the Streamlit-based UI and prediction logic.
- **Lung_cancer_Prediction.ipynb**: Jupyter Notebook containing the model training process.
- **requirements.txt**: List of required dependencies for running the app.

## Installation and Setup
### Prerequisites
Make sure you have **Python 3.x** installed.

### Steps to Run the Application
1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/lung_cancer_prediction.git
   cd lung_cancer_prediction
   ```
2. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Ensure the trained model (`Lung_cancer_model.pkl`) is present in the same directory.
4. Run the Streamlit app:
   ```sh
   streamlit run Lung_cancer_predi_app.py
   ```
5. The app will launch in your default web browser.

## Model Inputs
The app takes the following inputs:
- **GENDER**: Male/Female
- **Age**: Numeric input
- **SMOKING**: Yes/No
- **YELLOW FINGERS**: Yes/No
- **ANXIETY**: Yes/No
- **PEER PRESSURE**: Yes/No
- **CHRONIC DISEASE**: Yes/No
- **FATIGUE**: Yes/No
- **ALLERGY**: Yes/No
- **WHEEZING**: Yes/No
- **ALCOHOL CONSUMPTION**: Yes/No
- **COUGHING**: Yes/No
- **SHORTNESS OF BREATH**: Yes/No
- **SWALLOWING DIFFICULTY**: Yes/No
- **CHEST PAIN**: Yes/No

## Output
- **High Risk**: The patient is at a high risk of lung cancer.
- **Low Risk**: The patient is at a low risk of lung cancer.

## Dependencies
The following libraries are required:
- **pandas**
- **scikit-learn**
- **matplotlib**
- **joblib**
- **streamlit** *(Ensure it's installed separately if needed: `pip install streamlit`)*

## License
This project is licensed under the MIT License.

## Author
Developed by **Your Name**. Feel free to reach out for any inquiries.

## Acknowledgments
- Machine Learning Model trained using **scikit-learn**.
- UI built with **Streamlit**.
