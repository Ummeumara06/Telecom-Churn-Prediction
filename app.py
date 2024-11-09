# app/app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Function to get absolute path relative to this file


def get_absolute_path(relative_path):
    return os.path.join(os.path.dirname(__file__), relative_path)


# Load the trained model and scaler
try:
    model = joblib.load(get_absolute_path("C:/Users/khanu/Downloads/microsoft edge/my project/models/random_forest_churn_model.pkl"))
    scaler = joblib.load(get_absolute_path("C:/Users/khanu/Downloads/microsoft edge/my project/models/scaler.pkl"))
    state_le = joblib.load(get_absolute_path("C:/Users/khanu/Downloads/microsoft edge/my project/models/label_encoder_state.pkl"))
    area_code_le = joblib.load(get_absolute_path("C:/Users/khanu/Downloads/microsoft edge/my project/models/label_encoder_area_code.pkl"))
    feature_names = joblib.load(get_absolute_path("C:/Users/khanu/Downloads/microsoft edge/my project/models/features.pkl"))
except Exception as e:
    st.error(f"Error loading models or encoders: {e}")
    st.stop()

# Load unique states and area codes for dropdowns
try:
    df = pd.read_excel(get_absolute_path("C:/Users/khanu/Documents/classexcelr/Project/Churn.xlsx"))
    df = df.drop(columns=['Unnamed: 0'])
    unique_states = sorted(df['state'].unique())
    unique_area_codes = sorted(df['area.code'].unique())
except Exception as e:
    st.error(f"Error loading data for dropdowns: {e}")
    st.stop()

# Streamlit App


def main():
    st.set_page_config(
        page_title="Customer Churn Prediction", layout="centered")
    st.title("📈 Customer Churn Prediction")
    st.write("""
    ### Predict whether a customer will churn based on their account details and usage.
    Please provide the following information:
    """)

    # Collect user input with better layout
    col1, col2 = st.columns(2)

    with col1:
        state = st.selectbox("**State**", options=unique_states)
    with col2:
        area_code = st.selectbox("**Area Code**", options=unique_area_codes)

    account_length = st.number_input(
        "**Account Length (months)**", min_value=0, max_value=100, value=12)

    voice_plan = st.selectbox("**Voice Plan**", options=['yes', 'no'])
    voice_plan_yes = 1 if voice_plan == 'yes' else 0

    intl_plan = st.selectbox("**International Plan**", options=['yes', 'no'])
    intl_plan_yes = 1 if intl_plan == 'yes' else 0

    voice_messages = st.number_input(
        "**Number of Voicemail Messages**", min_value=0, max_value=1000, value=100)

    intl_mins = st.number_input(
        "**International Minutes**", min_value=0.0, max_value=1000.0, value=50.0)
    intl_calls = st.number_input(
        "**International Calls**", min_value=0, max_value=1000, value=10)

    day_mins = st.number_input(
        "**Day Minutes**", min_value=0.0, max_value=1000.0, value=200.0)
    day_calls = st.number_input(
        "**Day Calls**", min_value=0, max_value=1000, value=50)

    eve_mins = st.number_input(
        "**Evening Minutes**", min_value=0.0, max_value=1000.0, value=150.0)
    eve_calls = st.number_input(
        "**Evening Calls**", min_value=0, max_value=1000, value=40)

    night_mins = st.number_input(
        "**Night Minutes**", min_value=0.0, max_value=1000.0, value=100.0)
    night_calls = st.number_input(
        "**Night Calls**", min_value=0, max_value=1000, value=30)

    # Encode categorical variables
    try:
        state_encoded = state_le.transform([state])[0]
    except Exception as e:
        st.error("Selected state is not recognized. Please check the data.")
        return

    try:
        area_code_encoded = area_code_le.transform([area_code])[0]
    except Exception as e:
        st.error("Selected area code is not recognized. Please check the data.")
        return

    # Create a DataFrame for prediction
    input_data = pd.DataFrame({
        'account.length': [account_length],
        'voice.messages': [voice_messages],
        'intl.mins': [intl_mins],
        'intl.calls': [intl_calls],
        'day.mins': [day_mins],
        'day.calls': [day_calls],
        'eve.mins': [eve_mins],
        'eve.calls': [eve_calls],
        'night.mins': [night_mins],
        'night.calls': [night_calls],
        'state': [state_encoded],
        'voice.plan_yes': [voice_plan_yes],
        'intl.plan_yes': [intl_plan_yes],
    })

    # Reorder the columns to match the training data
    try:
        input_data = input_data[feature_names]
    except Exception as e:
        st.error(f"Error reordering input data columns: {e}")
        st.write("**Expected Features:**", feature_names)
        st.write("**Input Data Columns:**", input_data.columns.tolist())
        return

    if st.button("🔮 Predict"):
        with st.spinner("Predicting..."):
            try:
                # Preprocess the input
                input_scaled = scaler.transform(input_data)

                # Make prediction
                prediction = model.predict(input_scaled)
                prediction_proba = model.predict_proba(input_scaled)

                # Display the results
                if prediction[0] == 1:
                    st.error(f"**Churn Prediction:** The customer is likely to **churn** with a probability of {
                             prediction_proba[0][1]*100:.2f}%")
                else:
                    st.success(f"**Churn Prediction:** The customer is likely to **stay** with a probability of {
                               prediction_proba[0][0]*100:.2f}%")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.write("**Feature Names:**", feature_names)
                st.write("**Input Data Columns:**",
                         input_data.columns.tolist())

    st.markdown("---")
    # st.write("Developed by Umara")


if __name__ == '__main__':
    main()
