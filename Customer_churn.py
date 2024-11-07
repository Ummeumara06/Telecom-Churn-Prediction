# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:15:12 2024

@author: khanu
"""

import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model
model = pickle.load(open("C:/Users/khanu/Downloads/microsoft edge/smote_trained_model.sav", 'rb'))

# Define the columns to drop (based on training preprocessing)
columns_to_drop = ['intl.charge', 'eve.charge', 'night.charge', 'day.charge','area.code','customer.calls']

# Define scalers based on typical scaling process (mean, std).
# Note: Replace these values with the exact means and stds used during training, if known.
scaler = StandardScaler()
#scaler.mean_ = [70, 10, 5, 200, 200, 200, 1]  # Example values for mean of features
#scaler.scale_ = [30, 5, 3, 100, 100, 100, 2]   # Example values for std of features

# App title
st.title("Telecom Churn Prediction")

# User input function for customer data
def user_input_features():
    # Collecting user inputs
    state = st.selectbox('State', options=list(range(52)))  # Assume numerical state encoding
    area_code = st.selectbox('Area Code', options=[408, 415, 510])
    account_length = st.slider('Account Length', 0, 300, 100)
    voice_plan = st.selectbox('Voice Plan', options=['yes', 'no'])
    voice_messages = st.slider('Voice Messages', 0, 50, 0)
    intl_plan = st.selectbox('International Plan', options=['yes', 'no'])
    intl_mins = st.slider('International Minutes', 0, 50, 0)
    intl_calls = st.slider('International Calls', 0, 20, 0)
    day_mins = st.slider('Day Minutes', 0, 500, 250)
    eve_mins = st.slider('Evening Minutes', 0, 500, 250)
    night_mins = st.slider('Night Minutes', 0, 500, 250)
    customer_calls = st.slider('Customer Service Calls', 0, 10, 0)

    # Dictionary of raw inputs
    data = {
        'state': state,
        'area.code': area_code,
        'account.length': account_length,
        'voice.plan': voice_plan,
        'voice.messages': voice_messages,
        'intl.plan': intl_plan,
        'intl.mins': intl_mins,
        'intl.calls': intl_calls,
        'day.mins': day_mins,
        'eve.mins': eve_mins,
        'night.mins': night_mins,
        'customer.calls': customer_calls
    }

    return pd.DataFrame(data, index=[0])

# Preprocess input data function
def preprocess_input(df):
    # label encoding
    le= LabelEncoder()
    # Apply label encoding to 'state' and 'area.code'
    df['state'] = le.fit_transform(df['state'])
    df['area.code'] = le.fit_transform(df['area.code'])
    
    # apply one hot encoding to 'voice.plan', 'intl.plan','churn'
    df = pd.get_dummies(df, columns=['voice.plan', 'intl.plan'], drop_first=True)
    
    # Drop unnecessary columns
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Apply scaling
    numerical_columns = df.select_dtypes(include=['int64','float64']).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Reorder columns to match the trained model
    expected_columns = model.feature_names_in_
    df = df.reindex(columns=expected_columns, fill_value=0)

    return df

# Display user inputs
input_df = user_input_features()
st.write("Input customer data (pre-processed):", input_df)

# Preprocess the user input and make predictions
processed_input = preprocess_input(input_df)

# Predict and display results
#'''if st.button("Predict Churn"):
   # prediction = model.predict(processed_input)
    #result = "Churn" if prediction[0] == 1 else "No Churn"
    #st.subheader(f"Prediction: {result}")'''
    
    
#'''if st.button("Predict Churn"):
 #   probability = model.predict_proba(processed_input)
  #  st.write(f"Churn Probability: {probability[0][1]:.2f}")
   # prediction = model.predict(processed_input)
    #result = "Churn" if prediction[0] == 1 else "No Churn"
    #st.subheader(f"Prediction: {result}")'''

if st.button("Predict Churn"):
    probability = model.predict_proba(processed_input)
    churn_threshold = 0.3  # Set a custom threshold, e.g., 20%
    result = "Churn" if probability[0][1] > churn_threshold else "No Churn"
    st.subheader(f"Prediction: {result}")
    st.write(f"Churn Probability: {probability[0][1]:.2f}")
