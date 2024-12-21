# scripts/train_model.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Ensure the models directory exists
os.makedirs("../models", exist_ok=True)

# Load the dataset
df = pd.read_excel("../data/Churn.xlsx")

# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 0'])

# Handle missing values by converting to numeric and dropping rows with NaNs in 'day.charge' and 'eve.mins'
df['day.charge'] = pd.to_numeric(df['day.charge'], errors='coerce')
df['eve.mins'] = pd.to_numeric(df['eve.mins'], errors='coerce')
df = df.dropna(subset=['day.charge', 'eve.mins'])

# Drop charge columns as they might not be needed for prediction
df.drop(columns=['intl.charge', 'eve.charge',
        'night.charge', 'day.charge'], inplace=True)

# Encode categorical variables with separate LabelEncoders
state_le = LabelEncoder()
df['state'] = state_le.fit_transform(df['state'])
joblib.dump(state_le, "../models/label_encoder_state.pkl")
print("State LabelEncoder saved as label_encoder_state.pkl")

area_code_le = LabelEncoder()
df['area.code'] = area_code_le.fit_transform(df['area.code'])
joblib.dump(area_code_le, "../models/label_encoder_area_code.pkl")
print("Area Code LabelEncoder saved as label_encoder_area_code.pkl")

# One-hot encode 'voice.plan' and 'intl.plan'
df = pd.get_dummies(df, columns=['voice.plan', 'intl.plan'], drop_first=True)

# Encode target variable 'churn'
df['churn'] = df['churn'].map({'yes': 1, 'no': 0})

# Feature selection: dropping less important features
df.drop(columns=['area.code', 'customer.calls'], inplace=True)

# Define features and target
X = df.drop('churn', axis=1)
y = df['churn']

# Save the feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, "../models/features.pkl")
print("Feature names saved as features.pkl")

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, "../models/scaler.pkl")
print("Scaler saved as scaler.pkl")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(rf_model, "../models/random_forest_churn_model.pkl")
print("Model saved as random_forest_churn_model.pkl")
