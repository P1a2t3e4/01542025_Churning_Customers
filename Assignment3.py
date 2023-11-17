import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('churn_model.h5')

# Load data for the additional features
data = pd.read_csv('"C:\Users\Patricia Okletey\Downloads\CustomerChurn_dataset.csv"')

# Select the relevant features
selected_features = ['MonthlyCharges', 'tenure', 'TotalCharges', 'Contract', 'PaymentMethod', 'TechSupport', 'OnlineSecurity', 'gender', 'OnlineBackup', 'InternetService']
X_additional = data[selected_features]

# Preprocess the additional data
scaler = StandardScaler()
X_additional_scaled = scaler.fit_transform(X_additional)

# Evaluate the model
y_pred = model.predict(X_additional_scaled)
y_pred_binary = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_binary)  # Note: You need to define y_test for this evaluation
auc_score = roc_auc_score(y_test, y_pred)  # Note: You need to define y_test for this evaluation

# Streamlit App
st.title('Churn Prediction App')

# Display evaluation results
st.write(f'Accuracy: {accuracy}')
st.write(f'AUC Score: {auc_score}')
