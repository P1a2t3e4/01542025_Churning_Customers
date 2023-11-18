import requests
import pickle
import streamlit as st
import numpy as np
import pandas as pd

# Load the trained model
with open(r'best_model .pkl', 'rb') as file:
    best_module = pickle.load(file)

# Load the trained scaler
with open(r'scaler.pkl', 'rb') as file:
    scaler_module = pickle.load(file)

# Set Streamlit page configuration
st.set_page_config(
    page_title="CUSTOMER CHURN PREDICTIONS",
    page_icon=":wave:",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Streamlit sidebar for user input
st.sidebar.header("Enter Customer Details", divider='rainbow')
MonthlyCharges = st.slider("What is your monthly charges", 0, 120)
tenure = st.slider("How long have you been a customer", 1, 100)
TotalCharges = st.slider("What is your total charges", 0, 10000)
Contract = st.radio("What is your contract", ["Month-to-month", "One year", "Two year"])
PaymentMethod = st.radio("What is your payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
OnlineSecurity = st.radio("Do you have online security", ["YES", "NO"])
TechSupport = st.radio("Do you have tech support", ["YES", "NO"])
gender = st.radio("What is your gender", ["Male", "Female"])
InternetService = st.radio("What is your internet service", ["DSL", "Fiber optic", "No"])
OnlineBackup = st.radio("Do you have online backup", ["YES", "NO"])

submit = st.button("Submit")

# If the user clicks Submit
if submit:
    user_response = {
        'MonthlyCharges': MonthlyCharges,
        'tenure': tenure,
        'TotalCharges': TotalCharges,
        'Contract': Contract,
        'PaymentMethod': PaymentMethod,
        'OnlineSecurity': OnlineSecurity,
        'TechSupport': TechSupport,
        'gender': gender,
        'InternetService': InternetService,
        'OnlineBackup': OnlineBackup
        
    }

    #print(user_data.dtypes)

    # Assuming categorical_columns contains the names of categorical columns
    user_data[categorical_columns] = user_data[categorical_columns].astype('category')

    print(user_data.isnull().sum())

    # Assuming numerical_columns contains the names of numerical columns
    user_data[numerical_columns] = user_data[numerical_columns].apply(pd.to_numeric, errors='coerce')


    # Create a DataFrame with user input
    user_data = pd.DataFrame([user_response])

    # Ensure that columns have the correct data types
    user_data = user_data.astype({'MonthlyCharges': float, 'tenure': float, 'TotalCharges': float})

    # Transform user input with the scaler
    scaled_data = scaler_module.transform(user_data)

    # Make prediction with the model
    predict = best_module.predict(scaled_data)

    # Display the prediction
    st.subheader(f"The customer churn rate is {round(predict[0], 2)}", divider='rainbow')
