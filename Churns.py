import requests
import pickle
import streamlit as st
import numpy as np
import pandas as pd

with open(r'best_model (1).pkl', 'rb') as file:
    model = pickle.load(file)

with open(r'scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.set_page_config(
    page_title="CUSTOMER CHURN PREDICTIONS",
    page_icon=":wave:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.sidebar.header("Enter Customer Details", divider='rainbow')

tenure = st.slider("How long have you been a customer", 1, 100)
MonthlyCharges = st.slider("What is your monthly charges", 0, 120)
TotalCharges = st.slider("What is your total charges", 0, 10000)
Contract = st.radio("What is your contract", ["Month-to-month", "One year", "Two year"])
PaymentMethod = st.radio("What is your payment method",
                        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
OnlineSecurity = st.radio("Do you have online security", ["YES", "NO"])
TechSupport = st.radio("Do you have tech support", ["YES", "NO"])
InternetService = st.radio("What is your internet service", ["DSL", "Fiber optic", "No"])
gender = st.radio("What is your gender", ["Male", "Female"])
OnlineBackup = st.radio("Do you have online backup", ["YES", "NO"])

submit = st.button("Submit")

user_response = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract',
                 'PaymentMethod', 'OnlineSecurity', 'TechSupport',
                 'InternetService', 'gender', 'OnlineBackup']

attributes = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract',
              'PaymentMethod', 'OnlineSecurity', 'TechSupport',
              'InternetService', 'gender', 'OnlineBackup']

if submit:
    newData = pd.DataFrame([user_response], columns=attributes)
    scaledData = scaler.transform(newData)
    predict = model.predict(scaledData)
    st.subheader("The customer churn rate is " + str(round(predict[0])), divider='rainbow')
    st.text("There's a 95% chance that the rating is between " +
            str(round(predict[0] - (0.8809915725973115 * 1.96))) +
            " and " +
            str(round(predict[0] + (0.8809915725973115 * 1.96))))
