
import requests
import pickle
import streamlit as st
import numpy as np
import pandas as pd
#from tensorflow.keras.models import load_model

with open(r'best_model.pkl','rb') as file:
    football_module = pickle.load(file)
with open(r'scaler_model.joblib','rb') as file:
    scaler_module =joblib.load(file)

st.set_page_config(
    page_title="CUSTOMER CHURN PREDICTIONS",
    page_icon=":wave:",
    layout="centered",
    initial_sidebar_state="expanded",
  
)

st.sidebar.header("Enter Customer Details", divider='rainbow')
tenure = st.slider('Tenure', 0, 90, 50)
monthly_charges = st.slider('Monthly Charges', 0.0, 200.0, 100.0)
total_charges = st.slider('Total Charges', 0.0, 5000.0, 2500.0)
contract = st.selectbox('Contract', ['month-to-month', 'One year', 'Two year'])
online_security = st.selectbox('Online Security', ['No', 'yes', 'No internet service'])
payment_method = st.selectbox('Payment Method', ['electronic check', 'mailed check', 'card'])
tech_support = st.selectbox('Tech Support', ['No', 'yes', 'No internet service'])
internet_service = st.selectbox('Internet Service', ['DSL', 'fiber optic', 'No'])
gender = st.radio('Gender', ['Male', 'Female'])
online_backup = st.selectbox('Online Backup', ['No', 'yes', 'No internet service'])
            

        
submit =st.button("Submit")
user_response = [gender,SeniorCitizen,tenure,InternetService,OnlineSecurity,
                 OnlineBackup,TechSupport,Contract,PaymentMethod,MonthlyCharges,TotalCharges]

attributes = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract',
       'PaymentMethod', 'TechSupport', 'OnlineSecurity', 'gender',
       'OnlineBackup', 'InternetService']
if submit:
        newData = pd.DataFrame([user_response],columns= attributes)
        scaledData = scaler_module.transform(newData)
        predict = football_module.predict(scaledData)
        print(scaledData)
        st.subheader("The customer churn rate is "+str( round(predict[0])),divider='rainbow')
        print(predict[0]+(0.8809915725973115*1.96))
        print(predict[0]+(0.8809915725973115-1.96))
        st.text("There's a 95% chance that the rating is between " + str(round(predict[0]+(0.8809915725973115-1.96))) +" and "+str(round(predict[0]+(0.8809915725973115*1.96))))
