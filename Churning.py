import requests
import pickle
import streamlit as st
import numpy as np
import pandas as pd

with open(r'best_model .pkl','rb') as file:
    best_module = pickle.load(file)
with open(r'scaler.pkl','rb') as file:
    scaler_module = pickle.load(file)

st.set_page_config(
    page_title="CUSTOMER CHURN PREDICTIONS",
    page_icon=":wave:",
    layout="centered",
    initial_sidebar_state="expanded",
  
)

st.sidebar.header("Enter Customer Details", divider='rainbow')
MonthlyCharges = st.slider("What is your monthly charges",0,120)
tenure = st.slider("How long have you been a customer",1,100)
TotalCharges = st.slider("What is your total charges",0,10000)
Contract = st.radio("What is your contract",["Month-to-month","One year","Two year"])
PaymentMethod = st.radio("What is your payment method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
OnlineSecurity = st.radio("Do you have online security", ["YES", "NO"])
TechSupport = st.radio("Do you have tech support",["YES","NO"])
gender = st.radio("What is your gender", ["Male", "Female"])
InternetService = st.radio("What is your internet service",["DSL","Fiber optic","No"])
OnlineBackup = st.radio("Do you have online backup",["YES","NO"])





submit =st.button("Submit")
user_response = ['MonthlyCharges', 'tenure', 'TotalCharges', 'Contract',
       'PaymentMethod', 'OnlineSecurity', 'TechSupport', 'gender',
       'InternetService', 'OnlineBackup']

attributes = ['MonthlyCharges', 'tenure', 'TotalCharges', 'Contract',
       'PaymentMethod', 'OnlineSecurity', 'TechSupport', 'gender',
       'InternetService', 'OnlineBackup']
if submit:
        
    # Assuming numerical_columns contains the names of numerical columns
        #newData[numerical_columns] = newData[numerical_columns].apply(pd.to_numeric, errors='coerce')
        
        # Assuming original_data_shape is a tuple representing the shape of the original data
        #if newData.shape[1] != original_data_shape[1]:
            #raise ValueError("Number of columns in newData does not match the original data.")


        #print(newData.shape)  # Print the shape of newData

        #newData = newData.astype(float)  # Convert all columns to float, for example

       

        
        #scaler_module.fit(original_data)
        #newData = scaler_module.transform(newData)



        newData = pd.DataFrame([user_response],columns= attributes)
        scaledData = scaler_module.transform(newData)
        predict = best_module.predict(scaledData)
        print(scaledData)
        st.subheader("The customer churn rate is "+str( round(predict[0])),divider='rainbow')
        print(predict[0]+(0.8809915725973115*1.96))
        print(predict[0]+(0.8809915725973115-1.96))
        st.text("There's a 95% chance that the rating is between " + str(round(predict[0]+(0.8809915725973115-1.96))) +" and "+str(round(predict[0]+(0.8809915725973115*1.96))))
