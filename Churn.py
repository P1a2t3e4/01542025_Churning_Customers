

# import streamlit as st
# import pandas as pd
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import pickle

# # Load the trained model
# model = load_model('best_model.pkl','rb')

# # Load the trained StandardScaler
# with open('scaler_model.joblib', 'rb') as file:
#     scaled = joblib.load(file)

# # Streamlit app
# def main():
#     st.title('Churning  Customer  App')

#     # Collect user input
#     tenure = st.slider('Tenure', 0, 90, 50)
#     monthly_charges = st.slider('Monthly Charges', 0.0, 200.0, 100.0)
#     total_charges = st.slider('Total Charges', 0.0, 5000.0, 2500.0)
#     contract = st.selectbox('Contract', ['month-to-month', 'One year', 'Two year'])
#     online_security = st.selectbox('Online Security', ['No', 'yes', 'No internet service'])
#     payment_method = st.selectbox('Payment Method', ['electronic check', 'mailed check', 'card'])
#     tech_support = st.selectbox('Tech Support', ['No', 'yes', 'No internet service'])
#     internet_service = st.selectbox('Internet Service', ['DSL', 'fiber optic', 'No'])
#     gender = st.radio('Gender', ['Male', 'Female'])
#     online_backup = st.selectbox('Online Backup', ['No', 'yes', 'No internet service'])
   
#     # Make a prediction
#     if st.button('Feature values for prediction of churn'):
#         # Transform user input
#         user_input = pd.DataFrame({
#             'tenure': [tenure],
#             'MonthlyCharges': [monthly_charges],
#             'TotalCharges': [total_charges],
#             'Contract': [contract],
#             'OnlineSecurity': [online_security],
#             'PaymentMethod': [payment_method],
#             'TechSupport': [tech_support],
#             'InternetService': [internet_service],
#             'OnlineBackup': [online_backup],
#             'gender': [gender]
#         })

#         # Encode categorical variables
#         label_encoder = LabelEncoder()
#         categorical_columns = ['Contract', 'OnlineSecurity', 'PaymentMethod', 'TechSupport', 'InternetService', 'OnlineBackup', 'gender']
        
#         for column in categorical_columns:
#             user_input[column] = label_encoder.fit_transform(user_input[column])

#         # Scale the input
#         scaled_input = scaled.transform(user_input)

#         # Make a prediction
#         prediction = model.predict(scaled_input)

#         # Display the result
#         churn_probability = prediction[0]
#         churn_prediction = 'Yes, ' if churn_probability >= 0.5 else 'No will not  Churn'
#         st.write(f'Churn Probability: {churn_probability}')
#         st.write(f'prediction: {churn_prediction}')

# if _name_ == '_main_':
#     main()
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
