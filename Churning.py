import streamlit as st
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Load the trained model
model = load_model('best_model.h5')

# Load the trained StandardScaler
with open('scaler.pkl', 'rb') as file:
    scaled = pickle.load(file)

# Streamlit app
def main():
    st.title('Churn Prediction App')

    # Collect user input
    tenure = st.slider('Tenure', 0, 70, 20)
    monthly_charges = st.slider('Monthly Charges', 0.0, 200.0, 100.0)
    total_charges = st.slider('Total Charges', 0.0, 5000.0, 2500.0)
    contract = st.selectbox('Contract', ['month-to-month', 'one year', 'Two years'])
    online_security = st.selectbox('Online Security', ['No', 'yes', 'No internet service'])
    payment_method = st.selectbox('Payment Method', ['electronic check', 'mailed check', 'card'])
    tech_support = st.selectbox('Tech Support', ['No', 'yes', 'No internet service'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'fiber optic', 'No'])
    online_backup = st.selectbox('Online Backup', ['No', 'yes', 'No internet service'])
    gender = st.radio('Gender', ['Male', 'Female'])

    # Make a prediction
    if st.button('Predict Churn'):
        # Transform user input
        user_input = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'Contract': [contract],
            'OnlineSecurity': [online_security],
            'PaymentMethod': [payment_method],
            'TechSupport': [tech_support],
            'InternetService': [internet_service],
            'OnlineBackup': [online_backup],
            'gender': [gender]
        })

        # Encode categorical variables
        label_encoder = LabelEncoder()
        categorical_columns = ['Contract', 'OnlineSecurity', 'PaymentMethod', 'TechSupport', 'InternetService', 'OnlineBackup', 'gender']
        
        for column in categorical_columns:
            user_input[column] = label_encoder.fit_transform(user_input[column])

        # Scale the input
        scaled_input = scaled.transform(user_input)

        # Make a prediction
        prediction = model.predict(scaled_input)

        # Display the result
        churn_probability = prediction[0]
        churn_prediction = 'Yes, Customer will Churn' if churn_probability >= 0.5 else 'Customer will not Churn'
        st.write(f'Churn Probability: {churn_probability}')
        st.write(f'Prediction: {churn_prediction}')

if _name_ == '_main_':
    main()
