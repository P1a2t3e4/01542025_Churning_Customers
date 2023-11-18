import streamlit as st
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

with open(r'best_model (1).pkl', 'rb') as file:
    model = pickle.load(file)

with open(r'scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
def main():
    st.title('Churn Prediction App')

    # Collect user input
    tenure = st.slider("How long have you been a customer", 1, 100)
    MonthlyCharges = st.slider("What is your monthly charges", 0, 120)
    TotalCharges = st.slider("What is your total charges", 0, 10000)
    Contract = st.radio("What is your contract", ["Month-to-month", "One year", "Two year"])
    PaymentMethod = st.radio("What is your payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    OnlineSecurity = st.radio("Do you have online security", ["YES", "NO"])
    TechSupport = st.radio("Do you have tech support", ["YES", "NO"])
    InternetService = st.radio("What is your internet service", ["DSL", "Fiber optic", "No"])
    gender = st.radio("What is your gender", ["Male", "Female"])
    OnlineBackup = st.radio("Do you have online backup", ["YES", "NO"])

    # Make a prediction
    if st.button('Predict Churn'):
        # Transform user input
        user_input = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [MonthlyCharges],
            'TotalCharges': [TotalCharges],
            'Contract': [Contract],
            'OnlineSecurity': [OnlineSecurity],
            'PaymentMethod': [PaymentMethod],
            'TechSupport': [TechSupport],
            'InternetService': [InternetService],
            'gender': [gender],
            'OnlineBackup': [OnlineBackup]
        })

        # Encode categorical variables
        label_encoder = LabelEncoder()
        categorical_columns = ['Contract', 'OnlineSecurity', 'PaymentMethod', 'TechSupport', 'InternetService', 'OnlineBackup', 'gender']

        for column in categorical_columns:
            user_input[column] = label_encoder.fit_transform(user_input[column])

        # Ensure feature names match the training phase
        expected_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'OnlineSecurity', 'PaymentMethod', 'TechSupport', 'InternetService', 'gender', 'OnlineBackup']
        assert user_input.columns.tolist() == expected_features, "Feature names do not match"

        # Scale the input
        scaled_input = scaler.transform(user_input)

        # Make a prediction
        prediction = model.predict(scaled_input)

        # Display the result
        churn_probability = prediction[0]
        churn_prediction = 'Yes, Customer will Churn' if churn_probability >= 0.5 else 'Customer will not Churn'
        st.write(f'Churn Probability: {churn_probability}')
        st.write(f'Prediction: {churn_prediction}')

if __name__ == '__main__':
    main()
