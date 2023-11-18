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
    st.title('Churning Customer  Prediction App')

    # Collect user input
    tenure = st.slider("What is your tenure", 1, 100)
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
    
       
      
    # Transform user input


     class CustomScaler(StandardScaler):
         
def __init__(self, input_features=None, **kwargs):
    super().__init__(**kwargs)
    self.input_features = input_features

def fit(self, X, y=None):
     if self.input_features:
        X = X[self.input_features]
        return super().fit(X, y)

def transform(self, X, y=None, **kwargs):
    if self.input_features:
        
        X = X[self.input_features]
        return super().transform(X, y, **kwargs)

def fit_transform(self, X, y=None, **kwargs):
    if self.input_features:
        X = X[self.input_features]
        return super().fit_transform(X, y, **kwargs)
   

        # Encode categorical variables
        label_encoder = LabelEncoder()
        categorical_columns = ['Contract', 'OnlineSecurity', 'PaymentMethod', 'TechSupport', 'InternetService', 'OnlineBackup', 'gender']

        for column in categorical_columns:
            user_input[column] = label_encoder.fit_transform(user_input[column])

        # Set feature names for the scaler
        user_input_columns = user_input.columns
        scaler_input_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'OnlineSecurity', 'PaymentMethod', 'TechSupport', 'InternetService', 'gender', 'OnlineBackup']
        scaler.set_params(input_features=scaler_input_features)

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
