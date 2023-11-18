import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the saved label encoder
label_encoder_file = "label_encoder.pkl"
with open(label_encoder_file, 'rb') as file:
    label_encoder = pickle.load(file)

# Assuming top_features is defined somewhere in your script or loaded from a file
top_features = ['MonthlyCharges', 'tenure', 'TotalCharges', 'Contract',
                'PaymentMethod', 'OnlineSecurity', 'TechSupport', 'gender',
                'InternetService', 'OnlineBackup']  # Replace [...] with the actual definition or loading logic

# Function to preprocess user input
def preprocess_input(user_input):
    # Check if label_encoder is fitted
    if label_encoder is not None and hasattr(label_encoder, "classes_"):
        # Apply label encoding to categorical columns
        for col in top_features:
            if col in user_input and col in label_encoder.classes_:
                user_input[col] = label_encoder.transform([user_input[col]])[0]

    return user_input

# Streamlit app
def main():
    st.title("Customer Churn Prediction")

    # User input form
    st.sidebar.header("User Input")
    user_input = {}
    for feature in top_features:
        user_input[feature] = st.sidebar.text_input(f"Enter {feature}", "")

    if st.sidebar.button("Predict"):
        # Preprocess user input
        input_df = pd.DataFrame([preprocess_input(user_input)])

        # Display the preprocessed input DataFrame
        st.write("## Preprocessed Input Data")
        st.write(input_df)

if __name__ == "__main__":
    main()
