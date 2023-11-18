import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the best model and preprocessing objects
best_model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Function to preprocess user input
def preprocess_input(user_input):
    # Assuming user_input is a dictionary with keys as column names
    input_df = pd.DataFrame([user_input])
    
    # Apply label encoding to categorical columns
    for col in input_df.select_dtypes(include=['object']).columns:
        input_df[col] = label_encoder.transform(input_df[col])
    
    # Scale numerical features using the previously trained scaler
    input_df[top_features] = scaler.transform(input_df[top_features])

    return input_df

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
        input_df = preprocess_input(user_input)

        # Make predictions using the best model
        prediction = best_model.predict(input_df)

        # Display the prediction
        st.write("## Prediction")
        st.write(f"The predicted churn status is: {prediction[0]}")

if __name__ == "__main__":
    main()
