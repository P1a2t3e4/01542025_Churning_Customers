import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the best model and preprocessing objects
best_model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load the label encoder only if it exists (check if the file exists)
label_encoder_file = "label_encoder.pkl"
if os.path.exists(label_encoder_file):
    label_encoder = pickle.load(open(label_encoder_file, "rb"))
else:
    st.error("Label encoder file not found. Make sure to fit and save the label encoder during training.")
    st.stop()

# Assuming top_features is defined somewhere in your script or loaded from a file
top_features = ['MonthlyCharges', 'tenure', 'TotalCharges', 'Contract',
       'PaymentMethod', 'OnlineSecurity', 'TechSupport', 'gender',
       'InternetService', 'OnlineBackup']  # Replace [...] with the actual definition or loading logic

# Function to preprocess user input
def preprocess_input(user_input):
    # Assuming user_input is a dictionary with keys as column names
    input_df = pd.DataFrame([user_input])
    
    # Check if label_encoder is fitted
    if label_encoder is not None and hasattr(label_encoder, "classes_"):
        # Apply label encoding to categorical columns
        for col in input_df.select_dtypes(include=['object']).columns:
            input_df[col] = label_encoder.transform(input_df[col])
    else:
        st.error("Label encoder is not fitted. Make sure to fit and save the label encoder during training.")
        st.stop()

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


