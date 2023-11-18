import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf





# Load the saved label encoder
label_encoder_file = "label_encoder.pkl"
with open(label_encoder_file, 'rb') as file:
    label_encoder = pickle.load(file)

# Load the saved best model
best_model_file = "best_model .pkl"
with open(best_model_file, 'rb') as file:
    best_model = pickle.load(file)

# Compile the model
best_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Assuming X_input is your input data (e.g., user input)
# Apply the same scaling to the input features
X_input_scaled = scaler.transform(X_input)

# Make predictions using the loaded model
predictions = best_model.predict(X_input_scaled)


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




# Load the scaler
scaler_file = 'scaler.pkl'  # Update with the correct file path if needed
with open(scaler_file, 'rb') as file:
    scaler = pickle.load(file)

# Assuming X_input is your input data (e.g., user input)
# Check the shape of X_input
print("Shape of X_input:", X_input.shape)

# Check the features in X_input
print("Features in X_input:", X_input.columns)

# Apply the same scaling to the input features
X_input_scaled = scaler.transform(X_input)

# Make predictions using the loaded model
predictions = best_model.predict(X_input_scaled)




# Streamlit app
def main():
    st.title("Customer Churn Prediction")

    # User input form
    st.sidebar.header("User Input")
    user_input = {}
    for feature in top_features:
        # Append a unique identifier to the feature name for the widget ID
        widget_id = f"{feature}_input"
        user_input[feature] = st.sidebar.text_input(f"Enter {feature}", key=widget_id)

    if st.sidebar.button("Predict"):
        # Preprocess user input
        input_df = pd.DataFrame([preprocess_input(user_input)])

        # Compile the model before making predictions
        best_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Check the input data shape before making predictions
        print("Input data shape:", input_df.shape)

        try:
            # Make predictions using the best model
            prediction = best_model.predict(input_df)
            print("Prediction successful")
        except Exception as e:
            print("Prediction failed. Error:", e)
            st.error(f"Prediction failed. Error: {e}")

        # Display the preprocessed input DataFrame
        st.write("## Preprocessed Input Data")
        st.write(input_df)

        # Display the prediction
        st.write("## Prediction")
        if 'prediction' in locals():
            st.write(f"The predicted churn status is: {prediction[0]}")

if __name__ == "__main__":
    main()
