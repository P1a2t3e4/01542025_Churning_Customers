01542025_Churning_Customers 
Churning Customers Prediction App

Introduction
This Streamlit web application is designed to predict the churn rate of customers using a machine learning model. It offers a user-friendly interface for entering customer data and quickly obtaining churn predictions.

Features
User-Friendly Interface Streamlit sliders and selectors for easy input of customer details.
Predictive Analytics: Utilizes a pickled machine learning model to forecast customer churn rates.
*Scalable Infrastructure Implements joblib scaler for input data normalization, ensuring reliable prediction outputs.

Repository Contents
Churn.py The main Python script that runs the Streamlit app.
best_model.pkl Serialized version of the machine learning model used for predictions.
scaler_model.joblib The scaling model to normalize input features for prediction.
requirements.txt  A list of Python libraries required to run the app.

Installation & Usage
To run the application locally, clone the repository, install the dependencies listed in `requirements.txt`, and execute `streamlit run Churn.py`.

 How It Works
Fill in the customer details in the designated sidebar section.
The app takes in parameters such as tenure, monthly charges, and service details.
 After inputting the data, click "Submit" to receive the churn prediction and confidence intervals.

Contribution
Feel free to fork this repository, submit pull requests, or send suggestions to improve the application.
