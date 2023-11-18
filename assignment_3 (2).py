# -*- coding: utf-8 -*-
"""Assignment 3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HX2EB5XpjAKVSd8QkKXEEqHrqLH6rB67
"""

import pandas as  pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV
from google.colab import drive

drive.mount('/content/drive')
df= pd.read_csv (r"/content/drive/MyDrive/CustomerChurn_dataset.csv")

#df= pd.read_csv ()

# Creating a variable with the name "df" that keeps the data
df

# Removing the column titled CustomerID
df.drop(columns='customerID', inplace=True)

# Checking for values that have misssing values greater than 30 %
missing_percentage = (df.isnull().mean() * 100)

# We defined a threshold of 30% for the maximum allowed missing values
threshold = 30

# To get the list of columns with missing values exceeding the threshold
columns_to_drop = missing_percentage[missing_percentage > threshold].index.tolist()

# dropping columns with excessive missing values from the DataFrame
df.drop(columns=columns_to_drop, inplace=True)
df

#Checking there are missing values in there
print(df.isnull().sum())

import seaborn as sns

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y=' The Features', data=feature_importance_data)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# # Assuming df is your DataFrame
# # Separate the numerical and categorical columns
# numerical_columns = df.select_dtypes(include=['number']).columns
# categorical_columns = df.select_dtypes(exclude=['number']).columns

# # Initializing the LabelEncoder
# label_encoder = LabelEncoder()

# # Looping through each column in the imputed categorical data
# for col in categorical_columns:
#     df[col] = label_encoder.fit_transform(df[col])

# # Display the updated DataFrame
# df.head()

import pickle
from sklearn.preprocessing import LabelEncoder

# Assuming df is your DataFrame
# Separate the numerical and categorical columns
numerical_columns = df.select_dtypes(include=['number']).columns
categorical_columns = df.select_dtypes(exclude=['number']).columns

# Initializing the LabelEncoder
label_encoder = LabelEncoder()

# Looping through each column in the imputed categorical data
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Save the label encoder using pickle
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

# To download the file in Google Colab
from google.colab import files

files.download('label_encoder.pkl')

# Combine the DataFrame
df_combined = df[numerical_columns.union(categorical_columns)]

# Display the updated DataFrame
df_combined.head()

# Split target and feature variables
y = df_combined['Churn']
X = df_combined.drop('Churn',axis=1)

# Scaling the data values for training
sc=StandardScaler()
scaled_data = sc.fit_transform(X)

# Transform it into a dataframe
newData= pd.DataFrame(scaled_data, columns=X.columns)
newData.head()

the_model= RandomForestClassifier()
the_model.fit(newData,y)

#  Getting important features in the dataset
feature = X.columns
feature_importance =the_model.feature_importances_

# Sorting the feature importance
feature_importance_data = pd.DataFrame({' The Feature': feature, 'Importance': feature_importance})

# Sorting the features in descending order
feature_importance_data= feature_importance_data.sort_values(by='Importance', ascending=False)
feature_importance_data

#selecting the top 8 feature
top_features = feature_importance_data[' The Feature'].values[:10]
top_features

import  matplotlib.pyplot as plt

print('The shape of the dataset : ', df.shape)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

"""EXPLANATORY DATA ANALYSIS"""

import matplotlib.pyplot as plt
import seaborn as sns

# Plot numerical features distribution based on Churn
num_features = numerical_columns.tolist() + ['Churn']
sns.pairplot(df_combined[num_features], hue='Churn', diag_kind='kde')
plt.suptitle('Numerical Features Distribution by Churn Status', y=1.02)
plt.show()

# Plot categorical features distribution based on Churn
for col in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col, hue='Churn', data=df)
    plt.title(f'{col} Distribution by Churn')
    plt.show()

"""**MULTI_PLAYER PERCEPTRON - API**"""

#Multi_player Perceptron model using functional API
import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight

# Standardize the input features
# scaler = StandardScaler()
# Xtrain_scaled = scaler.fit_transform(X[top_features])
# Xscaled = pd.DataFrame(Xtrain_scaled, columns= X[top_features].columns)
# Xscaled

# import joblib
# from joblib import dump
# joblib.dump(scaler, 'scaler_model.joblib')

import pickle
from sklearn.preprocessing import StandardScaler

# Assuming X is your feature matrix and top_features is a list of column names
scaler = StandardScaler()
Xtrain_scaled = scaler.fit_transform(X[top_features])
Xscaled = pd.DataFrame(Xtrain_scaled, columns=X[top_features].columns)

# Save the scaler using pickle
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# To download the file in Google Colab
from google.colab import files

files.download('scaler.pkl')

# Split the data into training (80%), validation (10%), and testing (10%)
Xtrain, X_temp, Ytrain, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

"""MODEL'S ACCURACY AND AUC"""

!pip install tensorflow scikeras scikit-learn

"""HOSTING THE MODEL"""

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

from sklearn import metrics

num_classes=1
epochs=50
batch_size=10

import tensorflow as tf


def create_model(dropout_rate, weight_constraint,neurons,activation,num_classes,X_Corr):
  # create modeloptimizer=optimizer
  input_shape = (X_Corr.shape[1],)
  inputs = tf.keras.Input(shape=input_shape)
  input = tf.keras.layers.Dense((28)+neurons, activation=activation)(inputs)
  x= tf.keras.layers.Dropout(dropout_rate)(input)
  second=tf.keras.layers.Dense((12)+neurons, activation=activation)(x)
  x= tf.keras.layers.Dropout(dropout_rate)(second)
  third=tf.keras.layers.Dense((4)+neurons, activation=activation)(x)
  x= tf.keras.layers.Dropout(dropout_rate)(third)
  fourth=tf.keras.layers.Dense((-4)+neurons, activation=activation)(x)
  x= tf.keras.layers.Dropout(dropout_rate)(fourth)
  fifth=tf.keras.layers.Dense((-12)+neurons, activation=activation)(x)
# Add more hidden layers if necessary

# Add output layer with softmax activation
  outputs = tf.keras.layers.Dense(num_classes, activation='sigmoid')(fifth)

# Create the model
  model_optim = tf.keras.Model(inputs=inputs, outputs=outputs)
  model_optim.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
  return  model_optim

def create_model(dropout_rate, weight_constraint,neurons,activation,num_classes,X_Corr):
  # create modeloptimizer=optimizer
  input_shape = (X_Corr.shape[1],)
  inputs = tf.keras.Input(shape=input_shape)
  input = tf.keras.layers.Dense((28)+neurons, activation=activation)(inputs)
  x= tf.keras.layers.Dropout(dropout_rate)(input)
  second=tf.keras.layers.Dense((12)+neurons, activation=activation)(x)
  x= tf.keras.layers.Dropout(dropout_rate)(second)
  third=tf.keras.layers.Dense((4)+neurons, activation=activation)(x)
  x= tf.keras.layers.Dropout(dropout_rate)(third)
  fourth=tf.keras.layers.Dense((-4)+neurons, activation=activation)(x)
  x= tf.keras.layers.Dropout(dropout_rate)(fourth)
  fifth=tf.keras.layers.Dense((-12)+neurons, activation=activation)(x)
# Add more hidden layers if necessary

# Add output layer with softmax activation
  outputs = tf.keras.layers.Dense(num_classes, activation='sigmoid')(fifth)

# Create the model
  model_optim = tf.keras.Model(inputs=inputs, outputs=outputs)
  model_optim.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
  return  model_optim

from sklearn.model_selection import StratifiedKFold

model = KerasClassifier(model_optim=create_model, epochs=epochs, batch_size=batch_size, verbose=0)
dropout_rate = [0.3, 0.5]
weight_constraint = [3.0, 5.0]
neurons = [20]
optimizer = ['SGD', 'Adam', 'RMSProp']
activation = ['relu']
param_grid = dict(model__dropout_rate=dropout_rate, model__weight_constraint=weight_constraint,
                  model__neurons=neurons,model__activation=activation)

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv, scoring='accuracy')
grid_search

!pip install keras-tuner

import keras_tuner

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(Xtrain.shape[1],)))

    # Tune the number of hidden layers and units
    for i in range(hp.Int('num_hidden_layers', min_value=1, max_value=4)):
        model.add(keras.layers.Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=96, step=32),
                             activation=hp.Choice(f'activation_{i}', values=['relu', 'tanh'])))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # Tune the learning rate
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
        )

    return model

build_model(keras_tuner.HyperParameters())

tuner = keras_tuner.Hyperband(
  hypermodel=build_model,
  objective='val_accuracy',
  max_epochs=100,
  factor=3,
  directory='tuning_dir',
  project_name='samples')

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)

tuner.search(Xtrain, Ytrain, epochs=30 ,validation_data=(X_val, y_val), callbacks=[early_stopping])
#tuner.search(Xtrain, Ytrain, epochs=30 ,validation_data=(X_val, y_val))

tuner.results_summary()

best_model = tuner.get_best_models(num_models=2)[0]

best_model.summary()

"""Best MODEL'S ACCURACY AND AUC"""

_, accuracy = best_model.evaluate(Xtrain, Ytrain)
accuracy*100

# Evaluate the model on the test set
_, test_accuracy = best_model.evaluate(X_test, y_test)
test_accuracy*100

# Evaluate the AUC Score on the test set
from sklearn.metrics import roc_auc_score

test_preds = best_model.predict(X_test)
test_preds_binary = (test_preds > 0.5).astype(int)
auc = roc_auc_score(y_test, test_preds_binary)

print(f"Final AUC Score: {auc:.2f}")

#Saving the model
# Save the model to a file
import pickle
pickle_out=open("best_model.pkl","wb")
pickle.dump(best_model, pickle_out)
pickle_out.close()

