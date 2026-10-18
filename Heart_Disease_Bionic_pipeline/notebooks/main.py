import pandas as pd
from src.data_processor import process_heart_data
from src.model_engine import train_logistic_regression 

df = pd.read_csv('data/heart.csv')

X_train, X_test, y_train, y_test, scaler, features = process_heart_data(df)

print("--- Heart Disease Research Pipeline ---")
