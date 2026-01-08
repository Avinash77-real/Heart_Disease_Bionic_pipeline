import pandas as pd
from src.data_processor import process_heart_data
from src.model_engine import train_logistic_regression 
# (Note: Use the LogisticRegression code from previous response for model_engine)

# Load the dataset you provided
df = pd.read_csv('data/heart.csv')

# Execute Pipeline
X_train, X_test, y_train, y_test, scaler, features = process_heart_data(df)

# Output Results for Research Portfolio
print("--- Heart Disease Research Pipeline ---")
# [Insert training and evaluation logic here]