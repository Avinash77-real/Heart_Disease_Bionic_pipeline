from src.data_processor import load_and_preprocess_data
from src.model_engine import BaselineModel
from src.online_learning import ContinuousLearner

def run_pipeline():
    print("Initializing Data Pipeline...")
    filepath = 'data/heart.csv'
    X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess_data(filepath)
    
    print("\nTraining Baseline Logistic Regression...")
    baseline = BaselineModel()
    baseline.train(X_train, y_train)
    baseline.evaluate(X_test, y_test)
    
    print("\nSimulating Data Stream for Continuous Learning...")
    online_learner = ContinuousLearner()
    
    half_index = len(X_train) // 2
    X_batch_1, y_batch_1 = X_train[:half_index], y_train.iloc[:half_index]
    X_batch_2, y_batch_2 = X_train[half_index:], y_train.iloc[half_index:]
    
    print("Processing Batch 1...")
    online_learner.incremental_train(X_batch_1, y_batch_1)
    online_learner.evaluate(X_test, y_test)
    
    print("Processing Batch 2 (Updating weights)...")
    online_learner.incremental_train(X_batch_2, y_batch_2)
    online_learner.evaluate(X_test, y_test)

if __name__ == "__main__":
    run_pipeline()
