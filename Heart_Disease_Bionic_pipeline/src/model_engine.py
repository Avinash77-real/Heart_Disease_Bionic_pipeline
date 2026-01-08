from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # Used for saving the model for Cloud deployment

class HeartDiseaseModel:
    def __init__(self):
        # Logistic Regression is used for high interpretability in clinical bionics
        self.model = LogisticRegression(max_iter=1000, solver='lbfgs')
        self.is_trained = False

    def train(self, X_train, y_train):
        """Trains the model on the provided medical features."""
        print("Training Heart Disease Model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self.model

    def evaluate(self, X_test, y_test):
        """Evaluates the model and prints research-grade metrics."""
        if not self.is_trained:
            raise Exception("Model must be trained before evaluation.")
            
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:\n", report)
        return accuracy, report

    def save_model(self, filepath='models/heart_model.pkl'):
        """Saves the model for use in Cloud-connected bionic systems."""
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")