from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class BaselineModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions)
        
        print(f"--- Model Evaluation ---")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"\nConfusion Matrix:\n{conf_matrix}")
        print(f"(Top-Left: True Negatives, Top-Right: False Positives)")
        print(f"(Bottom-Left: False Negatives, Bottom-Right: True Positives)")
        print(f"\nClassification Report (Focus on Target 1 Recall):\n{report}")
        
        return self.model
