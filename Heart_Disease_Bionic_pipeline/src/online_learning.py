from sklearn.linear_model import SGDClassifier
import numpy as np

class ContinuousLearner:
    def __init__(self):
        self.model = SGDClassifier(loss='log_loss', random_state=42)
        self.classes = np.array([0, 1])
        self.is_initialized = False

    def incremental_train(self, X_batch, y_batch):
        """
        Simulates receiving a new batch of patient sensor data from the edge
        and updating the model weights incrementally without retraining from scratch.
        """
        self.model.partial_fit(X_batch, y_batch, classes=self.classes)
        self.is_initialized = True

    def evaluate(self, X_test, y_test):
        if not self.is_initialized:
            return
            
        predictions = self.model.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        print(f"\n--- Continuous Learner (SGD) Status ---")
        print(f"Incremental Update Complete. New Accuracy: {accuracy * 100:.2f}%")
