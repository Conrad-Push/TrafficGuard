from abc import ABC, abstractmethod

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class Classifier(ABC):
    def __init__(self, training_data, test_data):
        self.training_data = training_data
        self.test_data = test_data
        self.model = None

    @abstractmethod
    def train_and_evaluate(self):
        pass

    def print_statistics(self):
        X_test = self.test_data.drop('Class', axis=1)
        y_test = self.test_data['Class']

        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        confusion = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions)

        print(f"Accuracy: {accuracy}")
        print(f"Confusion:\n{confusion}")
        print(f"Report:\n{report}")
