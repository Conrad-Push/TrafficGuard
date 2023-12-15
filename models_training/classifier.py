from abc import ABC, abstractmethod

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class Classifier(ABC):
    def __init__(self, training_data, test_data):
        self.X_train = training_data.drop('Class', axis=1)
        self.y_train = training_data['Class']
        self.X_test = test_data.drop('Class', axis=1)
        self.y_test = test_data['Class']

        self.model = None

    @abstractmethod
    def train_and_evaluate(self):
        pass

    def print_statistics(self):
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        confusion = confusion_matrix(self.y_test, predictions)
        report = classification_report(self.y_test, predictions)

        print(f"Accuracy: {accuracy}")
        print(f"Confusion:\n{confusion}")
        print(f"Report:\n{report}")
