from abc import ABC, abstractmethod

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time


class Classifier(ABC):
    def __init__(self, training_data, test_data, disturbed_test_data, name):
        self.name = name
        self.X_train = training_data.drop('Class', axis=1)
        self.y_train = training_data['Class']
        self.X_test = test_data.drop('Class', axis=1)
        self.y_test = test_data['Class']
        self.X_disturbed_test = disturbed_test_data.drop('Class', axis=1)
        self.y_disturbed_test = disturbed_test_data['Class']

        self.model = None

    @abstractmethod
    def train_and_evaluate(self):
        pass

    def print_normal_statistics(self):
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        confusion = confusion_matrix(self.y_test, predictions)
        report = classification_report(self.y_test, predictions)

        self.print_statistics(accuracy, confusion, report)

        self.__test_and_print_time_of_prediction()

        print("\n")

    def print_disturbed_statistics(self):
        predictions = self.model.predict(self.X_disturbed_test)
        accuracy = accuracy_score(self.y_disturbed_test, predictions)
        confusion = confusion_matrix(self.y_disturbed_test, predictions)
        report = classification_report(self.y_disturbed_test, predictions)

        print("Disturbed Test Data: ")

        self.print_statistics(accuracy, confusion, report)

        self.__test_and_print_time_of_prediction()

        print("\n")

    def print_statistics(self, accuracy, confusion, report):
        print(f"Classifier: {self.name}")
        print(f"Accuracy: {accuracy}")
        print(f"Confusion Matrix:\n{confusion}")
        print(f"Classification Report:\n{report}")

    def __test_and_print_time_of_prediction(self):
        sizes = [1.0, 0.5, 0.1]

        for size in sizes:
            start_time = time.time()
            sample_X_test = self.__get_sample(self.X_test, size)
            self.model.predict(sample_X_test)
            elapsed_time = time.time() - start_time
            print(f"Time for size {size}: {elapsed_time:.5f} seconds")

    @staticmethod
    def __get_sample(data, size):
        sample_size = int(len(data) * size)
        return data[:sample_size]
