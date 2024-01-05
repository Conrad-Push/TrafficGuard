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
        self.__print_statistics(self.X_test, self.y_test)

    def print_disturbed_statistics(self):
        self.__print_statistics(self.X_disturbed_test, self.y_disturbed_test)

    def __print_statistics(self, x_data, y_data):
        predictions = self.model.predict(x_data)
        accuracy = accuracy_score(y_data, predictions)
        confusion = confusion_matrix(y_data, predictions)
        report = classification_report(y_data, predictions)

        print(f"Classifier: {self.name}")
        print(f"Accuracy: {accuracy}")
        print(f"Confusion Matrix:\n{confusion}")
        print(f"Classification Report:\n{report}")

        self.__test_and_print_time_of_prediction()

        print("\n")

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
