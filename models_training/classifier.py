from abc import ABC, abstractmethod


class Classifier(ABC):
    def __init__(self, training_data, test_data):
        self.training_data = training_data
        self.test_data = test_data

    @abstractmethod
    def train_and_evaluate(self):
        pass
