from abc import ABC, abstractmethod


class Tuner(ABC):
    def __init__(self, training_data):
        self.training_data = training_data

    @abstractmethod
    def model_parameters_tuning(self):
        pass
