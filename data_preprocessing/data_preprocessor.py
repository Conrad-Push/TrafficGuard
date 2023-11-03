import logging
from abc import ABC, abstractmethod


class DataPreprocessor(ABC):
    def __init__(self):
        self.data = None
        self.training_data = None
        self.test_data = None

    @abstractmethod
    def load_data(self, file_path):
        pass

    @abstractmethod
    def split_data_to_training_and_test(self, test_size=0.2):
        pass

    @abstractmethod
    def check_missing_values(self):
        pass

    @abstractmethod
    def choose_columns(self, columns: list[str]):
        pass

    @abstractmethod
    def transform_column_data_to_logarithmic_scale(self, column: str):
        pass

    @abstractmethod
    def show_basic_statistics(self):
        pass

    @abstractmethod
    def plot_histogram(self, column):
        pass

    @abstractmethod
    def save_data(self, folder_path: str):
        pass
