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

    # TODO: Column on discord channel
    @abstractmethod
    def choose_columns(self, columns: list[str]):
        pass

    # TODO: Check to high data, like TotBytes 90736
    @abstractmethod
    def filter_data(self):
        pass

    @abstractmethod
    def filter_by_max_value(self, column: str, max_value: int):
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
