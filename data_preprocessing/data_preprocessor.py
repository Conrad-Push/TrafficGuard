import logging
from abc import ABC, abstractmethod


class DataPreprocessor(ABC):
    def __init__(self):
        self.data = None

    @abstractmethod
    def load_data(self, file_path):
        pass

    @abstractmethod
    def check_missing_values(self):
        pass
