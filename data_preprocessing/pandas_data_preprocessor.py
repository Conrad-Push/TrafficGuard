import logging

from data_preprocessing.data_preprocessor import DataPreprocessor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np


class PandasDataPreprocessor(DataPreprocessor):
    def __init__(self, data_type='general'):
        super().__init__()
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(data_type)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.data_type = data_type
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def change_column_names_to_pascal_case(self):
        self.data.columns = [self._to_pascal_case_if_needed(col) for col in self.data.columns]

        self.logger.info(
            "Column names have been changed to PascalCase where necessary, first letter always capitalized.",
            extra={'data_type': self.data_type})

    @staticmethod
    def _to_pascal_case_if_needed(s: str):
        parts = s.split('_')
        if len(parts) > 1:
            return ''.join(word.capitalize() for word in parts)
        else:
            return s[0].upper() + s[1:]

    def choose_columns(self, columns: list[str]):
        self.data = self.data[columns]

    def save_data(self, folder_path):
        self.training_data.to_csv(folder_path + '/training.csv', index=False)
        self.test_data.to_csv(folder_path + '/test.csv', index=False)

        self.logger.info("Data saved successfully.", extra={'data_type': self.data_type})

    def transform_column_data_to_logarithmic_scale(self, column: str):
        self.data[column] = np.log(self.data[column] + 1)

    def scaling_column_data_numerical_attributes(self, columns: list[str]):
        self.data[columns] = self.scaler.fit_transform(self.data[columns])

    def encoding_column_data_categorical_attributes(self, column: str):
        self.data[column] = self.label_encoder.fit_transform(self.data[column])

    def split_data_to_training_and_test(self, test_size=0.2):
        self.data = self.data.sample(frac=1, random_state=42)
        split_index = int(len(self.data) * test_size)
        self.test_data = self.data.iloc[:split_index]
        self.training_data = self.data.iloc[split_index:]

        self.logger.info("Data split successfully.", extra={'data_type': self.data_type})

    def load_data(self, file_path):
        try:
            self.data = pd.read_csv(file_path)
            self.logger.info("Data loaded successfully.", extra={'data_type': self.data_type})
            return self.data
        except Exception as e:
            self.logger.error(f"Error occurred while loading data: {str(e)}", extra={'data_type': self.data_type})
            return None

    def check_missing_values(self):
        if self.data is not None:
            missing_values = self.data.isnull().sum()
            missing_values = missing_values[missing_values > 0]
            if len(missing_values) > 0:
                self.logger.warning("Missing values found:", extra={'data_type': self.data_type})
                self.logger.warning("\n" + str(missing_values), extra={'data_type': self.data_type})
            else:
                self.logger.info("No missing values found.", extra={'data_type': self.data_type})
            return missing_values
        else:
            self.logger.error("Data is not loaded!", extra={'data_type': self.data_type})
            return None

    def show_basic_statistics(self):
        if self.data is not None:
            stats = self.data.describe()
            for stat in stats:
                self.logger.info(f"Describe {stat}:\n{stats[stat]}", extra={'data_type': self.data_type})
        else:
            self.logger.error("Data is not loaded!", extra={'data_type': self.data_type})

    def plot_histogram(self, column):
        if self.data is not None:
            import seaborn as sns
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            sns.histplot(self.data[column], bins=30, kde=False)
            plt.title(f'Histogram of {column}')
            plt.show()
        else:
            self.logger.error("Data is not loaded!", extra={'data_type': self.data_type})
