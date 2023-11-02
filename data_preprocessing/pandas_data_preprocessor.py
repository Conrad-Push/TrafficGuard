import logging

from data_preprocessing.data_preprocessor import DataPreprocessor
import pandas as pd


class PandasDataPreprocessor(DataPreprocessor):
    def __init__(self, data_type='general'):
        super().__init__()
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(data_type)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.data_type = data_type

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
            self.logger.info("Basic statistics:\n" + str(self.data.describe()), extra={'data_type': self.data_type})
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
