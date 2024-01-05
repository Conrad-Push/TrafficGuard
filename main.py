import os
import logging
import pandas as pd

from data_preprocessing.main import main as data_preprocessing_main
from models_training.main import main as model_training_main

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(data_type)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(filepath):
    filename_with_extension = os.path.basename(filepath)
    filename_without_extension = os.path.splitext(filename_with_extension)[0]

    try:
        data = pd.read_csv(filepath)
        logger.info("Data loaded successfully.\n", extra={'data_type': filename_without_extension})
        return data
    except Exception as e:
        logger.error(f"Error occurred while loading data: {str(e)}\n", extra={'data_type': filename_without_extension})
        return None


def main():
    filepath = './data/IDS_Data1.csv'

    if os.path.exists('./data/test.csv') and os.path.exists('./data/training.csv') and os.path.exists('./data/disturbed_test.csv'):
        training_data = load_data('./data/training.csv')
        test_data = load_data('./data/test.csv')
        disturbed_test_data = load_data('./data/disturbed_test.csv')
    else:
        training_data, test_data, disturbed_test_data = data_preprocessing_main(filepath)

    model_training_main(training_data, test_data, disturbed_test_data)


if __name__ == '__main__':
    main()
