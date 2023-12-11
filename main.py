from data_preprocessing.main import main as data_preprocessing_main
from models_training.main import main as model_training_main


def main():
    filepath = './data/IDS_Data1.csv'

    training_data, test_data = data_preprocessing_main(filepath)
    model_training_main(training_data, test_data)


if __name__ == '__main__':
    main()
