from data_preprocessing.pandas_data_preprocessor import PandasDataPreprocessor


def main():
    train_preprocessor = PandasDataPreprocessor(data_type='training')
    test_preprocessor = PandasDataPreprocessor(data_type='test')

    train_preprocessor.load_data('../data/Train_data.csv')
    test_preprocessor.load_data('../data/Test_data.csv')

    train_preprocessor.check_missing_values()
    test_preprocessor.check_missing_values()


if __name__ == '__main__':
    main()
