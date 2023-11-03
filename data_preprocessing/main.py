from data_preprocessing.pandas_data_preprocessor import PandasDataPreprocessor


def main():
    data_processor = PandasDataPreprocessor(data_type='training')

    data_processor.load_data('../data/IDS_Data1.csv')

    data_processor.check_missing_values()

    data_processor.show_basic_statistics()

    data_processor.plot_histogram('TotBytes')

    data_processor.plot_histogram('flag')


if __name__ == '__main__':
    main()
