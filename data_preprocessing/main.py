from data_preprocessing.pandas_data_preprocessor import PandasDataPreprocessor


def main():
    columns = ["TotPkts", "TotBytes", "SrcBytes", "flag", "service", "count", "dst_bytes", "class"]

    data_processor = PandasDataPreprocessor(data_type='IDS_Data1')
    data_processor.load_data('../data/IDS_Data1.csv')
    data_processor.check_missing_values()
    data_processor.show_basic_statistics()
    data_processor.choose_columns(columns)
    for single_column in columns:
        data_processor.plot_histogram(single_column)

    data_processor.transform_column_data_to_logarithmic_scale('TotBytes')
    data_processor.plot_histogram('TotBytes')


if __name__ == '__main__':
    main()
