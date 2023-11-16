from data_preprocessing.pandas_data_preprocessor import PandasDataPreprocessor


def main():
    columns_to_filter = ["TotPkts", "TotBytes", "SrcBytes", "flag", "service", "count", "dst_bytes", "class"]
    columns_to_transform_to_log = ["TotPkts", "SrcBytes", "dst_bytes"]
    columns_to_encode = ["flag", "service", "class"]

    data_processor = PandasDataPreprocessor(data_type='IDS_Data1')

    data_processor.load_data('../data/IDS_Data1.csv')
    data_processor.check_missing_values()
    data_processor.show_basic_statistics()

    data_processor.choose_columns(columns_to_filter)
    for column in columns_to_filter:
        data_processor.plot_histogram(column)

    for column in columns_to_transform_to_log:
        data_processor.transform_column_data_to_logarithmic_scale(column)
        data_processor.plot_histogram(column)

    for column in columns_to_encode:
        data_processor.encoding_column_data_categorical_attributes(column)
        data_processor.plot_histogram(column)

    data_processor.change_column_names_to_pascal_case()
    data_processor.split_data_to_training_and_test()
    data_processor.save_data('../data')


if __name__ == '__main__':
    main()
