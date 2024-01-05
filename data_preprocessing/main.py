import os

from data_preprocessing.pandas_data_preprocessor import PandasDataPreprocessor


def main(filepath):
    filename_with_extension = os.path.basename(filepath)
    filename_without_extension = os.path.splitext(filename_with_extension)[0]

    columns_to_filter = ["TotPkts", "TotBytes", "SrcBytes", "flag", "service", "count", "dst_bytes", "class"]
    columns_to_transform_to_log = ["TotPkts", "TotBytes", "SrcBytes", "dst_bytes"]
    columns_to_scaling = ["TotPkts", "TotBytes", "SrcBytes", "dst_bytes"]
    columns_to_encode = ["flag", "service", "class"]

    columns_to_disturb = ['TotPkts', 'TotBytes', 'SrcBytes', 'DstBytes']
    distribution_rate = 0.1

    data_processor = PandasDataPreprocessor(data_type=filename_without_extension)

    data_processor.load_data(filepath)
    data_processor.check_missing_values()
    data_processor.show_basic_statistics()

    data_processor.choose_columns(columns_to_filter)
    for column in columns_to_filter:
        data_processor.plot_histogram(column)

    for column in columns_to_transform_to_log:
        data_processor.transform_column_data_to_logarithmic_scale(column)
        data_processor.plot_histogram(column)

    # Use this in case while testing the second approach to deal with the numerical data attributes
    # data_processor.scaling_column_data_numerical_attributes(columns_to_scaling)
    # for column in columns_to_scaling:
    #     data_processor.plot_histogram(column)

    for column in columns_to_encode:
        data_processor.encoding_column_data_categorical_attributes(column)
        data_processor.plot_histogram(column)

    data_processor.change_column_names_to_pascal_case()
    data_processor.split_data_to_training_and_test()
    data_processor.prepare_disturbed_test_data(columns_to_disturb, distribution_rate)
    data_processor.save_data('./data')

    return data_processor.training_data, data_processor.test_data, data_processor.disturbed_test_data
