from models_training.naive_bayes_classifier import NaiveBayesClassifier
from models_training.decision_tree_classifier import DecisionTreeClassifierModel
from models_training.k_neighbors_classifier import KNeighborsClassifierModel


def main(training_data, test_data, disturbed_test_data):
    for classifier in [NaiveBayesClassifier, DecisionTreeClassifierModel, KNeighborsClassifierModel]:
        model = classifier(training_data, test_data, disturbed_test_data)
        model.train_and_evaluate()
        model.print_normal_statistics()
        model.print_disturbed_statistics()
