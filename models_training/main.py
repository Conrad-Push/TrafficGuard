from models_training.naive_bayes_classifier import NaiveBayesClassifier
from models_training.decision_tree_classifier import DecisionTreeClassifierModel
from models_training.k_neighbors_classifier import KNeighborsClassifierModel


def main(training_data, test_data):
    # Naive Bayes Classifier
    nb_classifier = NaiveBayesClassifier(training_data, test_data)
    nb_classifier.train_and_evaluate()

    # Decision Tree Classifier
    dt_classifier = DecisionTreeClassifierModel(training_data, test_data)
    dt_classifier.train_and_evaluate()

    # K Neighbors Classifier
    kn_classifier = KNeighborsClassifierModel(training_data, test_data)
    kn_classifier.train_and_evaluate()
