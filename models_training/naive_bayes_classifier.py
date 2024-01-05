from models_training.classifier import Classifier
from sklearn.naive_bayes import GaussianNB


class NaiveBayesClassifier(Classifier):
    def __init__(self, training_data, test_data, disturbed_test_data):
        super().__init__(training_data, test_data, disturbed_test_data, "Naive Bayes Classifier")

    def train_and_evaluate(self):
        self.model = GaussianNB(priors=[0.466, 0.534], var_smoothing=1e-9)
        self.model.fit(self.X_train, self.y_train)
