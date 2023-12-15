from models_training.classifier import Classifier
from sklearn.naive_bayes import GaussianNB


class NaiveBayesClassifier(Classifier):
    def train_and_evaluate(self):
        X_train = self.training_data.drop('Class', axis=1)
        y_train = self.training_data['Class']

        self.model = GaussianNB(priors=[0.466, 0.534], var_smoothing=1e-9)
        self.model.fit(X_train, y_train)
