from models_training.classifier import Classifier
from sklearn.naive_bayes import GaussianNB


class NaiveBayesClassifier(Classifier):
    def train_and_evaluate(self):
        self.model = GaussianNB(priors=[0.466, 0.534], var_smoothing=1e-9)
        self.model.fit(self.X_train, self.y_train)
