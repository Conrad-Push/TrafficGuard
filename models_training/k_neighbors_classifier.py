from models_training.classifier import Classifier
from sklearn.neighbors import KNeighborsClassifier

class KNeighborsClassifierModel(Classifier):
    def train_and_evaluate(self):
        X_train = self.training_data.drop('Class', axis=1)
        y_train = self.training_data['Class']

        self.model = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='auto', leaf_size=9, p=1)
        self.model.fit(X_train, y_train)
