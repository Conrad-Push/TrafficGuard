from models_training.classifier import Classifier
from sklearn.neighbors import KNeighborsClassifier


class KNeighborsClassifierModel(Classifier):
    def __init__(self, training_data, test_data, disturbed_test_data):
        super().__init__(training_data, test_data, disturbed_test_data, "K Neighbors Classifier")

    def train_and_evaluate(self):
        self.model = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='auto', leaf_size=9, p=1)
        self.model.fit(self.X_train, self.y_train)
