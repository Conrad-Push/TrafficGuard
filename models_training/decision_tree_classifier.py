from models_training.classifier import Classifier
from sklearn.tree import DecisionTreeClassifier


class DecisionTreeClassifierModel(Classifier):
    def __init__(self, training_data, test_data):
        super().__init__(training_data, test_data, "Decision Tree Classifier")

    def train_and_evaluate(self):
        self.model = DecisionTreeClassifier(criterion= 'gini', max_depth= 10, max_leaf_nodes= 50, min_impurity_decrease= 0, min_samples_leaf= 2, min_samples_split= 2)
        self.model.fit(self.X_train, self.y_train)
