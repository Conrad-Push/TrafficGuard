from models_training.classifier import Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class DecisionTreeClassifierModel(Classifier):
    def train_and_evaluate(self):
        X_train = self.training_data.drop('Class', axis=1)
        y_train = self.training_data['Class']
        X_test = self.test_data.drop('Class', axis=1)
        y_test = self.test_data['Class']

        model = DecisionTreeClassifier(criterion= 'gini', max_depth= 10, max_leaf_nodes= 50, min_impurity_decrease= 0, min_samples_leaf= 2, min_samples_split= 2)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        confusion = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions)
        
        print(f"Accuracy: {accuracy}")
        print(f"Confusion:\n{confusion}")
        print(f"Report:\n{report}")
