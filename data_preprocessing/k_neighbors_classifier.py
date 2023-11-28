from classifier import Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class KNeighborsClassifierModel(Classifier):
    def train_and_evaluate(self):
        X_train = self.training_data.drop('Class', axis=1)
        y_train = self.training_data['Class']
        X_test = self.test_data.drop('Class', axis=1)
        y_test = self.test_data['Class']

        model = KNeighborsClassifier()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        confusion = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions)
        
        print(f"Accuracy: {accuracy}")
        print(f"Confusion:\n{confusion}")
        print(f"Report:\n{report}")
