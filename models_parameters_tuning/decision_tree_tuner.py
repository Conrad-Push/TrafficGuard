from tuner import Tuner
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score


class DecisionTreeModelTuner(Tuner):
    def model_parameters_tuning(self):
        X_train = self.training_data.drop('Class', axis=1)
        y_train = self.training_data['Class']

        def custom_score(y_true, y_pred):
            score = accuracy_score(y_true, y_pred)
            print(f"Score: {score}")
            return score

        scorer = make_scorer(custom_score, greater_is_better=True)

        # Define the parameter values that should be searched
        criterion = ['gini', 'entropy']
        max_depth = [None, 10, 20, 30, 40, 50, 60, 70]
        min_samples_split = [2, 5, 10, 15, 20]
        min_samples_leaf = [1, 2, 4, 6, 8]
        max_leaf_nodes = [None, 10, 20, 30, 40, 50]
        min_impurity_decrease = [0, 0.001, 0.01, 0.1]

        param_grid = dict(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease)

        model = DecisionTreeClassifier()
        grid = GridSearchCV(model, param_grid, cv=10, scoring=scorer)
        grid.fit(X_train, y_train)

        print(grid.best_params_)
        print(grid.best_score_)

        return grid