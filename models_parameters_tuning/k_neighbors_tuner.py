from tuner import Tuner
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score


class KNeighborsModelTuner(Tuner):
    def model_parameters_tuning(self):
        X_train = self.training_data.drop('Class', axis=1)
        y_train = self.training_data['Class']

        def custom_score(y_true, y_pred):
            score = accuracy_score(y_true, y_pred)
            print(f"Score: {score}")
            return score

        scorer = make_scorer(custom_score, greater_is_better=True)

        # Define the parameter values that should be searched
        k_range = list(range(1, 20))
        weight_options = ['uniform', 'distance']
        algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
        leaf_size = list(range(1, 30))
        p=[1,2]

        param_grid = dict(n_neighbors=k_range, weights=weight_options, algorithm=algorithms, leaf_size=leaf_size, p=p)

        model = KNeighborsClassifier()
        grid = GridSearchCV(model, param_grid, cv=10, scoring=scorer)
        grid.fit(X_train, y_train)

        print(grid.best_params_)
        print(grid.best_score_)

        return grid