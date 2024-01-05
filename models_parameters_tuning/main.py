from decision_tree_tuner import DecisionTreeModelTuner
from k_neighbors_tuner import KNeighborsModelTuner
import pandas as pd

def main():
    train_data = pd.read_csv('../data/training.csv')

    # Decision Tree Classifier
    print("\nDecision Tree Tuner:\n")
    dt_tuner = DecisionTreeModelTuner(train_data)
    dt_grid = dt_tuner.model_parameters_tuning()

    # K Neighbors Classifier
    print("\nK Neighbors Tuner:\n")
    kn_tuner = KNeighborsModelTuner(train_data)
    kn_grid = kn_tuner.model_parameters_tuning()

    print("\nDecision Tree Tuner - best params and accuracy:\n")
    print(dt_grid.best_params_)
    print(dt_grid.best_score_)

    print("\nK Neighbors Tuner - best params and accuracy:\n")
    print(kn_grid.best_params_)
    print(kn_grid.best_score_)

if __name__ == '__main__':
    main()