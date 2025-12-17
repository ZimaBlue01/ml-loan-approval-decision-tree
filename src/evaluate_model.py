# src/evaluate_model.py

import numpy as np
from sklearn.model_selection import KFold, cross_val_score

from config import CV_FOLDS, CV_RANDOM_STATE
from data_preparation import load_dataset, prepare_features_and_target
from train_decision_tree import train_model


def evaluate_train_test_accuracy():
    """Compute training and testing accuracy for the trained Decision Tree model."""
    model, X_train, X_test, y_train, y_test = train_model()

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    print("Training Accuracy:", train_accuracy)
    print("Testing Accuracy:", test_accuracy)


def cross_validate_accuracy():
    """Compute cross-validated accuracy (5-fold) for the Decision Tree model."""
    df = load_dataset()
    X, y = prepare_features_and_target(df)

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)

    # create a fresh model each fold via cross_val_score
    from sklearn.tree import DecisionTreeClassifier
    from config import TREE_RANDOM_STATE

    clf = DecisionTreeClassifier(random_state=TREE_RANDOM_STATE)
    cv_scores = cross_val_score(clf, X, y, cv=kf, scoring="accuracy", n_jobs=-1)

    print("Fold accuracies:", cv_scores)
    print("Mean CV accuracy:", np.mean(cv_scores))
    print("Std CV accuracy:", np.std(cv_scores))


if __name__ == "__main__":
    evaluate_train_test_accuracy()
    print("\n--- Cross Validation ---")
    cross_validate_accuracy()
