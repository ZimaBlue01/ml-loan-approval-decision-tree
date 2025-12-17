# src/train_decision_tree.py

from sklearn.tree import DecisionTreeClassifier

from config import TREE_RANDOM_STATE
from data_preparation import load_dataset, prepare_features_and_target, split_data


def train_model():
    """Train a Decision Tree classifier and return trained model + split data."""
    df = load_dataset()
    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = DecisionTreeClassifier(random_state=TREE_RANDOM_STATE)
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    model, X_train, X_test, y_train, y_test = train_model()
    print("Model trained successfully.")
