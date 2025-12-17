# src/visualize_tree.py

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from config import FEATURE_COLUMNS
from train_decision_tree import train_model


def plot_decision_tree(save_path: str | None = None):
    """Plot the trained Decision Tree rules and optionally save as an image."""
    model, X_train, X_test, y_train, y_test = train_model()

    plt.figure(figsize=(12, 8))
    plot_tree(
        model,
        feature_names=FEATURE_COLUMNS,
        class_names=["No", "Yes"],
        filled=True,
        rounded=True
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved decision tree image to: {save_path}")

    plt.show()


if __name__ == "__main__":
    # Save into assets folder if you want:
    plot_decision_tree(save_path="assets/charts/decision_tree.png")
