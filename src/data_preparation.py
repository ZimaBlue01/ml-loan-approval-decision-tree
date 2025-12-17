# src/data_preparation.py

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TRAIN_TEST_SPLIT_TEST_SIZE,
    TRAIN_TEST_SPLIT_RANDOM_STATE,
)


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the loan approval dataset from CSV."""
    return pd.read_csv(path)


def prepare_features_and_target(df: pd.DataFrame):
    """
    Select required feature columns and encode target:
    Yes -> 1, No -> 0
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].replace({"Yes": 1, "No": 0}).copy()
    return X, y


def split_data(X, y):
    """Split into train/test sets with 85% train and 15% test."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TRAIN_TEST_SPLIT_TEST_SIZE,
        random_state=TRAIN_TEST_SPLIT_RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test
