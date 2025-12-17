# src/config.py

DATA_PATH = "datasets/Loan_Approval_data.csv"

FEATURE_COLUMNS = ["Age", "Income_Level", "Credit_Score", "Years_Employed"]
TARGET_COLUMN = "Loan_Status"

TRAIN_TEST_SPLIT_TEST_SIZE = 0.15
TRAIN_TEST_SPLIT_RANDOM_STATE = 1

TREE_RANDOM_STATE = 1

CV_FOLDS = 5
CV_RANDOM_STATE = 1
