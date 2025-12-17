# src/loan_automation.py

import pandas as pd
from config import FEATURE_COLUMNS
from train_decision_tree import train_model


def predict_approval(model, age: int, income: float, credit: float, years: float) -> str:
    """Predict loan approval for a single applicant."""
    candidate_x = pd.DataFrame(
        {
            "Age": [age],
            "Income_Level": [income],
            "Credit_Score": [credit],
            "Years_Employed": [years],
        }
    )

    prediction = model.predict(candidate_x)[0]
    return "Yes, candidate qualifies." if prediction == 1 else "No, unfortunately candidate does not qualify."


if __name__ == "__main__":
    model, *_ = train_model()

    # Example candidates
    print(predict_approval(model, 28, 170, 660, 2))
    print(predict_approval(model, 32, 200, 720, 4))
    print(predict_approval(model, 50, 240, 680, 5))
    print(predict_approval(model, 45, 300, 740, 9))
    print(predict_approval(model, 60, 220, 650, 6))
