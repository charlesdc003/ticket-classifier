import json
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)


def evaluate(model, X_test: list, y_test: list, model_name: str = "model") -> dict:
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n{model_name} Results:")
    print(f"Accuracy:  {round(accuracy * 100)}%")
    print(f"F1:        {round(f1, 3)}")
    print(f"Precision: {round(precision, 3)}")
    print(f"Recall:    {round(recall, 3)}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {
        "model": model_name,
        "accuracy": round(accuracy, 3),
        "f1": round(f1, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "predictions": y_pred.tolist()
    }