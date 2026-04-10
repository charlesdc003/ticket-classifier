import json
import pandas as pd
from pathlib import Path


CATEGORY_LABELS = [
    "billing",
    "auth",
    "feature_request",
    "bug",
    "general"
]


def load_tickets(path: str) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def prepare_features(df: pd.DataFrame) -> tuple:
    df["text"] = df["subject"] + " " + df["message"]
    X = df["text"].tolist()
    y = df["expected_category"].tolist()
    return X, y


def train_test_split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1):
    from sklearn.model_selection import train_test_split

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test