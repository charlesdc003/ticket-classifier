import sys
sys.path.insert(0, ".")

from src.classifier.data import load_tickets, prepare_features, train_test_split_data
from src.classifier.baseline import build_tfidf_lr_pipeline, train
from src.classifier.evaluate import evaluate


def test_load_tickets():
    df = load_tickets("data/tickets.jsonl")
    assert len(df) == 100
    assert "subject" in df.columns
    assert "expected_category" in df.columns


def test_prepare_features():
    df = load_tickets("data/tickets.jsonl")
    X, y = prepare_features(df)
    assert len(X) == 100
    assert len(y) == 100
    assert all(isinstance(text, str) for text in X)


def test_train_test_split():
    df = load_tickets("data/tickets.jsonl")
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_data(df)
    assert len(X_train) + len(X_val) + len(X_test) == 100
    assert len(X_train) > len(X_test)


def test_baseline_trains_and_predicts():
    df = load_tickets("data/tickets.jsonl")
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_data(df)
    model = train(build_tfidf_lr_pipeline(), X_train, y_train)
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)


def test_baseline_accuracy_above_threshold():
    df = load_tickets("data/tickets.jsonl")
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_data(df)
    model = train(build_tfidf_lr_pipeline(), X_train, y_train)
    results = evaluate(model, X_test, y_test, "test")
    assert results["accuracy"] >= 0.6