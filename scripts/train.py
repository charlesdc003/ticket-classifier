import sys
import json
sys.path.insert(0, ".")

from src.classifier.data import load_tickets, train_test_split_data
from src.classifier.baseline import build_tfidf_lr_pipeline, build_tfidf_nb_pipeline, train
from src.classifier.evaluate import evaluate


def main():
    print("Loading data...")
    df = load_tickets("data/tickets.jsonl")
    print(f"Loaded {len(df)} tickets")

    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_data(df)
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print("\nTraining TF-IDF + Logistic Regression baseline...")
    lr_model = train(build_tfidf_lr_pipeline(), X_train, y_train)
    lr_results = evaluate(lr_model, X_test, y_test, "TF-IDF + Logistic Regression")

    print("\nTraining TF-IDF + Naive Bayes baseline...")
    nb_model = train(build_tfidf_nb_pipeline(), X_train, y_train)
    nb_results = evaluate(nb_model, X_test, y_test, "TF-IDF + Naive Bayes")

    results = {
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "models": [lr_results, nb_results]
    }

    with open("data/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to data/results.json")


if __name__ == "__main__":
    main()