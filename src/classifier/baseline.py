from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


def build_tfidf_lr_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])


def build_tfidf_nb_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf", MultinomialNB())
    ])


def train(pipeline: Pipeline, X_train: list, y_train: list) -> Pipeline:
    pipeline.fit(X_train, y_train)
    return pipeline