from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


@dataclass
class TfidfLRBaseline:
    vectorizer: TfidfVectorizer
    clf: OneVsRestClassifier
    label_fields: List[str]

    def predict_proba_matrix(self, texts: Sequence[str]) -> np.ndarray:
        X = self.vectorizer.transform(list(texts))
        probs = self.clf.predict_proba(X)
        # sklearn returns shape (n_samples, n_labels)
        return probs.astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "clf": self.clf, "label_fields": self.label_fields}, path)

    @staticmethod
    def load(path: Path) -> "TfidfLRBaseline":
        obj = joblib.load(path)
        return TfidfLRBaseline(vectorizer=obj["vectorizer"], clf=obj["clf"], label_fields=obj["label_fields"])


def train_tfidf_lr(
    train_texts: Sequence[str],
    train_labels: np.ndarray,
    label_fields: Sequence[str],
    tfidf_max_features: int = 80000,
    tfidf_ngram_range: Tuple[int, int] = (1, 2),
    lr_C: float = 4.0,
) -> TfidfLRBaseline:
    vectorizer = TfidfVectorizer(
        max_features=tfidf_max_features,
        ngram_range=tfidf_ngram_range,
        lowercase=True,
    )
    X = vectorizer.fit_transform(list(train_texts))

    base_lr = LogisticRegression(
        C=lr_C,
        max_iter=200,
        solver="liblinear",
    )
    clf = OneVsRestClassifier(base_lr)
    clf.fit(X, train_labels)

    return TfidfLRBaseline(vectorizer=vectorizer, clf=clf, label_fields=list(label_fields))
