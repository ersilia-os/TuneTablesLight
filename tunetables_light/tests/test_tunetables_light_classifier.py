import unittest

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.datasets import fetch_covtype
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from tunetables_light.scripts.transformer_prediction_interface import (
    TuneTablesClassifierLight,
)


def make_random_dataset():
    X, y = make_classification(
        n_samples=100_000,
        n_features=100,
        n_informative=20,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        weights=[0.5, 0.5],  # ~99 800 negatives, ~200 positives
        flip_y=0,
        class_sep=1.0,
        random_state=42,
    )
    return train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42,
    )


def make_forest_cov_dataset(slice_size):
    data = fetch_covtype(download_if_missing=True)
    X_full, y_full = data.data, data.target
    X, y = X_full[:slice_size], y_full[:slice_size]
    return train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)


def make_breast_cancer_dataset():
    data = load_breast_cancer()
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


class TestTuneTablesClassifierFit(unittest.TestCase):
    def test_main(self):
        X_train, X_test, y_train, y_test = make_breast_cancer_dataset()
        print("X_train:", X_train.shape, "y_train:", y_train.shape)
        print("X_test: ", X_test.shape, "y_test: ", y_test.shape)

        model = TuneTablesClassifierLight(epoch=3, device="cuda", dropout=0.2)
        model.fit(X_train, y_train)
        model.save_model("my_model")
        y_hat = model.predict_proba(X_test)
        n_classes = 2
        y_pred = np.argmax(y_hat, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        if n_classes == 2:
            positive_class_ix = 1
            auc = roc_auc_score(y_test, y_hat[:, positive_class_ix])
            print(f"ROC-AUC (binary): {auc:.4f}")
        else:
            auc = roc_auc_score(y_test, y_hat, multi_class="ovr", average="macro")
            print(f"ROC-AUC (multiclass, macro): {auc:.4f}")


class TestTuneTablesClassifierLoadWithClassImbalance(unittest.TestCase):
    def test_main(self):
        _, X_test, y_train, y_test = make_breast_cancer_dataset()
        model = TuneTablesClassifierLight.load_model("my_model")
        y_hat = model.predict_proba(X_test)
        print(y_hat)

        y_pred = np.argmax(y_hat, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

        n_classes = y_hat.shape[1]
        if n_classes == 2:
            positive_class_ix = 1
            auc = roc_auc_score(y_test, y_hat[:, positive_class_ix])
            print(f"ROC-AUC (binary): {auc:.4f}")
        else:
            auc = roc_auc_score(y_test, y_hat, multi_class="ovr", average="macro")
            print(f"ROC-AUC (multiclass, macro): {auc:.4f}")


class TestTuneTablesClassifierLoad(unittest.TestCase):
    def test_main(self):
        X_train, X_test, y_train, y_test = make_forest_cov_dataset(100_000)
        model = TuneTablesClassifierLight.load_model("my_model")
        y_hat = model.predict_proba(X_test)
        y_pred = np.argmax(y_hat, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
