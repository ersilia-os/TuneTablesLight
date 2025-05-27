import unittest

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from tunetables.scripts.transformer_prediction_interface import TuneTablesClassifier
from tunetables.scripts.transformer_prediction_interface import TabPFNClassifier


# class TestTuneTablesClassifierSingle(unittest.TestCase):
#     def test_main(self):
#         X, y = load_breast_cancer(return_X_y=True)
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#         clf = TuneTablesClassifier()
#         clf.fit(X_train,y_train)
#         y_eval = clf.predict(X_test)
#         accuracy = accuracy_score(y_test, y_eval)
#         print("Accuracy:", np.round(accuracy, 2))


class TestTuneTablesClassifierMulticlass(unittest.TestCase):
    def test_main(self):
        X, y = make_classification(
            n_samples=100000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            random_state=42,
        )

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42
        )
        clf = TuneTablesClassifier()
        clf.fit(X_train, y_train)
        y_eval = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_eval)
        print("Accuracy:", np.round(accuracy, 2))



# class TestTuneTablesClassifierInferenceOnly(unittest.TestCase):
#     def test_main(self):
#         X, y = make_classification(
#             n_samples=100000,
#             n_features=20,
#             n_informative=15,
#             n_redundant=5,
#             n_classes=2,
#             random_state=42,
#         )

#         # Split into train and test
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.15, random_state=42
#         )
#         clf = TuneTablesClassifier(device="cuda", only_inference=True, model_file="logs/prior_diff_real_checkpoint_multiclass_05_27_2025_15_23_16_n_0_epoch_0.cpkt")
#         # clf = TabPFNClassifier(device="cuda", only_inference=True)
#         clf.fit(X_train, y_train)

#         predictions = (
#             clf.predict_proba(X_test)
#         )
#         predictions = np.argmax(predictions, axis=1)
#         accuracy = accuracy_score(y_test, predictions)
#         print(f"Accuracy: {accuracy}")
