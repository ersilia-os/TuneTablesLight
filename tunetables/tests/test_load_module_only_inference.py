import unittest

from tunetables.datasets import load_openml_list, open_cc_dids
from tunetables.scripts.transformer_prediction_interface import TabPFNClassifier
from sklearn.metrics import accuracy_score

# class TestLoadModuleOnlyInference(unittest.TestCase):
#     def test_main(self):
#         test_datasets, cc_test_datasets_multiclass_df = load_openml_list(
#             open_cc_dids[:1],
#             multiclass=True,
#             shuffled=True,
#             filter_for_nan=False,
#             max_samples=10000,
#             num_feats=100,
#             return_capped=True,
#         )

#         classifier_with_only_inference = TabPFNClassifier(
#             device="cuda", only_inference=True
#         )
#         classifier_normal = TabPFNClassifier(device="cuda", only_inference=True)

#         for dataset in test_datasets:
#             xs, ys = dataset[1].clone(), dataset[2].clone()
#             eval_position = xs.shape[0] // 2
#             train_xs, train_ys = xs[0:eval_position], ys[0:eval_position]
#             test_xs, test_ys = xs[eval_position:], ys[eval_position:]

#             classifier_with_only_inference.fit(train_xs, train_ys)
#             classifier_normal.fit(train_xs, train_ys)

#             prediction_with_only_inference = (
#                 classifier_with_only_inference.predict_proba(test_xs)
#             )
#             prediction_normal = classifier_normal.predict_proba(test_xs)
#             print(f"Prediction 1 & 2: {prediction_with_only_inference} & {prediction_normal}")

#             self.assertTrue(
#                 prediction_normal.shape == prediction_with_only_inference.shape
#             )
#             number_of_predictions, number_of_classes = prediction_normal.shape

#             for number in range(number_of_predictions):
#                 for class_nr in range(number_of_classes):
#                     self.assertTrue(
#                         prediction_with_only_inference[number][class_nr]
#                         == prediction_normal[number][class_nr]
#                     )

class TestLoadModuleOnlyInference(unittest.TestCase):
    def test_main(self):
        test_datasets, cc_test_datasets_multiclass_df = load_openml_list(
            open_cc_dids[:1],
            multiclass=True,
            shuffled=True,
            filter_for_nan=False,
            max_samples=10000,
            num_feats=100,
            return_capped=True,
        )

        clf = TabPFNClassifier(
            device="cpu", only_inference=True
        )

        for dataset in test_datasets:
            xs, ys = dataset[1].clone(), dataset[2].clone()
            eval_position = xs.shape[0] // 2
            train_xs, train_ys = xs[0:eval_position], ys[0:eval_position]
            test_xs, test_ys = xs[eval_position:], ys[eval_position:]
            print(len(train_xs))
            clf.fit(train_xs, train_ys)

            predictions = (
                clf.predict_proba(test_xs)
            )
            print("Accuracy", accuracy_score(test_ys, predictions.argmax(axis=1)))
            # number_of_predictions, number_of_classes = prediction_normal.shape

            # for number in range(number_of_predictions):
            #     for class_nr in range(number_of_classes):
            #         self.assertTrue(
            #             prediction_with_only_inference[number][class_nr]
            #             == prediction_normal[number][class_nr]
            #         )
