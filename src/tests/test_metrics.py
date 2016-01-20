from unittest import TestCase

import numpy as np

import toolkit.metrics as metrics


class TestMotion(TestCase):
    def test_true_positives(self):
        y_true = np.array([[1, 1, 1, 1],
                           [1, 0, 1, 1],
                           [0, 0, 0, 1]])
        y_pred = np.array([[0, 1, 1, 1],
                           [0, 1, 1, 1],
                           [0, 1, 0, 1]])
        tp, _, _, _ = metrics.multilabel_tp_fp_tn_fn_scores(y_true, y_pred)
        self.assertSequenceEqual(tp.tolist(), [0, 1, 2, 3])

    def test_false_positives(self):
        y_true = np.array([[1, 1, 1, 0],
                           [1, 1, 0, 0],
                           [1, 0, 0, 0]])
        y_pred = np.array([[0, 0, 1, 1],
                           [1, 0, 1, 1],
                           [1, 1, 1, 1]])
        _, fp, _, _ = metrics.multilabel_tp_fp_tn_fn_scores(y_true, y_pred)
        self.assertSequenceEqual(fp.tolist(), [0, 1, 2, 3])

    def test_true_negatives(self):
        y_true = np.array([[1, 1, 1, 0],
                           [1, 1, 0, 0],
                           [1, 0, 0, 0]])
        y_pred = np.array([[1, 0, 1, 0],
                           [1, 0, 0, 0],
                           [1, 0, 0, 0]])
        _, _, tn, _ = metrics.multilabel_tp_fp_tn_fn_scores(y_true, y_pred)
        self.assertSequenceEqual(tn.tolist(), [0, 1, 2, 3])

    def test_false_negatives(self):
        y_true = np.array([[1, 0, 0, 1],
                           [1, 1, 1, 1],
                           [1, 0, 1, 1]])
        y_pred = np.array([[1, 0, 1, 0],
                           [1, 0, 0, 0],
                           [1, 0, 0, 0]])
        _, _, _, fn = metrics.multilabel_tp_fp_tn_fn_scores(y_true, y_pred)
        self.assertSequenceEqual(fn.tolist(), [0, 1, 2, 3])