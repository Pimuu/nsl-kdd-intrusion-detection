"""Smoke tests (stdlib unittest, no network needed).

Run with:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src import config
from src.data import add_labels
from src.evaluation import binary_metrics, multiclass_metrics
from src.preprocessing import build_preprocessor, split_xy
from src.synthetic import generate_synthetic_dataset


class TestLabelsAndPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train, test = generate_synthetic_dataset(n_train=3000, n_test=800)
        cls.train, cls.test = add_labels(train), add_labels(test)

    def test_schema(self):
        self.assertEqual(list(self.train.columns[:41]), config.FEATURE_COLUMNS)
        self.assertTrue(set(self.train["category"]) <= set(config.CATEGORY_ORDER))
        self.assertTrue(((self.train["category"] == "normal") == (self.train["is_attack"] == 0)).all())

    def test_every_attack_is_mapped(self):
        cats = set(config.ATTACK_CATEGORY.values())
        self.assertEqual(cats, set(config.CATEGORY_ORDER))

    def test_preprocessor_handles_unseen_categories(self):
        X_tr, _ = split_xy(self.train, "binary")
        X_te, _ = split_xy(self.test, "binary")
        X_te = X_te.copy()
        X_te.iloc[0, X_te.columns.get_loc("service")] = "never_seen_service"
        prep = build_preprocessor().fit(X_tr)
        Z = prep.transform(X_te)
        self.assertEqual(Z.shape[0], len(X_te))
        self.assertTrue(np.isfinite(Z).all())
        self.assertNotIn("num__num_outbound_cmds", prep.get_feature_names_out())


class TestMetrics(unittest.TestCase):
    def test_binary_metrics(self):
        y = np.array([0, 0, 1, 1])
        m = binary_metrics(y, np.array([0, 1, 1, 0]), np.array([0.1, 0.6, 0.9, 0.4]))
        self.assertAlmostEqual(m["accuracy"], 0.5)
        self.assertAlmostEqual(m["false_alarm_rate"], 0.5)
        self.assertAlmostEqual(m["roc_auc"], 0.75)

    def test_multiclass_perfect(self):
        labels = config.CATEGORY_ORDER
        y = np.array(labels * 3)
        proba = np.tile(np.eye(5), (3, 1))
        m = multiclass_metrics(y, y, proba, labels)
        self.assertAlmostEqual(m["macro_f1"], 1.0)
        self.assertAlmostEqual(m["macro_roc_auc_ovr"], 1.0)


class TestPipelineEndToEnd(unittest.TestCase):
    def test_synthetic_run(self):
        from src.pipeline import run

        with tempfile.TemporaryDirectory() as tmp:
            info = run(out_dir=Path(tmp), force_synthetic=True, perm_repeats=1, perm_sample=500)
            self.assertEqual(info["data_source"], "synthetic")
            for f in ["metrics_all.csv", "metrics_comparison_binary.csv",
                      "metrics_comparison_multiclass.csv", "run_info.json",
                      "figures/roc_curve_binary.png", "figures/feature_importance.png",
                      "figures/class_distribution.png", "figures/confusion_binary_random_forest.png"]:
                self.assertTrue((Path(tmp) / f).exists(), f)
            self.assertEqual(json.loads((Path(tmp) / "run_info.json").read_text())["random_state"],
                             config.RANDOM_STATE)


if __name__ == "__main__":
    unittest.main()
