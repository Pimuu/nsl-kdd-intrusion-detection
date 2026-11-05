"""Model zoo.

Every model is wrapped in a full sklearn ``Pipeline(preprocess -> classifier)``
so that preprocessing is re-fitted on whatever data the model is trained on.

Models compared
---------------
* ``majority``  - always predicts the most frequent training class. A floor
                  that every real model must beat.
* ``logreg``    - multinomial logistic regression. Linear, fast, well
                  calibrated, interpretable; the classic baseline.
* ``random_forest`` - bagged ensemble of deep decision trees (Breiman, 2001).
                  Handles mixed feature types and non-linear interactions with
                  little tuning; a strong NSL-KDD baseline in the literature.
* ``hist_gb``   - histogram-based gradient-boosted trees (LightGBM-style
                  boosting, Friedman 2001 / Ke et al. 2017) as implemented in
                  scikit-learn.

Hyper-parameters are fixed, conventional values rather than tuned on KDDTest+;
tuning on the test set would leak information and inflate the results. (A
random validation split of KDDTrain+ is uninformative for tuning here: every
model scores >99% on it - see the paper, Section 4.2.)
"""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config
from .preprocessing import build_preprocessor

# Human-readable names used in tables and figure legends.
MODEL_LABELS = {
    "majority": "Majority baseline",
    "logreg": "Logistic regression",
    "random_forest": "Random forest",
    "hist_gb": "Gradient boosting",
}
MODEL_NAMES = list(MODEL_LABELS)


def make_classifier(name: str, class_weight: str | None = None, seed: int = config.RANDOM_STATE):
    """Return an unfitted classifier.

    ``class_weight="balanced"`` re-weights samples inversely to class frequency,
    which matters for the rare R2L/U2R classes (used in the ablation study).
    """
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "logreg":
        return LogisticRegression(C=1.0, max_iter=3000, class_weight=class_weight,
                                  random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=1,
                                      max_features="sqrt", class_weight=class_weight,
                                      n_jobs=-1, random_state=seed)
    if name == "hist_gb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                              max_leaf_nodes=31, l2_regularization=1.0,
                                              early_stopping=False,
                                              class_weight=class_weight, random_state=seed)
    raise KeyError(name)


def make_pipeline(name: str, class_weight: str | None = None) -> Pipeline:
    """Preprocessing + classifier as one estimator."""
    return Pipeline([
        ("prep", build_preprocessor()),
        ("clf", make_classifier(name, class_weight=class_weight)),
    ])
