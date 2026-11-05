"""Feature preprocessing.

The preprocessor is an sklearn ``ColumnTransformer`` that lives *inside* each
model pipeline. Consequently it is fitted on training data only and simply
applied to validation/test data - no statistic (category vocabulary, mean,
variance) ever leaks from the evaluation split into training.

Transformations
---------------
* categorical (protocol_type, service, flag) -> one-hot encoding.
  ``handle_unknown="ignore"`` maps a category never seen in training to the
  all-zero vector instead of crashing - important in deployment, where new
  services will appear.
* heavy-tailed counts (bytes, duration, ...) -> log1p, then standardisation.
* remaining numeric features (rates in [0,1], binary flags) -> standardisation.
* ``num_outbound_cmds`` (constant 0) -> dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config


def log1p_nonneg(x):
    """log(1 + x) with negative inputs clipped to 0 (counts are never < 0).

    A named module-level function (not a lambda) keeps the pipeline picklable.
    """
    return np.log1p(np.clip(x, 0, None))


def numeric_features() -> tuple[list[str], list[str]]:
    """Split numeric columns into (log-transformed, linear) groups."""
    numeric = [c for c in config.FEATURE_COLUMNS
               if c not in config.CATEGORICAL_FEATURES and c not in config.DROP_FEATURES]
    log_cols = [c for c in numeric if c in config.LOG_FEATURES]
    lin_cols = [c for c in numeric if c not in config.LOG_FEATURES]
    return log_cols, lin_cols


def build_preprocessor() -> ColumnTransformer:
    """Create an unfitted ColumnTransformer implementing the steps above."""
    log_cols, lin_cols = numeric_features()
    log_pipe = Pipeline([
        ("log1p", FunctionTransformer(log1p_nonneg, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             config.CATEGORICAL_FEATURES),
            ("log", log_pipe, log_cols),
            ("num", StandardScaler(), lin_cols),
        ],
        remainder="drop",  # drops num_outbound_cmds and any label columns
        verbose_feature_names_out=True,
    )


def split_xy(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Return raw feature frame X and target vector y for ``task``.

    ``task`` is ``"binary"`` (normal vs attack) or ``"multiclass"``
    (normal / DoS / Probe / R2L / U2R).
    """
    X = df[config.FEATURE_COLUMNS]
    if task == "binary":
        y = df["is_attack"].to_numpy()
    elif task == "multiclass":
        y = df["category"].to_numpy()
    else:
        raise ValueError(task)
    return X, y
