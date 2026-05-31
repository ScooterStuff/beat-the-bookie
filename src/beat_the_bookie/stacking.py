"""Stacking ensemble combining the base models with a Logistic Regression meta-learner."""

from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVC

from .models import SklearnCompatibleElasticNetMLP, SklearnCompatibleGRUMLP


def get_stacking_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    svm_params: dict,
    rf_params: dict,
    xgb_params: dict,
    n_splits: int = 10,
    grumlp_epochs: int = 10,
    elastic_epochs: int = 50,
) -> np.ndarray:
    """Run the stacking ensemble end-to-end and return test predictions.

    1. Generate out-of-fold ``predict_proba`` features from each base learner
       on the training set.
    2. Fit a Logistic Regression meta-model on the stacked OOF features.
    3. Re-train the base learners on the full train set and stack their
       test-set probabilities for the meta-model to predict on.
    """
    n_classes = len(set(np.asarray(y_train).tolist()))

    base_factory = lambda: [
        SklearnCompatibleGRUMLP(num_epochs=grumlp_epochs),
        SklearnCompatibleElasticNetMLP(num_epochs=elastic_epochs),
        SVC(**svm_params, probability=True, random_state=100),
        RandomForestClassifier(**rf_params, random_state=100),
        xgb.XGBClassifier(objective="multi:softmax", **xgb_params, random_state=100),
    ]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    base_models = base_factory()

    # 1) out-of-fold meta-features on the train set
    meta_train_blocks: list[np.ndarray] = []
    for base_model in base_models:
        meta_train = np.zeros((X_train.shape[0], n_classes))
        for tr_idx, te_idx in tscv.split(X_train):
            base_model.fit(X_train[tr_idx], y_train[tr_idx])
            meta_train[te_idx] = base_model.predict_proba(X_train[te_idx])
        meta_train_blocks.append(meta_train)
    X_train_meta = np.hstack(meta_train_blocks)

    meta_model = LogisticRegression(random_state=100, max_iter=1500)
    meta_model.fit(X_train_meta, y_train)

    # 2) re-train base models on full train set, stack test probabilities
    base_models = base_factory()
    meta_test_blocks: list[np.ndarray] = []
    for base_model in base_models:
        base_model.fit(X_train, y_train)
        meta_test_blocks.append(base_model.predict_proba(X_test))
    X_test_meta = np.hstack(meta_test_blocks)

    return meta_model.predict(X_test_meta)
