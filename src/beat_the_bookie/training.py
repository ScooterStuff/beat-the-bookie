"""Training routines for each individual model."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt.space import Integer, Real

from .models import GRU, MLP, DeepElasticNetMLP, HybridGRUMLP


# ---------------------------------------------------------------------------
# Linear Regression baseline (predicts Home/Away expected goals)
# ---------------------------------------------------------------------------

def train_linear_baseline(X_train, X_test, y_home_train, y_away_train):
    home_model = LinearRegression().fit(X_train, y_home_train)
    away_model = LinearRegression().fit(X_train, y_away_train)
    home_xg = home_model.predict(X_test)
    away_xg = away_model.predict(X_test)

    def _result(h: float, a: float) -> str:
        return "H" if h > a else ("A" if a > h else "D")

    le = LabelEncoder().fit(["A", "D", "H"])
    preds = le.transform([_result(h, a) for h, a in zip(home_xg, away_xg)])
    return preds, home_xg, away_xg


# ---------------------------------------------------------------------------
# MLP & ElasticNet (with TimeSeriesSplit cross-validation)
# ---------------------------------------------------------------------------

def _train_simple_torch_classifier(model_cls, X_train, y_train, num_classes: int, epochs: int = 50, lr: float = 1e-3, use_elastic: bool = False):
    tscv = TimeSeriesSplit(n_splits=10)
    X_tr = torch.tensor(np.asarray(X_train), dtype=torch.float32)
    y_tr = torch.tensor(np.asarray(y_train), dtype=torch.long)

    fold_acc = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_tr)):
        X_tr_f, X_va_f = X_tr[tr_idx], X_tr[va_idx]
        y_tr_f, y_va_f = y_tr[tr_idx], y_tr[va_idx]
        model = model_cls(X_tr_f.shape[1], num_classes) if use_elastic else model_cls(X_tr_f.shape[1], 64, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(X_tr_f)
            loss = model.elastic_net_loss(out, y_tr_f, criterion) if use_elastic else criterion(out, y_tr_f)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            preds = model(X_va_f).argmax(dim=1)
            fold_acc.append(accuracy_score(y_va_f, preds))
        print(f"  fold {fold + 1}: val_acc={fold_acc[-1]:.4f}")

    final = model_cls(X_tr.shape[1], num_classes) if use_elastic else model_cls(X_tr.shape[1], 64, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        final.train()
        optimizer.zero_grad()
        out = final(X_tr)
        loss = final.elastic_net_loss(out, y_tr, criterion) if use_elastic else criterion(out, y_tr)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return final, fold_acc, losses


def train_mlp(X_train, y_train, num_classes: int, epochs: int = 50):
    print("[MLP] cross-validating...")
    return _train_simple_torch_classifier(MLP, X_train, y_train, num_classes, epochs=epochs, use_elastic=False)


def train_elastic_net(X_train, y_train, num_classes: int, epochs: int = 50):
    print("[ElasticNet] cross-validating...")
    return _train_simple_torch_classifier(DeepElasticNetMLP, X_train, y_train, num_classes, epochs=epochs, use_elastic=True)


def predict_torch(model, X_test) -> np.ndarray:
    X_te = torch.tensor(np.asarray(X_test), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        return model(X_te).argmax(dim=1).numpy()


# ---------------------------------------------------------------------------
# SVM (with RF-based feature selection + grid-search tuning)
# ---------------------------------------------------------------------------

def train_svm(X_train, X_test, y_train, n_features_to_select: int = 10):
    rf_params = OrderedDict([("max_depth", 10), ("min_samples_leaf", 3), ("min_samples_split", 5), ("n_estimators", 100)])
    rf = RandomForestClassifier(**rf_params).fit(X_train, y_train)
    indices = np.argsort(rf.feature_importances_)[::-1][:n_features_to_select]
    X_tr_sel = X_train[:, indices]

    grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]}
    search = GridSearchCV(SVC(), grid, cv=TimeSeriesSplit(n_splits=5), verbose=1)
    search.fit(X_tr_sel, y_train)
    print(f"[SVM] best params: {search.best_params_}")
    print(f"[SVM] best CV score: {search.best_score_:.4f}")

    model = SVC(**search.best_params_, probability=True).fit(X_train, y_train)
    return model, search.best_params_, model.predict(X_test)


# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------

def _to_seq_tensor(X) -> torch.Tensor:
    X = np.asarray(X)
    return torch.FloatTensor(X.reshape((X.shape[0], 1, X.shape[1])))


def train_gru(X_train, y_train, num_classes: int, epochs: int = 50, lr: float = 1e-3):
    X_t = _to_seq_tensor(X_train)
    y_t = torch.LongTensor(y_train.values if isinstance(y_train, pd.Series) else np.asarray(y_train))
    model = GRU(input_size=X_t.shape[2], num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
    return model


def predict_gru(model, X_test) -> np.ndarray:
    X_t = _to_seq_tensor(X_test)
    model.eval()
    with torch.no_grad():
        return torch.argmax(model(X_t), dim=1).numpy()


# ---------------------------------------------------------------------------
# Hybrid GRU + MLP
# ---------------------------------------------------------------------------

def train_hybrid_gru_mlp(X_train, y_train, num_classes: int, epochs: int = 50, lr: float = 3e-4):
    X = np.asarray(X_train)
    Xg = torch.FloatTensor(X.reshape((X.shape[0], 1, X.shape[1])))
    Xm = torch.FloatTensor(X)
    y_t = torch.LongTensor(y_train.values if isinstance(y_train, pd.Series) else np.asarray(y_train))
    model = HybridGRUMLP(seq_features=X.shape[1], n_features=X.shape[1], num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(Xg, Xm)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
    return model


def predict_hybrid_gru_mlp(model, X_test) -> np.ndarray:
    X = np.asarray(X_test)
    Xg = torch.FloatTensor(X.reshape((X.shape[0], 1, X.shape[1])))
    Xm = torch.FloatTensor(X)
    model.eval()
    with torch.no_grad():
        return torch.argmax(model(Xg, Xm), dim=1).numpy()


# ---------------------------------------------------------------------------
# Random Forest + XGBoost (Bayesian hyper-parameter tuning)
# ---------------------------------------------------------------------------

def _bayes_tune(model, search_space, X_train, y_train, n_iter: int = 30, cv_splits: int = 10, scoring=None):
    print(f"  tuning {type(model).__name__}...")
    search = BayesSearchCV(
        model,
        search_spaces=search_space,
        cv=TimeSeriesSplit(cv_splits),
        scoring=scoring,
        n_iter=n_iter,
        n_jobs=-1,
        verbose=1,
        refit=False,
    )
    search.fit(X_train, y_train)
    return search.best_params_


def get_classifier_dict() -> dict:
    return {
        "RandomForestClassifier": {
            "model": lambda: ensemble.RandomForestClassifier(random_state=1000, n_estimators=100),
            "search_space": {
                "max_depth": Integer(3, 20),
                "min_samples_split": Integer(2, 10),
                "min_samples_leaf": Integer(1, 5),
            },
        },
        "XGBClassifier": {
            "model": lambda: xgb.XGBClassifier(
                random_state=999,
                enable_categorical=True,
                objective="multi:softmax",
                n_estimators=100,
            ),
            "search_space": {
                "max_depth": Integer(2, 6),
                "subsample": Real(0.5, 1.0),
                "colsample_bytree": Real(0.5, 1.0),
                "eta": Real(0.01, 0.3, prior="log-uniform"),
            },
        },
    }


def train_ensembles(X_train, X_test, y_train, n_iter: int = 30):
    classifiers = get_classifier_dict()
    for name, info in classifiers.items():
        info["tuned_params"] = _bayes_tune(info["model"](), info["search_space"], X_train, y_train, n_iter=n_iter)

    rf_params = classifiers["RandomForestClassifier"]["tuned_params"]
    xgb_params = classifiers["XGBClassifier"]["tuned_params"]

    rf_model = ensemble.RandomForestClassifier(random_state=100, **rf_params)
    xgb_model = xgb.XGBClassifier(random_state=100, **xgb_params)

    rf_cv = cross_val_score(rf_model, X_train, y_train, cv=TimeSeriesSplit(10))
    xgb_cv = cross_val_score(xgb_model, X_train, y_train, cv=TimeSeriesSplit(10))
    print(f"[RandomForest] CV acc: {np.mean(rf_cv):.4f} ± {np.std(rf_cv):.4f}")
    print(f"[XGBoost]      CV acc: {np.mean(xgb_cv):.4f} ± {np.std(xgb_cv):.4f}")

    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    return {
        "rf_model": rf_model, "rf_params": rf_params, "rf_predictions": rf_model.predict(X_test),
        "xgb_model": xgb_model, "xgb_params": xgb_params, "xgb_predictions": xgb_model.predict(X_test),
    }
