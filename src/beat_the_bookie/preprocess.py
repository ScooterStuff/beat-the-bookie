"""Feature selection, encoding, scaling, and time-aware train/test split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler


OBJECT_FORM_COLUMNS = [
    "f_HM1", "f_HM2", "f_HM3", "f_HM4", "f_HM5",
    "f_AM1", "f_AM2", "f_AM3", "f_AM4", "f_AM5",
    "f_HTFormPtsStr", "f_ATFormPtsStr",
]


@dataclass
class PreprocessedData:
    X: np.ndarray
    y: np.ndarray
    label_encoder: LabelEncoder
    features_to_remove: set[str]
    y_home_goals: pd.Series
    y_away_goals: pd.Series
    data: pd.DataFrame  # data with year column added, after column drops


def select_correlated_features(df_features: pd.DataFrame, threshold: float = 0.9) -> set[str]:
    """Drop the less-important member of every (>threshold) correlated feature pair.

    Importance is decided by Random Forest feature importance computed on
    standard-scaled numeric ``f_*`` columns.
    """
    numeric_data = df_features.select_dtypes(include=["number"])
    numeric_data = numeric_data[[c for c in numeric_data.columns if c.startswith("f_")]]

    rf = RandomForestClassifier(random_state=100)
    rf.fit(StandardScaler().fit_transform(numeric_data), df_features["FTR"])
    importances = rf.feature_importances_

    pairwise = numeric_data.corr().unstack().sort_values(ascending=False)
    similar = pairwise[(pairwise > threshold) & (pairwise != 1.0)]

    cols_idx = {c: i for i, c in enumerate(numeric_data.columns)}
    to_remove: set[str] = set()
    for col1, col2 in similar.index:
        if importances[cols_idx[col1]] < importances[cols_idx[col2]]:
            to_remove.add(col1)
        else:
            to_remove.add(col2)
    return to_remove


def preprocess(df_features: pd.DataFrame, features_to_remove: Iterable[str] | None = None) -> PreprocessedData:
    """Encode target, drop string form columns + correlated features, scale & impute."""
    if features_to_remove is None:
        features_to_remove = select_correlated_features(df_features)
    features_to_remove = set(features_to_remove)

    data = df_features.copy()
    data["FTR"] = data["FTR"].astype("category")
    data = data.drop(columns=OBJECT_FORM_COLUMNS, errors="ignore")
    data = data.drop(columns=list(features_to_remove), errors="ignore")

    label_encoder = LabelEncoder()
    data["FTR"] = label_encoder.fit_transform(data["FTR"])
    data["year"] = data["Date"].dt.year

    feature_cols = [c for c in data.columns if c.startswith("f_")]
    X = data[feature_cols].to_numpy()
    y = data["FTR"].to_numpy()

    X = SimpleImputer(strategy="mean").fit_transform(X)
    X = StandardScaler().fit_transform(X)

    return PreprocessedData(
        X=X,
        y=y,
        label_encoder=label_encoder,
        features_to_remove=features_to_remove,
        y_home_goals=data["HomeGoals"],
        y_away_goals=data["AwayGoals"],
        data=data,
    )


def time_split(X: np.ndarray, y: np.ndarray, train_frac: float = 0.8) -> tuple:
    """Sequential 80/20 split (no shuffling — preserves temporal order)."""
    n = len(X)
    cut = int(train_frac * n)
    return X[:cut], X[cut:], y[:cut], y[cut:]
