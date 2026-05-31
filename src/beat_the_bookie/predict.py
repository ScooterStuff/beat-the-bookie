"""Final-prediction pipeline (Section 7 of the original notebook).

Builds a combined train+test dataset (with team-name aliases mapped),
runs the full feature pipeline, then trains the stacking ensemble on
everything before the cut-off date and predicts the test fixtures.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .data import get_all_season_data
from .features import transform_raw_data
from .preprocess import OBJECT_FORM_COLUMNS


TEAM_NAME_REMAPPING = {
    "AFC Bournemouth": "Bournemouth",
    "Man City": "Manchester City",
    "Nottingham Forest": "Nott'ham Forest",
    "Newcastle": "Newcastle Utd",
    "Spurs": "Tottenham",
    "Man Utd": "Manchester Utd",
}


def _restructure_test_data(df: pd.DataFrame) -> pd.DataFrame:
    """Expand each test fixture into the long (per-team) format expected upstream."""
    new_rows: list[dict] = []
    for _, row in df.iterrows():
        new_rows.append({
            "Date": pd.to_datetime(row["Date"]),
            "Team": TEAM_NAME_REMAPPING.get(row["HomeTeam"], row["HomeTeam"]),
            "Opp": TEAM_NAME_REMAPPING.get(row["AwayTeam"], row["AwayTeam"]),
            "Unnamed: 13": "",
        })
        new_rows.append({
            "Date": pd.to_datetime(row["Date"]),
            "Team": TEAM_NAME_REMAPPING.get(row["AwayTeam"], row["AwayTeam"]),
            "Opp": TEAM_NAME_REMAPPING.get(row["HomeTeam"], row["HomeTeam"]),
            "Unnamed: 13": "@",
        })
    return pd.DataFrame(new_rows)


def get_combined_data(test_data_path: str | os.PathLike, data_dir: str | os.PathLike) -> pd.DataFrame:
    """Concat scraped historical seasons + the (fixture-only) test set, then build features."""
    raw = get_all_season_data(data_dir)
    test_data = pd.read_csv(test_data_path)
    test_data = _restructure_test_data(test_data)
    test_data = test_data.reindex(columns=raw.columns).fillna(0)
    combined = pd.concat([raw, test_data], ignore_index=True)
    return transform_raw_data(combined, data_dir=data_dir)


def _prepare_for_stacking(data: pd.DataFrame, features_to_remove: set[str]):
    data = data.copy()
    data["FTR"] = data["FTR"].astype("category")
    encoder = LabelEncoder()
    encoder.fit(data["FTR"])
    data["FTR"] = encoder.transform(data["FTR"])

    data = data.drop(columns=OBJECT_FORM_COLUMNS, errors="ignore")
    data = data.drop(columns=list(features_to_remove), errors="ignore")
    feature_cols = [c for c in data.columns if c.startswith("f_")]
    X = data[feature_cols].to_numpy()
    X = SimpleImputer(strategy="mean").fit_transform(X)
    X = StandardScaler().fit_transform(X)
    return X, data["FTR"].to_numpy(), encoder


def get_predictions(
    test_data_path: str | os.PathLike,
    data_dir: str | os.PathLike,
    features_to_remove: set[str],
    svm_params: dict,
    rf_params: dict,
    xgb_params: dict,
    cutoff_date: str = "2025-02-01",
) -> pd.DataFrame:
    """End-to-end inference: returns a DataFrame of predicted FTRs for the test fixtures."""
    from .stacking import get_stacking_predictions

    df_features = get_combined_data(test_data_path, data_dir)
    X, y, encoder = _prepare_for_stacking(df_features, features_to_remove)

    mask = df_features["Date"] >= cutoff_date
    X_train, y_train = X[~mask], y[~mask]
    X_test = X[mask]

    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    preds = get_stacking_predictions(
        X_train, y_train, X_test,
        svm_params=svm_params, rf_params=rf_params, xgb_params=xgb_params,
    )
    y_pred = encoder.inverse_transform(preds)

    out = df_features[mask].copy()
    out["FTR"] = y_pred
    return out[["Date", "HomeTeam", "AwayTeam", "FTR"]].reset_index(drop=True)
