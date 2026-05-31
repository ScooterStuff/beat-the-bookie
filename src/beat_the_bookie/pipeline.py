"""High-level pipelines that orchestrate the modules end-to-end.

Two entry points are exposed:

- :func:`run_full_training` – feature-engineers the historical data,
  trains every model, prints a comparison table and confusion-matrix
  plots; useful for reproducing the coursework experiments.
- :func:`run_predictions` – produces the final stacking-ensemble
  predictions for a given test-fixture CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import get_all_season_data
from .evaluation import plot_confusion_matrix, report_model, summary_table
from .features import transform_raw_data
from .predict import get_predictions
from .preprocess import preprocess, time_split
from .training import (
    predict_gru,
    predict_hybrid_gru_mlp,
    predict_torch,
    train_elastic_net,
    train_ensembles,
    train_gru,
    train_hybrid_gru_mlp,
    train_linear_baseline,
    train_mlp,
    train_svm,
)


def run_full_training(
    data_dir: str = "data",
    *,
    epochs: int = 50,
    bayes_iter: int = 30,
    plots_dir: str | None = "outputs/plots",
    verbose: bool = True,
):
    """Reproduce the full training + evaluation workflow from the report."""
    print("=== Loading & feature engineering ===")
    raw = get_all_season_data(data_dir)
    df_features = transform_raw_data(raw, data_dir=data_dir, verbose=verbose)
    print(f"Final feature dataset: {df_features.shape}")

    pp = preprocess(df_features)
    X_train, X_test, y_train, y_test = time_split(pp.X, pp.y)
    home_train, _ = pp.y_home_goals[: len(X_train)], pp.y_home_goals[len(X_train):]
    away_train, _ = pp.y_away_goals[: len(X_train)], pp.y_away_goals[len(X_train):]

    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    n_classes = len(pp.label_encoder.classes_)
    results: list[dict] = []
    all_preds: dict[str, np.ndarray] = {}

    print("\n=== Linear regression baseline ===")
    lin_preds, *_ = train_linear_baseline(X_train, X_test, home_train, away_train)
    all_preds["Linear_Regression"] = lin_preds

    print("\n=== MLP ===")
    mlp_model, mlp_folds, _ = train_mlp(X_train, y_train, n_classes, epochs=epochs)
    print(f"MLP mean fold acc: {np.mean(mlp_folds):.4f}")
    all_preds["MLP"] = predict_torch(mlp_model, X_test)

    print("\n=== Elastic-Net deep MLP ===")
    en_model, en_folds, _ = train_elastic_net(X_train, y_train, n_classes, epochs=epochs)
    print(f"ElasticNet mean fold acc: {np.mean(en_folds):.4f}")
    all_preds["Elastic_Net"] = predict_torch(en_model, X_test)

    print("\n=== SVM ===")
    svm_model, svm_params, svm_preds = train_svm(X_train, X_test, y_train)
    all_preds["SVM"] = svm_preds

    print("\n=== GRU ===")
    gru_model = train_gru(X_train, y_train, n_classes, epochs=epochs)
    all_preds["GRU"] = predict_gru(gru_model, X_test)

    print("\n=== Hybrid GRU + MLP (final model) ===")
    hybrid_model = train_hybrid_gru_mlp(X_train, y_train, n_classes, epochs=epochs)
    all_preds["GRU+MLP"] = predict_hybrid_gru_mlp(hybrid_model, X_test)

    print("\n=== Random Forest + XGBoost (Bayesian tuning) ===")
    ens = train_ensembles(X_train, X_test, y_train, n_iter=bayes_iter)
    all_preds["Random_Forest"] = ens["rf_predictions"]
    all_preds["XGB"] = ens["xgb_predictions"]

    print("\n=== Stacking ensemble ===")
    from .stacking import get_stacking_predictions
    stacking_preds = get_stacking_predictions(
        X_train, y_train, X_test,
        svm_params=svm_params, rf_params=ens["rf_params"], xgb_params=ens["xgb_params"],
    )
    all_preds["Stacking_Ensemble"] = stacking_preds

    if plots_dir:
        plot_dir = Path(plots_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

    for name, preds in all_preds.items():
        results.append(report_model(name, y_test, preds))
        if plots_dir:
            plot_confusion_matrix(
                y_test, preds, name,
                label_classes=pp.label_encoder.classes_,
                save_path=str(Path(plots_dir) / f"{name}_confusion_matrix.png"),
            )

    table = summary_table(results)
    print("\n=== Final comparison ===")
    print(table.to_string(index=False))

    return {
        "table": table,
        "predictions": all_preds,
        "preprocessed": pp,
        "svm_params": svm_params,
        "rf_params": ens["rf_params"],
        "xgb_params": ens["xgb_params"],
        "y_test": y_test,
    }


def run_predictions(
    test_data_path: str,
    data_dir: str = "data",
    output_path: str = "FINAL_PREDICTIONS.csv",
    *,
    svm_params: dict | None = None,
    rf_params: dict | None = None,
    xgb_params: dict | None = None,
    features_to_remove: set[str] | None = None,
):
    """Produce the final stacking-ensemble predictions and save to CSV.

    If hyper-parameters or ``features_to_remove`` are not supplied, sensible
    defaults derived from the historical data are computed on the fly.
    """
    if features_to_remove is None or svm_params is None or rf_params is None or xgb_params is None:
        # Derive missing settings by running the historical pipeline once.
        from .preprocess import select_correlated_features
        raw = get_all_season_data(data_dir)
        df_features = transform_raw_data(raw, data_dir=data_dir)
        if features_to_remove is None:
            features_to_remove = select_correlated_features(df_features)
        if svm_params is None or rf_params is None or xgb_params is None:
            pp = preprocess(df_features, features_to_remove=features_to_remove)
            X_train, X_test, y_train, _ = time_split(pp.X, pp.y)
            if svm_params is None:
                _, svm_params, _ = train_svm(X_train, X_test, y_train)
            if rf_params is None or xgb_params is None:
                ens = train_ensembles(X_train, X_test, y_train)
                rf_params = rf_params or ens["rf_params"]
                xgb_params = xgb_params or ens["xgb_params"]

    df_pred = get_predictions(
        test_data_path=test_data_path,
        data_dir=data_dir,
        features_to_remove=features_to_remove,
        svm_params=svm_params,
        rf_params=rf_params,
        xgb_params=xgb_params,
    )
    df_pred.to_csv(output_path, index=False)
    print(f"Saved {len(df_pred)} predictions → {output_path}")
    return df_pred
