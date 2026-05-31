# Beat the Bookie — Premier League Match Outcome Prediction

End-to-end machine learning pipeline that predicts English Premier League
match outcomes (Home win / Draw / Away win) from historical fixture and
team-performance data, originally produced for the **COMP0036 — Machine
Learning & Neural Computing** group coursework at UCL.

The project covers the full ML lifecycle: data ingestion, domain-driven
feature engineering (Elo, Pi ratings, EMA form, seasonal aggregates),
feature selection, model benchmarking, hyper-parameter tuning, and a
hybrid GRU + MLP neural network combined with a stacking ensemble for
the final submission.

> **Note:** the original Jupyter notebook has been refactored into a
> clean, modular Python package (`src/beat_the_bookie/`) with a CLI
> entry point — no notebook is required to reproduce the results.

---

## Highlights

- **Rich feature engineering** — exponential moving averages of every
  numeric statistic, seasonal aggregates (cumulative goals, points,
  win/loss streaks, recent form), pre-computed Elo ratings and Pi
  ratings derived from match results.
- **Model zoo benchmarked under time-series cross-validation:**
  Linear-Regression baseline, SVM (RF-selected features +
  `GridSearchCV`), Random Forest and XGBoost (Bayesian tuning via
  `scikit-optimize`), an Elastic-Net regularised deep MLP, a GRU
  sequence model, and a hybrid **GRU + MLP** architecture.
- **Stacking ensemble** — out-of-fold base-model probabilities fed
  into a Logistic-Regression meta-learner, used for the final
  predictions.
- **Reproducible** — pinned `requirements.txt`, `pyproject.toml` for
  installable package, deterministic seeds, single-command CLI.

## Repository layout

```
.
├── main.py                       # CLI entry point (`train` / `predict`)
├── src/beat_the_bookie/
│   ├── data.py                   # raw season-data loading & cleaning
│   ├── features.py               # performance indicators, EMA, seasonal, Elo, Pi
│   ├── preprocess.py             # feature selection, encoding, scaling, splitting
│   ├── models.py                 # MLP / GRU / DeepElasticNetMLP / HybridGRUMLP + sklearn wrappers
│   ├── training.py               # per-model training & cross-validation
│   ├── stacking.py               # stacking ensemble (LogReg meta-learner)
│   ├── predict.py                # final inference pipeline
│   ├── evaluation.py             # reports & confusion-matrix plots
│   └── pipeline.py               # high-level `run_full_training` / `run_predictions`
├── data/
│   ├── 1718_games.csv … 2425_games.csv
│   ├── team_elo.csv
│   └── sample-submission.csv
├── FINAL_PREDICTIONS.csv         # submitted predictions
├── Report-Beat_the_Bookie-1.pdf  # written report
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

## Quick start

```bash
# 1. Clone
git clone https://github.com/ScooterStuff/beat-the-bookie.git
cd beat-the-bookie

# 2. (Recommended) create a virtual environment
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

# 3. Install
pip install -r requirements.txt
# or, to install as a package:
pip install -e .

# 4. Run the full training + evaluation
python main.py train

# 5. Generate final predictions on a fixture CSV
python main.py predict --input data/sample-submission.csv --output FINAL_PREDICTIONS.csv
```

## Programmatic use

```python
from beat_the_bookie.pipeline import run_full_training, run_predictions

# Train every model, get a comparison table of weighted F1 / accuracy:
results = run_full_training(data_dir="data", epochs=50)
print(results["table"])

# Produce the submission file:
run_predictions(
    test_data_path="data/sample-submission.csv",
    data_dir="data",
    output_path="FINAL_PREDICTIONS.csv",
)
```

## Data

Eight Premier League seasons (2017/18 – 2024/25) of advanced match
statistics scraped from public football-statistics sources. Each
`xxyy_games.csv` contains per-team, per-match metrics (xG, npxG,
SCA/GCA, pass completion, defensive actions, possession, etc.), which
are merged into per-match rows during feature engineering.

## Methodology (summary)

1. **Cleaning** — drop duplicate rows/columns, harmonise team names,
   handle NaNs.
2. **Feature engineering** — domain-driven performance indicators
   (DCM/ACM/PIM/EM/EPM/...), per-team EMA form, seasonal aggregates,
   Elo and Pi ratings.
3. **Feature selection** — correlation pruning guided by Random
   Forest importance.
4. **Preprocessing** — label encoding, mean imputation, standard
   scaling.
5. **Modelling** — `TimeSeriesSplit` cross-validation, Bayesian
   hyper-parameter search, model comparison on accuracy / F1.
6. **Final model** — stacking ensemble combining a hybrid GRU + MLP,
   an Elastic-Net deep MLP, an SVM, a Random Forest, and an XGBoost
   classifier, with a Logistic-Regression meta-learner.

Quantitative results, plots and discussion are in
[Report-Beat_the_Bookie-1.pdf](Report-Beat_the_Bookie-1.pdf).

## Tech stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` ·
`imbalanced-learn` · `scikit-optimize` · `PyTorch` · `TensorFlow / Keras`
· `Matplotlib` · `seaborn`

## Acknowledgements

Coursework brief and starter data provided by the COMP0036 teaching
team, UCL. Additional match statistics scraped from public sources
for educational use only.

## License

Released under the [MIT License](LICENSE).
