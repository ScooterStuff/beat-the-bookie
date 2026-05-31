# Beat the Bookie — Premier League Match Outcome Prediction

End-to-end machine learning pipeline that predicts English Premier League
match outcomes (Home win / Draw / Away win) from historical fixture and
team-performance data, originally produced for the **COMP0036 — Machine
Learning & Neural Computing** group coursework at UCL.

The project covers the full ML lifecycle: web-scraped data ingestion,
domain-driven feature engineering (Elo, PI ratings, EMA form), feature
selection, model benchmarking, hyper-parameter tuning, and a hybrid
GRU + MLP neural network used for the final submission.

---

## Highlights

- **Rich feature engineering** — rolling form via Exponential Moving
  Averages, seasonal aggregates, custom Elo rating system, and Pi
  ratings derived from match results.
- **Model zoo benchmarked under time-series cross-validation**:
  Logistic Regression baseline, SVM, Random Forest, XGBoost, an
  ElasticNet-regularised deep MLP, a GRU sequence model, and a
  hybrid **GRU + MLP** architecture (final model).
- **Principled tuning** — Bayesian hyper-parameter search
  (`scikit-optimize`) and class-imbalance handling with SMOTE /
  random oversampling.
- **Reproducible** — single self-contained Jupyter notebook plus a
  pinned `requirements.txt`.

## Repository layout

```
.
├── Notebook-Beat_The_Bookie_Coursework_Final_Python_Notebook.ipynb
│       # End-to-end pipeline: data → features → models → predictions
├── Report-Beat_the_Bookie-1.pdf
│       # Written report (methodology, results, discussion)
├── data/
│   ├── 1718_games.csv … 2425_games.csv   # Per-season match & team stats
│   ├── team_elo.csv                       # Pre-computed Elo ratings
│   └── sample-submission.csv              # Submission template
├── FINAL_PREDICTIONS.csv                  # Submitted predictions
├── requirements.txt
├── LICENSE
└── README.md
```

## Data

Eight seasons of Premier League data (2017/18 – 2024/25) were scraped
with Selenium from public football-statistics sources. Each
`xxyy_games.csv` contains per-team, per-match advanced metrics (xG,
npxG, SCA/GCA, pass completion, defensive actions, possession, etc.),
which are merged into home/away match rows during preprocessing.

## Methodology (summary)

1. **Cleaning** — drop duplicate columns and rows, harmonise team
   names, handle NaNs.
2. **Feature engineering** — performance indicators, EMA form
   features, seasonal aggregates, Elo + Pi ratings.
3. **Feature selection** — correlation pruning and Random-Forest
   importance ranking.
4. **Preprocessing** — label encoding, scaling, imputation, SMOTE
   for class imbalance.
5. **Modelling** — time-series-aware splits, Bayesian tuning per
   model, comparison on log-loss / accuracy / F1.
6. **Final model** — hybrid GRU (sequence of recent fixtures) +
   MLP (current-match features) trained in PyTorch.

Full details, plots and quantitative results are in
[Report-Beat_the_Bookie-1.pdf](Report-Beat_the_Bookie-1.pdf).

## Getting started

```bash
# 1. Clone
git clone https://github.com/ScooterStuff/beat-the-bookie.git
cd beat-the-bookie

# 2. (Recommended) create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook Notebook-Beat_The_Bookie_Coursework_Final_Python_Notebook.ipynb
```

The notebook expects the `data/` folder at the repository root — no
further configuration is required.

## Tech stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` ·
`imbalanced-learn` · `scikit-optimize` · `PyTorch` · `TensorFlow / Keras`
· `Matplotlib` · `seaborn`

## Acknowledgements

Coursework brief and starter data provided by the COMP0036 teaching
team, UCL. Additional match statistics scraped from public sources
for educational use only.

## License

Released under the [MIT License](LICENSE). Data files are included
strictly for academic reproduction of the coursework results.
