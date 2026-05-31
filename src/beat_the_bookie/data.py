"""Raw data ingestion and cleaning.

Loads per-season scraped match-stat CSVs and produces a cleaned long-format
DataFrame (one row per team-match) ready for feature engineering.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def get_all_season_data(data_dir: str | os.PathLike) -> pd.DataFrame:
    """Concatenate all `*_games.csv` season files inside ``data_dir``."""
    data_dir = Path(data_dir)
    season_files = sorted(p for p in data_dir.iterdir() if p.name.endswith("_games.csv"))
    if not season_files:
        raise FileNotFoundError(f"No *_games.csv files found in {data_dir}")
    frames = [pd.read_csv(p, encoding="latin1") for p in season_files]
    return pd.concat(frames, ignore_index=True)


def _drop_duplicate_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Drop columns whose values exactly duplicate an earlier column."""
    cols = data.columns
    duplicate_cols: list[str] = []
    for i in range(len(cols)):
        for j in range(i):
            if data[cols[i]].dtype == data[cols[j]].dtype and data[cols[i]].equals(data[cols[j]]):
                duplicate_cols.append(cols[i])
                break
    return data.drop(columns=duplicate_cols)


def clean_data(data: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Clean the raw concatenated season data.

    - drops fully-NaN rows, duplicate rows, duplicate columns
    - encodes the home/away flag from the unnamed ``@`` column
    - parses dates and derives the full-time result (FTR) target
    """

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    _log(f"Original Length: {len(data)} rows, {len(data.columns)} cols")
    data = data.dropna(how="all")
    data = data.drop_duplicates()
    data = _drop_duplicate_columns(data)

    # An "@" in the unnamed column indicates the team is playing away
    data["HomeGame"] = ~data["Unnamed: 13"].eq("@")
    data = data.drop(columns=["Unnamed: 13", "Match Report", "Comp", "Rk"])

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.drop(columns=["Result"])

    data["FTR"] = data.apply(
        lambda row: (
            "D"
            if row["GD"] == 0
            else ("H" if (not (row["GD"] > 0) ^ row["HomeGame"]) else "A")
        ),
        axis=1,
    )
    _log(f"After cleaning: {len(data)} rows, {len(data.columns)} cols")
    return data.dropna().copy()
