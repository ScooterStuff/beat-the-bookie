"""Feature engineering.

Turns the cleaned long-format match data into a wide per-match dataset
augmented with:
- domain-driven performance indicators (DCM, ACM, PIM, EM, EPM, ...)
- exponential moving averages (EMA) of recent form
- seasonal aggregates (cumulative goals, points, win/loss streaks, form)
- pre-computed Elo ratings
- Pi ratings (home/away rating system)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Performance indicators
# ---------------------------------------------------------------------------

def _normalise(series: pd.Series) -> pd.Series:
    return (series - series.min()) / (series.max() - series.min())


def calculate_defensive_capability(df: pd.DataFrame) -> pd.Series:
    weights = {"TklW": 0.25, "Int": 0.25, "Blocks": 0.20, "Recov": 0.15, "Clr": 0.10, "Def 3rd": 0.05}
    return _normalise(sum(weights[f] * df[f] for f in weights))


def calculate_attacking_capability(df: pd.DataFrame) -> pd.Series:
    weights = {"KP": 0.35, "SCA": 0.3, "GCA": 0.2, "PrgP": 0.15}
    return _normalise(sum(weights[f] * df[f] for f in weights))


def calculate_possession_influence(df: pd.DataFrame) -> pd.Series:
    weights = {"Poss": 0.3, "Cmp%": 0.25, "PrgDist": 0.2, "Carries": 0.15, "PrgC": 0.1}
    return _normalise(sum(weights[f] * df[f] for f in weights))


def calculate_efficiency(df: pd.DataFrame) -> pd.Series:
    weights = {"G/Sh": 0.4, "SoT%": 0.3, "Tkl%": 0.2, "Cmp%": 0.1}
    return _normalise(sum(weights[f] * df[f] for f in weights))


def calculate_expected_performance(df: pd.DataFrame) -> pd.Series:
    weights = {"xG": 0.4, "xA": 0.3, "xGD": 0.2, "npxGD": 0.1}
    return _normalise(sum(weights[f] * df[f] for f in weights))


def calculate_physical_intensity(df: pd.DataFrame) -> pd.Series:
    weights = {"Tkl": 0.45, "Int": 0.4, "TotDist": 0.15}
    return _normalise(sum(weights[f] * df[f] for f in weights))


def calculate_creativity(df: pd.DataFrame) -> pd.Series:
    weights = {"KP": 0.4, "SCA": 0.3, "PrgP": 0.2, "1/3": 0.1}
    return _normalise(sum(weights[f] * df[f] for f in weights))


def add_performance_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add the seven domain-driven performance metrics in-place."""
    data["DCM"] = calculate_defensive_capability(data)
    data["ACM"] = calculate_attacking_capability(data)
    data["PIM"] = calculate_possession_influence(data)
    data["EM"] = calculate_efficiency(data)
    data["EPM"] = calculate_expected_performance(data)
    data["Physical_Intensity"] = calculate_physical_intensity(data)
    data["Creativity"] = calculate_creativity(data)
    return data


# ---------------------------------------------------------------------------
# EMA features + restructure into per-match rows
# ---------------------------------------------------------------------------

def create_ema_features(data: pd.DataFrame, span: int = 30) -> pd.DataFrame:
    """Add per-team exponential moving averages of every numeric feature."""
    non_feature_cols = ["Date", "HomeGame", "FTR", "Team", "Opp"]
    feature_names = data.drop(columns=non_feature_cols).columns
    ema_features = []
    for feature_name in feature_names:
        feature_ema = data.groupby("Team")[feature_name].transform(
            lambda row: row.ewm(span=span, min_periods=2).mean().shift(1)
        )
        ema_features.append(pd.Series(feature_ema, name="f_ema_" + feature_name))
    out = pd.concat([data, pd.concat(ema_features, axis=1)], axis=1)
    return out.dropna().reset_index(drop=True)


def restructure_data(data: pd.DataFrame) -> pd.DataFrame:
    """Merge home and away rows for each match into a single per-match row."""
    unwanted_cols = ["Opp_home", "Opp_away", "HomeGame_home", "HomeGame_away"]
    data_merged = (
        data.query("HomeGame == True")
        .rename(columns={"Team": "HomeTeam"})
        .pipe(
            pd.merge,
            data.query("HomeGame == False")
            .rename(columns={"Team": "AwayTeam"})
            .drop(columns="FTR"),
            left_on=["Date", "HomeTeam", "Opp"],
            right_on=["Date", "Opp", "AwayTeam"],
            suffixes=("_home", "_away"),
        )
    )
    data_merged = data_merged.drop(columns=unwanted_cols)
    return data_merged.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Seasonal features
# ---------------------------------------------------------------------------

def _reset_team_stats() -> dict:
    return {
        "GoalsScored": 0,
        "GoalsConceded": 0,
        "Points": 0,
        "Form": ["M"] * 5,
        "WinStreak3": 0,
        "WinStreak5": 0,
        "LossStreak3": 0,
        "LossStreak5": 0,
    }


def _update_streaks(form: list[str]) -> tuple[int, int, int, int]:
    return (
        int(form[:3] == ["W", "W", "W"]),
        int(form[:5] == ["W", "W", "W", "W", "W"]),
        int(form[:3] == ["L", "L", "L"]),
        int(form[:5] == ["L", "L", "L", "L", "L"]),
    )


def add_seasonal_features(df: pd.DataFrame, season_gap_days: int = 60) -> pd.DataFrame:
    """Add cumulative season stats, recent-form letters and streak flags."""
    team_stats: dict[str, dict] = {}
    matchweek = 1
    df = df.sort_values(by="Date", ascending=True)
    last_date = df["Date"].iloc[0]
    new_features: list[dict] = []

    form_points = {"W": 3, "D": 1, "L": 0}

    for _, row in df.iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]
        fthg, ftag = row["GF_home"], row["GF_away"]
        ftr = row["FTR"]
        date = row["Date"]

        if (date - last_date).days > season_gap_days:
            team_stats.clear()
            matchweek = 1

        team_stats.setdefault(home_team, _reset_team_stats())
        team_stats.setdefault(away_team, _reset_team_stats())
        home = team_stats[home_team]
        away = team_stats[away_team]

        htgd = home["GoalsScored"] - home["GoalsConceded"]
        atgd = away["GoalsScored"] - away["GoalsConceded"]
        ht_form_pts = sum(form_points.get(x, 0) for x in home["Form"])
        at_form_pts = sum(form_points.get(x, 0) for x in away["Form"])

        (home["WinStreak3"], home["WinStreak5"], home["LossStreak3"], home["LossStreak5"]) = _update_streaks(home["Form"])
        (away["WinStreak3"], away["WinStreak5"], away["LossStreak3"], away["LossStreak5"]) = _update_streaks(away["Form"])

        row_stats = {
            "HTGS": home["GoalsScored"], "ATGS": away["GoalsScored"],
            "HTGC": home["GoalsConceded"], "ATGC": away["GoalsConceded"],
            "HTP": home["Points"], "ATP": away["Points"], "MatchWeek": matchweek,
            "HTWinStreak3": home["WinStreak3"], "HTWinStreak5": home["WinStreak5"],
            "HTLossStreak3": home["LossStreak3"], "HTLossStreak5": home["LossStreak5"],
            "ATWinStreak3": away["WinStreak3"], "ATWinStreak5": away["WinStreak5"],
            "ATLossStreak3": away["LossStreak3"], "ATLossStreak5": away["LossStreak5"],
            "HM1": home["Form"][0], "HM2": home["Form"][1], "HM3": home["Form"][2],
            "HM4": home["Form"][3], "HM5": home["Form"][4],
            "AM1": away["Form"][0], "AM2": away["Form"][1], "AM3": away["Form"][2],
            "AM4": away["Form"][3], "AM5": away["Form"][4],
            "HTFormPtsStr": "".join(home["Form"]), "ATFormPtsStr": "".join(away["Form"]),
            "HTFormPts": ht_form_pts, "ATFormPts": at_form_pts,
            "HTGD": htgd, "ATGD": atgd,
            "DiffPts": home["Points"] - away["Points"],
            "DiffFormPts": ht_form_pts - at_form_pts,
        }
        new_features.append({f"f_{k}": v for k, v in row_stats.items()})

        # update post-match stats
        home["GoalsScored"] += fthg
        home["GoalsConceded"] += ftag
        away["GoalsScored"] += ftag
        away["GoalsConceded"] += fthg

        if ftr == "H":
            home["Points"] += 3
            home["Form"] = (["W"] + home["Form"])[:5]
            away["Form"] = (["L"] + away["Form"])[:5]
        elif ftr == "A":
            away["Points"] += 3
            home["Form"] = (["L"] + home["Form"])[:5]
            away["Form"] = (["W"] + away["Form"])[:5]
        else:
            home["Points"] += 1
            away["Points"] += 1
            home["Form"] = (["D"] + home["Form"])[:5]
            away["Form"] = (["D"] + away["Form"])[:5]

        matchweek += 1
        last_date = date

    feat_df = pd.DataFrame(new_features, index=df.index)
    return pd.concat([df, feat_df], axis=1).copy()


# ---------------------------------------------------------------------------
# Elo + Pi ratings
# ---------------------------------------------------------------------------

def add_team_elo(df: pd.DataFrame, data_dir: str | os.PathLike) -> pd.DataFrame:
    """Join in pre-computed team Elo ratings from ``team_elo.csv``."""
    df_elo = pd.read_csv(Path(data_dir) / "team_elo.csv")
    for elo_column in [c for c in df_elo.columns if c != "Team"]:
        elo_dict = df_elo.set_index("Team")[elo_column].to_dict()
        df[f"f_elo_{elo_column}_home"] = df["HomeTeam"].map(elo_dict)
        df[f"f_elo_{elo_column}_away"] = df["AwayTeam"].map(elo_dict)
    return df.fillna(0).copy()


def _exp_goal_diff(c: float, hr: float, ar: float) -> float:
    egda = (10 ** np.abs(ar / c) - 1) * (1 if ar >= 0 else -1)
    egdh = (10 ** np.abs(hr / c) - 1) * (1 if hr >= 0 else -1)
    return egdh - egda


def get_pi_ratings(df: pd.DataFrame, params: tuple[float, float, float]) -> pd.DataFrame:
    """Compute home/away Pi ratings for every match (Constantinou & Fenton)."""
    teams = list(set(list(df["HomeTeam"]) + list(df["AwayTeam"])))
    pi: dict[str, float] = {f"Home {t}": 0.0 for t in teams}
    pi.update({f"Away {t}": 0.0 for t in teams})

    c, mu1, mu2 = params
    results: list[dict] = []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        hgs, ags = row["GF_home"], row["GF_away"]

        h_hr = pi[f"Home {home}"]
        h_ar = pi[f"Away {home}"]
        a_hr = pi[f"Home {away}"]
        a_ar = pi[f"Away {away}"]

        egd = _exp_goal_diff(c, h_hr, a_ar)
        obs_goals = hgs - ags
        error = np.abs(obs_goals - egd)

        if egd < obs_goals:
            we_home = c * np.log10(1 + error)
        else:
            we_home = -c * np.log10(1 + error)
        we_away = -we_home

        h_hr_new = h_hr + we_home * mu1
        h_ar_new = h_hr + (h_hr_new - h_hr) * mu2
        a_ar_new = a_ar + we_away * mu1
        a_hr_new = a_hr + (a_ar_new - a_ar) * mu2

        pi[f"Home {home}"] = h_hr_new
        pi[f"Away {home}"] = h_ar_new
        pi[f"Home {away}"] = a_hr_new
        pi[f"Away {away}"] = a_ar_new

        results.append({
            "Date": row["Date"], "HomeTeam": home, "AwayTeam": away,
            "HomeGoals": hgs, "AwayGoals": ags,
            "f_pi_Home Rating_home": h_hr,
            "f_pi_Away Rating_home": h_ar,
            "f_pi_Home Rating_away": a_hr,
            "f_pi_Away Rating_away": a_ar,
            "f_pi_Exp GD Pi": egd,
            "f_pi_Pi Diff": h_hr - a_ar,
        })

    return pd.DataFrame(results)


def add_pi_rating(df: pd.DataFrame, params: tuple[float, float, float] = (1.0, 0.1, 0.3)) -> pd.DataFrame:
    df_ratings = get_pi_ratings(df, params)
    return pd.merge(df, df_ratings, on=["Date", "HomeTeam", "AwayTeam"]).copy()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def transform_raw_data(raw_data: pd.DataFrame, data_dir: str | os.PathLike, verbose: bool = False) -> pd.DataFrame:
    """Run the full feature-engineering pipeline (3.1.1 → 3.1.5)."""
    from .data import clean_data

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    df = clean_data(raw_data, verbose=verbose).dropna()
    _log(f"cleaned: {df.shape}")
    df = add_performance_indicators(df)
    _log(f"performance indicators: {df.shape}")
    df = create_ema_features(df, span=30)
    _log(f"EMA: {df.shape}")
    df = restructure_data(df)
    _log(f"restructured: {df.shape}")
    df = add_seasonal_features(df)
    _log(f"seasonal: {df.shape}")
    df = add_team_elo(df, data_dir).dropna(how="all")
    _log(f"elo: {df.shape}")
    df = add_pi_rating(df).dropna(how="all")
    _log(f"pi: {df.shape}")
    return df.sort_values(by="Date").reset_index(drop=True)
