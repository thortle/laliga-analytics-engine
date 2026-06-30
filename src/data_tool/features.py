from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple
import unicodedata

import numpy as np
import pandas as pd

# Builds the leakage-safe Elo/DWA feature store: Elo_Diff + DWA_Goal_Diff +
# DWA_SOT_Diff from data/processed/merged_matches.csv (T-1, no leakage). This is the
# results-based rating spine of the ANALYTICS engine; rich API stats (xG_Diff, ...)
# are attached by separate scripts (src/data_tool/xg_features.py). The prior betting
# chapter concluded a rigorous null result (not included in this public repo).

DEFAULT_INPUT = Path("data/processed/merged_matches.csv")
DEFAULT_OUTPUT = Path("data/feature_store/final_features.csv")
DEFAULT_LEAKAGE_REPORT = Path("data/feature_store/leakage_report.txt")

DWA_WEIGHTS = np.array([0.35, 0.25, 0.20, 0.10, 0.10], dtype=float)


def normalize_name(name: str) -> str:
    text = str(name).strip().lower()
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )
    text = text.replace(".", " ").replace("-", " ").replace("_", " ")
    return " ".join(text.split())


def load_name_mapping(mapping_path: Path) -> Dict[str, str]:
    with mapping_path.open("r", encoding="utf-8") as f:
        raw_mapping = json.load(f)
    return {normalize_name(alias): canonical for alias, canonical in raw_mapping.items()}


def standardize_team_column(series: pd.Series, mapping: Dict[str, str]) -> pd.Series:
    normalized = series.astype(str).map(normalize_name)
    return normalized.map(lambda team: mapping.get(team, team.replace(" ", "_")))


def weighted_recent_average(window_values: Iterable[float]) -> float:
    values = np.asarray(window_values, dtype=float)
    if values.size == 0:
        return np.nan

    active_weights = DWA_WEIGHTS[: values.size]
    return float(np.dot(values[::-1], active_weights) / active_weights.sum())


def load_base_matches(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)

    if "MatchDateTime" in df.columns:
        df["MatchDateTime"] = pd.to_datetime(df["MatchDateTime"], errors="coerce")
    else:
        df["MatchDateTime"] = pd.NaT

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "KickoffTime" in df.columns:
        kickoff = df["KickoffTime"].astype("string").fillna("00:00")
        rebuilt = pd.to_datetime(df["Date"].dt.strftime("%Y-%m-%d") + " " + kickoff, errors="coerce")
        df["MatchDateTime"] = df["MatchDateTime"].fillna(rebuilt)
    df["MatchDateTime"] = df["MatchDateTime"].fillna(df["Date"])

    numeric_cols = ["FTHG", "FTAG", "HST", "AST"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HST", "AST"]).copy()
    df = df.sort_values(["MatchDateTime", "SeasonFile", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    df["MatchIdx"] = np.arange(len(df), dtype=int)
    df["HomeWin"] = (df["FTHG"] > df["FTAG"]).astype(int)

    return df


def compute_elo_diff(df: pd.DataFrame, k: float = 32.0, hfa: float = 100.0, start: float = 1500.0) -> pd.DataFrame:
    ratings: Dict[str, float] = defaultdict(lambda: start)
    elo_home_pre = np.zeros(len(df), dtype=float)
    elo_away_pre = np.zeros(len(df), dtype=float)

    for i, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        home_pre = ratings[home]
        away_pre = ratings[away]

        elo_home_pre[i] = home_pre
        elo_away_pre[i] = away_pre

        exp_home = 1.0 / (1.0 + 10.0 ** ((away_pre - (home_pre + hfa)) / 400.0))
        if row["FTHG"] > row["FTAG"]:
            act_home = 1.0
        elif row["FTHG"] < row["FTAG"]:
            act_home = 0.0
        else:
            act_home = 0.5

        delta = k * (act_home - exp_home)
        ratings[home] = home_pre + delta
        ratings[away] = away_pre - delta

    elo_df = pd.DataFrame(
        {
            "MatchIdx": df["MatchIdx"],
            "Elo_Home_Pre": elo_home_pre,
            "Elo_Away_Pre": elo_away_pre,
            "Elo_Diff": elo_home_pre - elo_away_pre,
        }
    )
    return elo_df


def reshape_matches_to_team_events(df: pd.DataFrame) -> pd.DataFrame:
    home = df[["MatchIdx", "Date", "MatchDateTime", "HomeTeam", "FTHG", "HST"]].copy()
    home["Side"] = "H"
    home = home.rename(columns={"HomeTeam": "Team", "FTHG": "GoalsFor", "HST": "SOTFor"})

    away = df[["MatchIdx", "Date", "MatchDateTime", "AwayTeam", "FTAG", "AST"]].copy()
    away["Side"] = "A"
    away = away.rename(columns={"AwayTeam": "Team", "FTAG": "GoalsFor", "AST": "SOTFor"})

    team_stats = pd.concat([home, away], ignore_index=True)
    team_stats = team_stats.sort_values(["Team", "MatchDateTime", "MatchIdx", "Side"]).reset_index(drop=True)

    for metric in ["GoalsFor", "SOTFor"]:
        shifted = team_stats.groupby("Team", group_keys=False)[metric].shift(1)
        team_stats[f"DWA_{metric}"] = shifted.groupby(team_stats["Team"], group_keys=False).rolling(
            window=5,
            min_periods=1,
        ).apply(weighted_recent_average, raw=True).reset_index(level=0, drop=True)

    return team_stats


def compute_dwa_goal_sot_diff(df: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    home = team_stats[team_stats["Side"] == "H"][
        ["MatchIdx", "DWA_GoalsFor", "DWA_SOTFor"]
    ].rename(
        columns={
            "DWA_GoalsFor": "DWA_Home_Goals",
            "DWA_SOTFor": "DWA_Home_SOT",
        }
    )
    away = team_stats[team_stats["Side"] == "A"][
        ["MatchIdx", "DWA_GoalsFor", "DWA_SOTFor"]
    ].rename(
        columns={
            "DWA_GoalsFor": "DWA_Away_Goals",
            "DWA_SOTFor": "DWA_Away_SOT",
        }
    )

    out = df.merge(home, on="MatchIdx", how="left").merge(away, on="MatchIdx", how="left")
    out["DWA_Goal_Diff"] = out["DWA_Home_Goals"] - out["DWA_Away_Goals"]
    out["DWA_SOT_Diff"] = out["DWA_Home_SOT"] - out["DWA_Away_SOT"]
    return out


def merge_all_features_to_matches(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "Elo_Diff",
        "DWA_Goal_Diff",
        "DWA_SOT_Diff",
        "HomeWin",
    ]
    feature_store = df[cols].copy()
    return feature_store


def leakage_check_single(team_history: pd.DataFrame, match_idx: int, observed_dwa: float, metric: str) -> Tuple[bool, float]:
    prior = team_history[team_history["MatchIdx"] < match_idx].sort_values("MatchIdx")[metric].tail(5).to_numpy()
    expected = weighted_recent_average(prior) if prior.size > 0 else np.nan

    if np.isnan(expected) and np.isnan(observed_dwa):
        return True, expected
    if np.isnan(expected) != np.isnan(observed_dwa):
        return False, expected

    return bool(np.isclose(expected, observed_dwa, atol=1e-9)), expected


def generate_leakage_report(df: pd.DataFrame, team_stats: pd.DataFrame, output_path: Path, sample_size: int = 5) -> None:
    rng = np.random.default_rng(42)
    sample_idx = np.sort(rng.choice(df["MatchIdx"].to_numpy(), size=min(sample_size, len(df)), replace=False))

    team_hist = team_stats[["MatchIdx", "Team", "GoalsFor", "DWA_GoalsFor"]].copy()
    report_lines = [
        "Leakage Validation Report",
        "Rule: DWA for Match T must use only matches T-1..T-5.",
        "",
    ]

    all_pass = True
    for match_idx in sample_idx:
        row = df.loc[df["MatchIdx"] == int(match_idx)].iloc[0]
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]

        home_history = team_hist[team_hist["Team"] == home_team]
        away_history = team_hist[team_hist["Team"] == away_team]

        home_obs = row["DWA_Home_Goals"]
        away_obs = row["DWA_Away_Goals"]

        home_ok, home_exp = leakage_check_single(home_history, int(match_idx), float(home_obs), "GoalsFor")
        away_ok, away_exp = leakage_check_single(away_history, int(match_idx), float(away_obs), "GoalsFor")

        row_pass = home_ok and away_ok
        all_pass = all_pass and row_pass

        report_lines.append(
            f"MatchIdx={int(match_idx)} Date={row['Date'].date()} {home_team} vs {away_team} -> "
            f"HOME_OK={home_ok} (obs={home_obs:.6f}, exp={home_exp if pd.isna(home_exp) else round(float(home_exp), 6)}), "
            f"AWAY_OK={away_ok} (obs={away_obs:.6f}, exp={away_exp if pd.isna(away_exp) else round(float(away_exp), 6)})"
        )

    report_lines.append("")
    report_lines.append(f"Overall leakage check status: {'PASS' if all_pass else 'FAIL'}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def print_team_progression(df: pd.DataFrame, team: str = "real_madrid") -> None:
    home = df[df["HomeTeam"] == team][
        ["Date", "HomeTeam", "AwayTeam", "FTHG", "DWA_Home_Goals"]
    ].rename(
        columns={
            "HomeTeam": "Team",
            "AwayTeam": "Opponent",
            "FTHG": "CurrentGoals",
            "DWA_Home_Goals": "DWA_Goal_Feature",
        }
    )

    away = df[df["AwayTeam"] == team][
        ["Date", "AwayTeam", "HomeTeam", "FTAG", "DWA_Away_Goals"]
    ].rename(
        columns={
            "AwayTeam": "Team",
            "HomeTeam": "Opponent",
            "FTAG": "CurrentGoals",
            "DWA_Away_Goals": "DWA_Goal_Feature",
        }
    )

    team_matches = pd.concat([home, away], ignore_index=True)
    team_matches = team_matches[["Date", "Team", "Opponent", "CurrentGoals", "DWA_Goal_Feature"]]
    team_matches = team_matches.sort_values("Date").reset_index(drop=True)

    year_2024 = team_matches[team_matches["Date"].dt.year == 2024]
    if len(year_2024) >= 3:
        sample = year_2024.head(3)
    else:
        sample = team_matches.tail(3)

    print("\nLeakage demonstration for team progression:")
    print(sample.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the leakage-safe baseline (Elo/DWA) feature store.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--leakage-report", type=Path, default=DEFAULT_LEAKAGE_REPORT)
    parser.add_argument("--demo-team", type=str, default="real_madrid")

    args = parser.parse_args()

    base = load_base_matches(args.input)
    elo = compute_elo_diff(base)
    with_elo = base.merge(elo, on="MatchIdx", how="left")

    team_stats = reshape_matches_to_team_events(with_elo)
    with_dwa = compute_dwa_goal_sot_diff(with_elo, team_stats)

    feature_store = merge_all_features_to_matches(with_dwa)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    feature_store.to_csv(args.output, index=False)

    generate_leakage_report(with_dwa, team_stats, args.leakage_report, sample_size=5)

    print(f"Feature store rows: {len(feature_store)}")
    print(f"Saved feature store: {args.output}")
    print(f"Saved leakage report: {args.leakage_report}")

    print_team_progression(with_dwa, args.demo_team)


if __name__ == "__main__":
    main()
