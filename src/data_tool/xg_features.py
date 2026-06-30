"""
xG features -- the core analytics signal (chance quality). xG out-predicts raw
goals for match outcome.

Per La Liga team-match, from real xG only (season 22/23-25/26 AND value > 0;
pre-22/23 stores 0 = missing, never fillna):
  - xG-for     = the team's own expected_goals
  - xG-against = the opponent's expected_goals (self-join on match_id + opp_team_id)
Both get the standard T-1 DWA rolling (shift(1) -> rolling(5) weights
[0.35,0.25,0.20,0.10,0.10]); shared NaN-robust rolling helper in rolling.py.

Emits three home-minus-away differentials joined to the feature store:
  - xG_Diff    = home rolling xG-for  - away rolling xG-for      (attack; DWA_Goal_Diff replacement)
  - xGA_Diff   = home rolling xG-against - away rolling xG-against (defence: lower home = stingier home)
  - netxG_Diff = (home for-against) - (away for-against)          (expected goal-supremacy from xG)

Usage:
    python -m src.data_tool.xg_features              # build + attach to feature store
    python -m src.data_tool.xg_features --selfcheck
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_tool.features import load_name_mapping, standardize_team_column
from src.data_tool.rolling import weighted_recent_average_nan

DEFAULT_TEAM_STATS = Path("data/raw/thestatsapi/team_match_stats.csv")
DEFAULT_MAPPING = Path("src/utils/mapping.json")
DEFAULT_FEATURE_STORE = Path("data/feature_store/final_features.csv")
LALIGA = "La Liga"
XG_SEASONS = {"22/23", "23/24", "24/25", "25/26"}  # pre-22/23 returns xG=0 (missing)
# per-side rolling xG (the goals-engine inputs) + the home-minus-away diffs
XG_SIDE = ["Home_xGf", "Home_xGa", "Away_xGf", "Away_xGa"]
XG_DIFFS = ["xG_Diff", "xGA_Diff", "netxG_Diff"]
XG_COLS = XG_SIDE + XG_DIFFS


def _t1_dwa_roll(df: pd.DataFrame, col: str, out: str) -> None:
    """In-place T-1 DWA rolling of `col` per Team (NaN-robust); writes `out`."""
    shifted = df.groupby("Team", group_keys=False)[col].shift(1)
    df[out] = (
        shifted.groupby(df["Team"], group_keys=False)
        .rolling(window=5, min_periods=1)
        .apply(weighted_recent_average_nan, raw=True)
        .reset_index(level=0, drop=True)
    )


def compute_team_xg(
    team_stats_path: Path = DEFAULT_TEAM_STATS,
    mapping_path: Path = DEFAULT_MAPPING,
) -> pd.DataFrame:
    """Pre-match T-1 rolling xG-for AND xG-against per La Liga team-match (clean-frame only)."""
    df = pd.read_csv(team_stats_path, low_memory=False)
    df = df[df["comp_name"] == LALIGA].copy()

    # xG-against = opponent's expected_goals in the same match (self-join)
    opp = df[["match_id", "team_id", "expected_goals"]].rename(
        columns={"team_id": "opp_team_id", "expected_goals": "opp_xg"}
    )
    df = df.merge(opp, on=["match_id", "opp_team_id"], how="left")

    real = df["season"].isin(XG_SEASONS)
    df["xG_for"] = df["expected_goals"].where(real & (df["expected_goals"] > 0))
    df["xG_against"] = df["opp_xg"].where(real & (df["opp_xg"] > 0))

    df["Team"] = standardize_team_column(df["team_name"], load_name_mapping(mapping_path))
    df["Date"] = (
        pd.to_datetime(df["utc_date"], errors="coerce", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    df = df.dropna(subset=["Date", "Team"]).copy()

    df = df.sort_values(["Team", "Date", "match_id"]).reset_index(drop=True)
    _t1_dwa_roll(df, "xG_for", "DWA_xG_for")
    _t1_dwa_roll(df, "xG_against", "DWA_xG_against")
    return df[["Date", "Team", "xG_for", "xG_against", "DWA_xG_for", "DWA_xG_against"]]


def attach_xg_features(
    feature_store_path: Path = DEFAULT_FEATURE_STORE,
    team_stats_path: Path = DEFAULT_TEAM_STATS,
    mapping_path: Path = DEFAULT_MAPPING,
) -> pd.DataFrame:
    fs = pd.read_csv(feature_store_path, low_memory=False)
    fs["Date"] = pd.to_datetime(fs["Date"], errors="coerce")
    fs = fs.drop(columns=XG_COLS + ["xG_Sum"], errors="ignore")  # idempotent re-attach

    team = compute_team_xg(team_stats_path, mapping_path)[
        ["Date", "Team", "DWA_xG_for", "DWA_xG_against"]
    ]
    home = team.rename(columns={"Team": "HomeTeam", "DWA_xG_for": "Home_xGf", "DWA_xG_against": "Home_xGa"})
    away = team.rename(columns={"Team": "AwayTeam", "DWA_xG_for": "Away_xGf", "DWA_xG_against": "Away_xGa"})

    out = fs.merge(home, on=["Date", "HomeTeam"], how="left").merge(away, on=["Date", "AwayTeam"], how="left")
    # diffs (1X2 features) + KEEP the four per-side rolling xG values (goals-engine inputs)
    out["xG_Diff"] = out["Home_xGf"] - out["Away_xGf"]
    out["xGA_Diff"] = out["Home_xGa"] - out["Away_xGa"]
    out["netxG_Diff"] = (out["Home_xGf"] - out["Home_xGa"]) - (out["Away_xGf"] - out["Away_xGa"])
    return out


def selfcheck(
    team_stats_path: Path = DEFAULT_TEAM_STATS,
    mapping_path: Path = DEFAULT_MAPPING,
    feature_store_path: Path = DEFAULT_FEATURE_STORE,
) -> None:
    team = compute_team_xg(team_stats_path, mapping_path)
    for col in ("xG_for", "xG_against"):
        present = team[col].dropna()
        assert (present > 0).all(), f"{col} has non-positive values"
        assert 0.3 < present.median() < 4.0, f"{col} median {present.median():.2f} out of range"
        print(f"selfcheck: {col} median {present.median():.2f}, range {present.min():.2f}-{present.max():.2f} ({len(present)})")

    first = team.sort_values(["Team", "Date"]).groupby("Team").head(1)
    assert first["DWA_xG_for"].isna().all() and first["DWA_xG_against"].isna().all(), "team-debut rolling must be NaN (T-1 leak!)"

    out = attach_xg_features(feature_store_path, team_stats_path, mapping_path)
    pre = out[out["Date"] < pd.Timestamp("2022-07-01")]
    xg_era = out[out["Date"] >= pd.Timestamp("2022-08-01")]
    for col in XG_COLS:
        assert pre[col].isna().all(), f"pre-22/23 {col} must be NaN (no xG), not filled"
    cov = xg_era["netxG_Diff"].notna().mean()
    assert cov > 0.9, f"xG-era coverage too low: {cov:.3f}"
    print(f"selfcheck: xG-era coverage {cov*100:.1f}% (xG_Diff/xGA_Diff/netxG_Diff), pre-22/23 all-NaN (no fillna)")
    print("selfcheck PASSED")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build + attach xG_Diff / xGA_Diff / netxG_Diff to the feature store.")
    ap.add_argument("--team-stats", type=Path, default=DEFAULT_TEAM_STATS)
    ap.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    ap.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck(args.team_stats, args.mapping, args.feature_store)
        return
    out = attach_xg_features(args.feature_store, args.team_stats, args.mapping)
    xg_era = out[out["Date"] >= pd.Timestamp("2022-08-01")]
    print(f"Feature store rows: {len(out)} | xG-era coverage {xg_era['netxG_Diff'].notna().mean()*100:.1f}% "
          f"({', '.join(XG_COLS)})")
    out.to_csv(args.feature_store, index=False)
    print(f"Attached {', '.join(XG_COLS)} -> {args.feature_store}")


if __name__ == "__main__":
    main()
