"""
Predictive spine -- one row per La Liga match with T-1 (pre-match)
features only, built SOLELY from TheStatsAPI (single-source rule). This is the input
for an honestly-evaluated 1X2 predictor; strict zero-leakage: every
feature for a match on date T uses only information knowable BEFORE T.

Single-source: Elo is rebuilt from API results (football-data retired as a model input;
the API has complete scores+goals+SOT for all 6 seasons incl. 25/26, which football-data
lacks). Keyed by team_id -- no name mapping. Reuses the DWA rolling machinery
(shift(1) -> rolling(5) weights [0.35,0.25,0.20,0.10,0.10], NaN-robust) from rolling.py,
and Elo (k=32, hfa=100, start=1500) from features.py.

Features (all home-minus-away diffs, T-1):
  - Elo_Diff           pre-match Elo rating gap (results-based strength)
  - DWA_Goal_Diff      rolling goals-for                 [PARTIAL_CIRCULAR, rolling = clean]
  - DWA_SOT_Diff       rolling shots-on-target-for       [PARTIAL_CIRCULAR, rolling = clean]
  - xG_Diff            rolling xG-for     (22/23+)        [CLEAN driver -- THE anchor]
  - xGA_Diff           rolling xG-against (22/23+)
  - netxG_Diff         rolling (for-against) supremacy    (22/23+)
  - TotalShots_Diff    rolling total-shots tendency       [test candidate, expect ~null]
  - Crosses_Diff       rolling accurate-crosses tendency  [test candidate, expect negative]
  - Poss_Diff          rolling possession tendency        [test candidate, expect null-to-neg]

Outcome: FTR (H/D/A) from API goals.

Usage:
    python -m src.data_tool.predictive_dataset            # build -> CSV
    python -m src.data_tool.predictive_dataset --selfcheck
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_tool.rolling import weighted_recent_average_nan

DEFAULT_TEAM_STATS = Path("data/raw/thestatsapi/team_match_stats.csv")
DEFAULT_OUT = Path("data/processed/predictive_features.csv")
LALIGA = "La Liga"
XG_SEASONS = {"22/23", "23/24", "24/25", "25/26"}  # pre-era stores xG=0 = missing

# source col -> rolled feature stem (for-side, per team, T-1 DWA). xG-against is added
# separately via an opponent self-join. xG stats gated to the xG era (NaN before).
ROLL_FOR = {
    "goals": "Goal", "shots_on_target": "SOT", "expected_goals": "xGf",
    "total_shots": "TotalShots", "pass_accurate_crosses": "Crosses", "ball_possession": "Poss",
}
XG_STEMS = {"xGf", "xGa"}  # 22/23+ only


def _t1_dwa(df: pd.DataFrame, col: str, out: str) -> None:
    """In-place T-1 DWA rolling of `col` per team_id (NaN-robust); writes `out`. shift(1)
    makes it strictly pre-match (a team's first match -> NaN, no leakage). 5-match
    recency-weighted = a RECENT-FORM signal."""
    shifted = df.groupby("team_id", group_keys=False)[col].shift(1)
    df[out] = (
        shifted.groupby(df["team_id"], group_keys=False)
        .rolling(window=5, min_periods=1)
        .apply(weighted_recent_average_nan, raw=True)
        .reset_index(level=0, drop=True)
    )


def _t1_longmean(df: pd.DataFrame, col: str, out: str, win: int = 10) -> None:
    """In-place T-1 plain rolling mean over a LONG window (carries across seasons) = a
    STRENGTH signal (stable, low-noise) -- the principled horizon for xG-as-strength.
    Prior evaluation showed the 5-match form window is too noisy for xG; a long window
    is near-Elo-strong standalone. min_periods=3 so early history stays honestly NaN."""
    shifted = df.groupby("team_id", group_keys=False)[col].shift(1)
    df[out] = (
        shifted.groupby(df["team_id"], group_keys=False)
        .rolling(window=win, min_periods=3).mean()
        .reset_index(level=0, drop=True)
    )


def compute_elo(matches: pd.DataFrame, k: float = 32.0, hfa: float = 100.0,
                start: float = 1500.0) -> pd.DataFrame:
    """Pre-match Elo per match from API results (chronological). Returns Elo_Diff keyed by match_id."""
    ratings: dict[str, float] = defaultdict(lambda: start)
    rows = []
    for r in matches.itertuples(index=False):
        hp, ap = ratings[r.home_team_id], ratings[r.away_team_id]
        rows.append((r.match_id, hp - ap))                       # pre-match gap = the feature
        exp_h = 1.0 / (1.0 + 10.0 ** ((ap - (hp + hfa)) / 400.0))
        act_h = 1.0 if r.home_goals > r.away_goals else (0.0 if r.home_goals < r.away_goals else 0.5)
        delta = k * (act_h - exp_h)
        ratings[r.home_team_id] = hp + delta
        ratings[r.away_team_id] = ap - delta
    return pd.DataFrame(rows, columns=["match_id", "Elo_Diff"])


def build(team_stats_path: Path = DEFAULT_TEAM_STATS) -> pd.DataFrame:
    df = pd.read_csv(team_stats_path, low_memory=False)
    df = df[df["comp_name"] == LALIGA].copy()
    df["Date"] = pd.to_datetime(df["utc_date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()

    # xG-against = opponent's expected_goals in the same match (self-join), gated to xG era
    opp = df[["match_id", "team_id", "expected_goals"]].rename(
        columns={"team_id": "opp_team_id", "expected_goals": "opp_xg"})
    df = df.merge(opp, on=["match_id", "opp_team_id"], how="left")
    in_era = df["season"].isin(XG_SEASONS)
    df["expected_goals"] = df["expected_goals"].where(in_era & (df["expected_goals"] > 0))
    df["xGa_src"] = df["opp_xg"].where(in_era & (df["opp_xg"] > 0))

    # per-team T-1 rolling (chronological per team)
    df = df.sort_values(["team_id", "Date", "match_id"]).reset_index(drop=True)
    for src, stem in ROLL_FOR.items():
        _t1_dwa(df, src, f"DWA_{stem}")                 # 5-match recent form
    _t1_dwa(df, "xGa_src", "DWA_xGa")
    _t1_longmean(df, "expected_goals", "LXG_xGf")        # long-horizon xG STRENGTH
    _t1_longmean(df, "xGa_src", "LXG_xGa")

    rolled = [f"DWA_{s}" for s in ROLL_FOR.values()] + ["DWA_xGa", "LXG_xGf", "LXG_xGa"]
    home = df[df["is_home"]][["match_id", "season", "Date", "team_id", "goals"] + rolled].add_prefix("home_")
    away = df[~df["is_home"]][["match_id", "team_id", "goals"] + rolled].add_prefix("away_")
    m = home.merge(away, left_on="home_match_id", right_on="away_match_id", how="inner")
    m = m.rename(columns={"home_match_id": "match_id", "home_season": "season", "home_Date": "Date"})
    m = m.sort_values(["Date", "match_id"]).reset_index(drop=True)

    out = pd.DataFrame({
        "match_id": m["match_id"], "season": m["season"], "Date": m["Date"],
        "home_team_id": m["home_team_id"], "away_team_id": m["away_team_id"],
        "home_goals": m["home_goals"], "away_goals": m["away_goals"],
    })
    out["FTR"] = np.where(out.home_goals > out.away_goals, "H",
                  np.where(out.home_goals < out.away_goals, "A", "D"))

    out = out.merge(compute_elo(out), on="match_id", how="left")
    # home-minus-away T-1 diffs
    out["DWA_Goal_Diff"] = m["home_DWA_Goal"] - m["away_DWA_Goal"]
    out["DWA_SOT_Diff"] = m["home_DWA_SOT"] - m["away_DWA_SOT"]
    out["xG_Diff"] = m["home_DWA_xGf"] - m["away_DWA_xGf"]
    out["xGA_Diff"] = m["home_DWA_xGa"] - m["away_DWA_xGa"]
    out["netxG_Diff"] = (m["home_DWA_xGf"] - m["home_DWA_xGa"]) - (m["away_DWA_xGf"] - m["away_DWA_xGa"])
    out["TotalShots_Diff"] = m["home_DWA_TotalShots"] - m["away_DWA_TotalShots"]
    out["Crosses_Diff"] = m["home_DWA_Crosses"] - m["away_DWA_Crosses"]
    out["Poss_Diff"] = m["home_DWA_Poss"] - m["away_DWA_Poss"]
    # long-horizon xG STRENGTH diffs (the principled predictive xG signal)
    out["xGstr_Diff"] = m["home_LXG_xGf"] - m["away_LXG_xGf"]
    out["xGAstr_Diff"] = m["home_LXG_xGa"] - m["away_LXG_xGa"]
    out["netxGstr_Diff"] = (m["home_LXG_xGf"] - m["home_LXG_xGa"]) - (m["away_LXG_xGf"] - m["away_LXG_xGa"])
    return out


FEATURES = ["Elo_Diff", "DWA_Goal_Diff", "DWA_SOT_Diff",
            "xG_Diff", "xGA_Diff", "netxG_Diff",                 # 5-match xG FORM
            "xGstr_Diff", "xGAstr_Diff", "netxGstr_Diff",        # long-horizon xG STRENGTH
            "TotalShots_Diff", "Crosses_Diff", "Poss_Diff"]


def selfcheck(team_stats_path: Path = DEFAULT_TEAM_STATS) -> None:
    out = build(team_stats_path)
    assert len(out) == 2280 and out["match_id"].is_unique, f"expected 2280 unique matches, got {len(out)}"

    # LEAKAGE: a team's FIRST-ever match has no prior history, so its rolling side is NaN
    # -> the match-level DWA_Goal_Diff must contain NaN (proves the shift(1) T-1 guard fired).
    assert out["DWA_Goal_Diff"].isna().any(), "no NaN in DWA_Goal_Diff -- T-1 shift not applied?"

    # Elo: should be 0 only at the very first appearances and spread out later
    assert out["Elo_Diff"].abs().max() > 100, "Elo_Diff never separates -- Elo not updating?"
    assert (out["Elo_Diff"] == 0).mean() < 0.05, "too many zero Elo_Diff (ratings not diverging)"

    # xG features: NaN pre-era, populated in-era
    pre = out[~out.season.isin(XG_SEASONS)]
    era = out[out.season.isin(XG_SEASONS)]
    for c in ("xG_Diff", "xGA_Diff", "netxG_Diff", "xGstr_Diff", "xGAstr_Diff", "netxGstr_Diff"):
        assert pre[c].isna().all(), f"{c} must be all-NaN pre-22/23 (no fillna)"
    cov = era["xG_Diff"].notna().mean()
    assert cov > 0.85, f"xG-era rolling coverage too low: {cov:.3f}"

    # non-xG rolling features broadly populated after the first few weeks
    assert out["DWA_Goal_Diff"].notna().mean() > 0.9, "DWA_Goal_Diff coverage too low"
    ftr = out["FTR"].value_counts(normalize=True)
    assert ftr["H"] > ftr["A"], f"home advantage missing: {ftr.to_dict()}"
    print(f"selfcheck PASSED: 2280 matches, {len(FEATURES)} T-1 features, xG-era coverage {cov*100:.1f}%, "
          f"Elo_Diff range [{out.Elo_Diff.min():.0f}, {out.Elo_Diff.max():.0f}], FTR {ftr.round(3).to_dict()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the T-1 predictive spine (single-source, API only).")
    ap.add_argument("--team-stats", type=Path, default=DEFAULT_TEAM_STATS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck(args.team_stats)
        return
    out = build(args.team_stats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} matches x {len(FEATURES)} T-1 features -> {args.out}")


if __name__ == "__main__":
    main()
