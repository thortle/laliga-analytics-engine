"""
Explanatory dataset -- one row per completed La Liga match,
each in-match team stat expressed as a home-minus-away differential, joined to the
1X2 outcome. This is the input for ranking *which stats decided who won* (no
leakage concern: we explain a finished match using its own stats; T-1 rolling for
the predictive layer is separate).

Source: data/raw/thestatsapi/team_match_stats.csv (La Liga, 2 rows/match). Self-
contained -- the 1X2 label comes from `goals`, so no football-data / mapping join.

Hygiene (verified on the real CSV):
  - 5 duplicate-twin pairs are 100% identical -> keep one, drop the redundant twin.
  - np_expected_goals is CORRUPT (24.6% of in-era rows have npxG > xG) -> dropped.
  - xG family (expected_goals, big_chances, att_big_chances_missed) is populated
    22/23+ only; pre-era it is 0 = MISSING. Gated to NaN pre-era -- never fillna
    (no imputation poisoning). For expected_goals, also require xG>0 on BOTH sides.

Output: data/processed/explanatory_match_stats.csv
  keys/meta: match_id, season, Date, home_team, away_team
  outcome:   home_goals, away_goals, FTR (H/D/A), HomeWin
  features:  <stat>_diff (home - away) for ~36 always-on stats + 3 xG-era stats.

Usage:
    python -m src.data_tool.explanatory_dataset            # build -> CSV
    python -m src.data_tool.explanatory_dataset --selfcheck
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TEAM_STATS = Path("data/raw/thestatsapi/team_match_stats.csv")
DEFAULT_OUTPUT = Path("data/processed/explanatory_match_stats.csv")
LALIGA = "La Liga"
XG_SEASONS = {"22/23", "23/24", "24/25", "25/26"}  # pre-era stores 0 = missing

# meta/key columns that are not stat candidates (goals -> outcome, handled apart)
NON_STAT = {
    "match_id", "comp_name", "season", "utc_date", "is_home",
    "team_id", "team_name", "opp_team_id", "goals",
}
# 100%-identical twins (keep the first, drop the second) + corrupt npxG
DROP_REDUNDANT = [
    "sh_total_shots", "sh_shots_on_target", "pass_accurate_passes",
    "def_tackles", "gk_saves", "np_expected_goals",
]
# xG-family stats: real only 22/23+ (gate to NaN pre-era, never fillna)
XG_GATED = ["expected_goals", "big_chances", "att_big_chances_missed"]


def build(team_stats_path: Path = DEFAULT_TEAM_STATS) -> pd.DataFrame:
    df = pd.read_csv(team_stats_path, low_memory=False)
    df = df[df["comp_name"] == LALIGA].drop(columns=DROP_REDUNDANT, errors="ignore").copy()
    df["Date"] = pd.to_datetime(df["utc_date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()

    stat_cols = [c for c in df.columns if c not in NON_STAT and c != "Date"]

    # pivot the 2 team-rows per match into home/away on match_id
    keep = ["match_id", "season", "Date", "team_name", "goals"] + stat_cols
    home = df[df["is_home"]][keep].add_prefix("home_")
    away = df[~df["is_home"]][keep].add_prefix("away_")
    m = home.merge(away, left_on="home_match_id", right_on="away_match_id", how="inner")

    out = pd.DataFrame({
        "match_id": m["home_match_id"],
        "season": m["home_season"],
        "Date": m["home_Date"],
        "home_team": m["home_team_name"],
        "away_team": m["away_team_name"],
        "home_goals": m["home_goals"],
        "away_goals": m["away_goals"],
    })
    out["FTR"] = np.where(out.home_goals > out.away_goals, "H",
                  np.where(out.home_goals < out.away_goals, "A", "D"))
    out["HomeWin"] = (out.FTR == "H").astype(int)

    in_era = out["season"].isin(XG_SEASONS)
    for s in stat_cols:
        diff = m[f"home_{s}"] - m[f"away_{s}"]
        if s == "expected_goals":          # require real xG on BOTH sides
            diff = diff.where(in_era & (m["home_expected_goals"] > 0) & (m["away_expected_goals"] > 0))
        elif s in XG_GATED:                # 0 is genuine in-era; gate only by season
            diff = diff.where(in_era)
        out[f"{s}_diff"] = diff.to_numpy()

    return out.sort_values("Date").reset_index(drop=True)


def selfcheck(team_stats_path: Path = DEFAULT_TEAM_STATS) -> None:
    out = build(team_stats_path)
    assert len(out) == 2280, f"expected 2280 matches, got {len(out)}"
    assert out["match_id"].is_unique, "duplicate match_id"
    # outcome sanity: home advantage means H is the plurality class
    ftr = out["FTR"].value_counts(normalize=True)
    assert ftr["H"] > ftr["A"] > 0.2 and 0.2 < ftr["D"] < 0.3, f"FTR distribution off: {ftr.to_dict()}"

    # no fillna: xG diffs NaN pre-era, present in-era
    pre = out[~out.season.isin(XG_SEASONS)]
    era = out[out.season.isin(XG_SEASONS)]
    for c in ("expected_goals_diff", "big_chances_diff", "att_big_chances_missed_diff"):
        assert pre[c].isna().all(), f"{c} must be all-NaN pre-22/23 (no fillna), found values"
        assert era[c].notna().mean() > 0.99, f"{c} in-era coverage too low: {era[c].notna().mean():.3f}"

    # non-xG diffs are fully populated (no fillna needed, none introduced)
    nonxg = [c for c in out.columns if c.endswith("_diff") and not any(c.startswith(g) for g in XG_GATED)]
    for c in nonxg:
        assert out[c].notna().all(), f"{c} has unexpected NaN"

    # spot-check one diff is genuinely home - away, computed from source
    src = pd.read_csv(team_stats_path, low_memory=False)
    src = src[src.comp_name == LALIGA]
    mid = out.iloc[1000]["match_id"]
    h = src[(src.match_id == mid) & (src.is_home)]["total_shots"].iloc[0]
    a = src[(src.match_id == mid) & (~src.is_home)]["total_shots"].iloc[0]
    assert np.isclose(out.set_index("match_id").loc[mid, "total_shots_diff"], h - a), "diff != home-away"

    nfeat = sum(c.endswith("_diff") for c in out.columns)
    print(f"selfcheck PASSED: {len(out)} matches, {nfeat} diff features "
          f"({len(XG_GATED)} xG-gated), FTR {ftr.round(3).to_dict()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the explanatory match-stats dataset.")
    ap.add_argument("--team-stats", type=Path, default=DEFAULT_TEAM_STATS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck(args.team_stats)
        return
    out = build(args.team_stats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    nfeat = sum(c.endswith("_diff") for c in out.columns)
    print(f"Wrote {len(out)} matches x {nfeat} diff features -> {args.out}")


if __name__ == "__main__":
    main()
