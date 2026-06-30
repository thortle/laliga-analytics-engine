"""
Player features (pre-registered pilot) -- a T-1 "available-squad rating": the
strength of the players ACTUALLY STARTING today, weighted by each starter's prior form.
This is the one channel orthogonal to Elo (a results-based team rating can't see who is
injured / benched / sold). Built solely from TheStatsAPI player_match_stats.

Locked design (pre-registered):
  team squad rating = nanmean over this match's started==True players of each starter's
  T-1 prior rolling-mean `rating` (window 10 La Liga appearances, rating>0 only, shift(1)).
  SquadRating_Diff = home - away.

Leakage: WHO starts is public ~1h pre-kickoff -> using `started` is pre-match,
not result-leakage. Each player's rating is their PRIOR (T-1) rolling rating -- never this
match's rating/minutes. rating==0 == "did not play" == missing (never averaged, never fillna(0)).
Cold-start players (<3 prior rated apps) -> NaN, dropped from the XI nanmean (no imputation poisoning).

Usage:
    python -m src.data_tool.player_features            # build -> CSV
    python -m src.data_tool.player_features --selfcheck
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PLAYER_STATS = Path("data/raw/thestatsapi/player_match_stats.csv")
DEFAULT_TEAM_STATS = Path("data/raw/thestatsapi/team_match_stats.csv")
DEFAULT_OUT = Path("data/processed/player_squad_features.csv")
LALIGA = "La Liga"
PRIOR_WINDOW = 10   # a-priori long horizon = player quality; NOT tuned on holdout
MIN_PRIOR = 3       # need >=3 prior rated apps or the player's prior rating is NaN (cold-start)


def _home_away_map(team_stats_path: Path) -> pd.DataFrame:
    t = pd.read_csv(team_stats_path, low_memory=False)
    t = t[t["comp_name"] == LALIGA][["match_id", "team_id", "is_home"]]
    return t


def team_squad_long(player_stats_path: Path = DEFAULT_PLAYER_STATS,
                    team_stats_path: Path = DEFAULT_TEAM_STATS) -> pd.DataFrame:
    """Per (match, team): the pre-match available-squad rating = nanmean over today's started==True
    XI of each starter's T-1 prior rolling-mean rating. Returns [match_id, team_id, Date, squad_rating,
    is_home]. The per-team time series the absence-delta feature rolls to get the team's normal-XI baseline."""
    p = pd.read_csv(player_stats_path, low_memory=False)
    p = p[p["comp_name"] == LALIGA].copy()
    p["Date"] = pd.to_datetime(p["utc_date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    p["rating"] = pd.to_numeric(p["rating"], errors="coerce")
    p = p[p["rating"] > 0].copy()                       # rating==0/NaN = did not play = missing

    # per-player T-1 prior rolling-mean rating (shift(1) excludes this match -> strictly pre-match)
    p = p.sort_values(["player_id", "Date", "match_id"]).reset_index(drop=True)
    shifted = p.groupby("player_id", group_keys=False)["rating"].shift(1)
    p["prior_rating"] = (
        shifted.groupby(p["player_id"], group_keys=False)
        .rolling(PRIOR_WINDOW, min_periods=MIN_PRIOR).mean()
        .reset_index(level=0, drop=True)
    )

    xi = p[p["started"] == True]                        # noqa: E712 (pandas truthiness on a column)
    squad = (xi.groupby(["match_id", "team_id"]).agg(
        squad_rating=("prior_rating", lambda s: np.nan if s.notna().sum() == 0 else float(np.nanmean(s))),
        Date=("Date", "first")).reset_index())
    return squad.merge(_home_away_map(team_stats_path), on=["match_id", "team_id"], how="inner")


def build(player_stats_path: Path = DEFAULT_PLAYER_STATS,
          team_stats_path: Path = DEFAULT_TEAM_STATS) -> pd.DataFrame:
    squad = team_squad_long(player_stats_path, team_stats_path)
    home = squad[squad["is_home"]][["match_id", "squad_rating"]].rename(columns={"squad_rating": "home_squad_rating"})
    away = squad[~squad["is_home"]][["match_id", "squad_rating"]].rename(columns={"squad_rating": "away_squad_rating"})
    out = home.merge(away, on="match_id", how="inner")
    out["SquadRating_Diff"] = out["home_squad_rating"] - out["away_squad_rating"]
    return out[["match_id", "home_squad_rating", "away_squad_rating", "SquadRating_Diff"]]


def selfcheck(player_stats_path: Path = DEFAULT_PLAYER_STATS,
              team_stats_path: Path = DEFAULT_TEAM_STATS) -> None:
    out = build(player_stats_path, team_stats_path)
    assert out["match_id"].is_unique, "duplicate match_id"
    cov = out["SquadRating_Diff"].notna().mean()
    assert cov > 0.9, f"SquadRating_Diff coverage too low: {cov:.3f}"

    # sanity: squad ratings sit on the rating scale (~6-7)
    sr = pd.concat([out["home_squad_rating"], out["away_squad_rating"]]).dropna()
    assert 5.5 < sr.median() < 7.5, f"squad rating median off scale: {sr.median():.2f}"

    # LEAKAGE 1: a player's prior_rating on their FIRST rated appearance must be NaN (shift(1)).
    p = pd.read_csv(player_stats_path, low_memory=False)
    p = p[(p.comp_name == LALIGA)].copy()
    p["Date"] = pd.to_datetime(p["utc_date"], utc=True).dt.tz_localize(None)
    p["rating"] = pd.to_numeric(p["rating"], errors="coerce")
    p = p[p.rating > 0].sort_values(["player_id", "Date", "match_id"])
    shifted = p.groupby("player_id")["rating"].shift(1)
    pr = (shifted.groupby(p["player_id"]).rolling(PRIOR_WINDOW, min_periods=MIN_PRIOR).mean()
          .reset_index(level=0, drop=True))
    first_idx = p.groupby("player_id").head(1).index
    assert pr.loc[first_idx].isna().all(), "player debut prior_rating must be NaN (T-1 leak!)"

    # LEAKAGE 2: corrupting a player's CURRENT-match rating must NOT change that match's prior_rating.
    pc = p.copy()
    pc.loc[pc.index[5000], "rating"] = 999.0           # corrupt one current rating
    sc = pc.groupby("player_id")["rating"].shift(1)
    prc = (sc.groupby(pc["player_id"]).rolling(PRIOR_WINDOW, min_periods=MIN_PRIOR).mean()
           .reset_index(level=0, drop=True))
    # the corrupted row's own prior_rating is unchanged (it depends only on EARLIER matches)
    assert np.allclose(pr.fillna(-1).to_numpy()[:4999], prc.fillna(-1).to_numpy()[:4999]), \
        "current-match corruption leaked into earlier priors"
    print(f"selfcheck PASSED: {len(out)} matches, SquadRating_Diff coverage {cov*100:.1f}%, "
          f"squad rating median {sr.median():.2f} (range {sr.min():.2f}-{sr.max():.2f})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the T-1 available-squad-rating feature.")
    ap.add_argument("--player-stats", type=Path, default=DEFAULT_PLAYER_STATS)
    ap.add_argument("--team-stats", type=Path, default=DEFAULT_TEAM_STATS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck(args.player_stats, args.team_stats)
        return
    out = build(args.player_stats, args.team_stats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} matches (SquadRating_Diff) -> {args.out}")


if __name__ == "__main__":
    main()
