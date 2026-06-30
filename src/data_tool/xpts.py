"""
Expected points from expected goals (the "xG table") -- a descriptive exhibit, not a prediction.

Given a completed match's team xG, how many points did each side's CHANCES deserve? Model goals as two
independent Poisson variables with means = the two teams' xG, read off P(win)/P(draw)/P(loss), and turn
those into expected points (3 x P(win) + 1 x P(draw)). Sum over a season -> an "xG table": where teams
would sit if results matched the chances they created. The gap to the real table = finishing/keeping
luck (or skill) and game-state effects.

This is EXPLANATORY (it uses each completed match's own xG -- no leakage, like the explanatory layer), not a
forecast. The independent-Poisson step is a classical probability transform on observed xG (permitted
under the tree-only rule; we are not training a model here). It ignores the small score correlation a
Dixon-Coles term would add -- immaterial for a descriptive table (add Dixon-Coles only if a result
ever hinges on it).

Usage:
    python -m src.data_tool.xpts --season 24/25     # print a season's xG table
    python -m src.data_tool.xpts --selfcheck
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.data_tool.team_profile import DEFAULT_TEAM_STATS, LALIGA, XG_SEASONS


def match_probs(xg_home: float, xg_away: float, max_goals: int = 12) -> tuple[float, float, float]:
    """Independent-Poisson P(home win), P(draw), P(away win) from the two teams' xG."""
    g = np.arange(max_goals + 1)
    M = np.outer(poisson.pmf(g, xg_home), poisson.pmf(g, xg_away))   # M[i,j] = P(home i, away j)
    M = M / M.sum()                                                  # renormalise the truncated tail
    return float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())


def expected_points(xg_home: float, xg_away: float) -> tuple[float, float]:
    """Expected points for (home, away) from their xG: 3 x P(win) + 1 x P(draw)."""
    ph, pd_, pa = match_probs(xg_home, xg_away)
    return 3 * ph + pd_, 3 * pa + pd_


def season_xg_table(season: str, team_stats_path=DEFAULT_TEAM_STATS) -> pd.DataFrame:
    """One row per team: actual points/GD vs xG-deserved points (xPts) and xG difference, for a season."""
    if season not in XG_SEASONS:
        raise ValueError(f"xG only exists for {sorted(XG_SEASONS)} (got {season})")
    df = pd.read_csv(team_stats_path, low_memory=False)
    df = df[(df.comp_name == LALIGA) & (df.season == season)].copy()
    h = df[df.is_home][["match_id", "team_name", "expected_goals", "goals"]].rename(
        columns={"team_name": "home", "expected_goals": "xg_h", "goals": "g_h"})
    a = df[~df.is_home][["match_id", "team_name", "expected_goals", "goals"]].rename(
        columns={"team_name": "away", "expected_goals": "xg_a", "goals": "g_a"})
    m = h.merge(a, on="match_id")

    rows: dict[str, dict] = {}
    def add(team, pts, xpts, gf, ga, xgf, xga):
        r = rows.setdefault(team, dict(team=team, played=0, pts=0, xpts=0.0, gf=0, ga=0, xgf=0.0, xga=0.0))
        r["played"] += 1; r["pts"] += pts; r["xpts"] += xpts
        r["gf"] += gf; r["ga"] += ga; r["xgf"] += xgf; r["xga"] += xga

    for r in m.itertuples(index=False):
        xph, xpa = expected_points(r.xg_h, r.xg_a)
        pts_h = 3 if r.g_h > r.g_a else (1 if r.g_h == r.g_a else 0)
        add(r.home, pts_h, xph, r.g_h, r.g_a, r.xg_h, r.xg_a)
        add(r.away, 3 - pts_h if r.g_h != r.g_a else 1, xpa, r.g_a, r.g_h, r.xg_a, r.xg_h)

    t = pd.DataFrame(rows.values())
    t["xpts"] = t["xpts"].round(1)
    t["gd"], t["xgd"] = t["gf"] - t["ga"], (t["xgf"] - t["xga"]).round(1)
    t["over_perf"] = (t["pts"] - t["xpts"]).round(1)          # + = scored more points than chances deserved
    return t.sort_values("xpts", ascending=False).reset_index(drop=True)


def _print(season: str, t: pd.DataFrame) -> None:
    print(f"\nLa Liga {season} -- xG table (ranked by deserved points; over_perf = actual - deserved)")
    print(f"{'#':>2}  {'team':22}{'P':>4}{'xPts':>7}{'over':>7}{'GD':>5}{'xGD':>7}")
    real_rank = {tm: i + 1 for i, tm in enumerate(t.sort_values("pts", ascending=False).team)}
    for i, r in enumerate(t.itertuples(), 1):
        mv = real_rank[r.team] - i                            # +ve = real table higher than deserved
        print(f"{i:>2}  {r.team:22}{r.pts:>4}{r.xpts:>7.1f}{r.over_perf:>+7.1f}{r.gd:>5}{r.xgd:>+7.1f}"
              f"   (real #{real_rank[r.team]}{'' if mv == 0 else f', {mv:+d}'})")


def selfcheck(team_stats_path=DEFAULT_TEAM_STATS) -> None:
    # a clear favourite (xG 2.5 vs 0.5) should deserve ~3 points; an even game ~ the same for both
    ph, pdr, pa = match_probs(2.5, 0.5)
    assert ph > 0.75 and pa < 0.10, f"strong xG edge mis-priced: {ph:.2f}/{pdr:.2f}/{pa:.2f}"
    assert abs(sum(match_probs(1.3, 1.3)) - 1.0) < 1e-9, "probs must sum to 1"
    xh, xa = expected_points(1.3, 1.3)
    assert abs(xh - xa) < 1e-9 and 1.0 < xh < 1.6, f"even game xPts off: {xh:.2f}"
    # season table sanity
    t = season_xg_table("24/25", team_stats_path)
    assert len(t) == 20 and (t.played == 38).all(), f"expected 20 teams x 38 games, got {len(t)}"
    assert abs(t.pts.sum() - t.xpts.sum()) < 0.5 * len(t), "total xPts should be near total real points"
    assert t.iloc[0].xpts > t.iloc[-1].xpts, "table must be ranked by xPts"
    print("selfcheck PASSED")
    _print("24/25", t)


def main() -> None:
    ap = argparse.ArgumentParser(description="Expected-points (xG) table for a La Liga season.")
    ap.add_argument("--season", default="24/25")
    ap.add_argument("--team-stats", default=DEFAULT_TEAM_STATS)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck(args.team_stats)
        return
    _print(args.season, season_xg_table(args.season, args.team_stats))


if __name__ == "__main__":
    main()
