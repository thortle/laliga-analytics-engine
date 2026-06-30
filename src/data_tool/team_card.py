"""
The demo CLI -- a small, presentable view onto the engine. Two commands, both reuse the
already-built + verified analysis modules; no new analysis here.

  card   "<team>" [--season]   -> a team's fingerprint: quality (Elo percentile, points) + style
                                  (where it sits on the style map) + a few recognizable stats.
  match  "<home>" "<away>" [--season]
                               -> explain a completed match: the chances each side created (xG), the
                                  points those chances "deserved" (Poisson, the xG-table engine), the
                                  actual score, and an honest read of result vs deserved.

Usage:
    python -m src.data_tool.team_card card  "Atletico Madrid" --season 24/25
    python -m src.data_tool.team_card match "Real Madrid" "Barcelona" --season 24/25
    python -m src.data_tool.team_card --selfcheck
"""
from __future__ import annotations

import argparse
import unicodedata

import numpy as np
import pandas as pd

from src.data_tool import team_style, xpts
from src.data_tool.team_profile import DEFAULT_TEAM_STATS, LALIGA, XG_SEASONS, team_seasons


def _norm(s: str) -> str:
    """lowercase + strip accents so 'atletico' matches 'Atlético'."""
    return "".join(c for c in unicodedata.normalize("NFD", str(s).lower()) if unicodedata.category(c) != "Mn")


def _ord(n: int) -> str:
    """1 -> '1st', 2 -> '2nd', 11 -> '11th' ..."""
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _resolve(query: str, names) -> str:
    """Find the one team_name whose normalized form contains the query; error helpfully otherwise."""
    q = _norm(query)
    hits = sorted({n for n in names if q in _norm(n)})
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(f"no team matches '{query}'. Available: {', '.join(sorted(set(names)))}")
    raise SystemExit(f"'{query}' is ambiguous: {', '.join(hits)} -- be more specific")


def card(team: str, season: str | None = None, team_stats_path=DEFAULT_TEAM_STATS) -> dict:
    ts, _ = team_seasons(team_stats_path)
    name = _resolve(team, ts.team_name)
    rows = ts[ts.team_name == name]
    season = season or sorted(rows.season)[-1]                 # default: most recent season on record
    if season not in set(rows.season):
        raise SystemExit(f"{name} has no {season} season (have: {', '.join(sorted(rows.season))})")
    r = rows[rows.season == season].iloc[0]

    elo_pct = round(float((ts.elo_pre <= r.elo_pre).mean() * 100))
    same = ts[ts.season == season].sort_values("points", ascending=False).reset_index(drop=True)
    finish = int(same.index[same.team_name == name][0]) + 1
    m = next((p for p in team_style.run(team_stats_path)["map"]
              if p["team"] == name and p["season"] == season), None)

    out = {"team": name, "season": season, "elo": round(float(r.elo_pre)), "elo_pct": elo_pct,
           "points": int(r.points), "goal_diff": int(round(r.goal_diff * r.n_matches)), "finish": finish,
           "possession": round(float(r.ball_possession), 1), "pass_acc": round(float(r.pass_accuracy) * 100, 1),
           "aerial_pct": round(float(r.duel_aerial_duels_percentage), 1),
           "final_third": round(float(r.pass_final_third_entries), 1),
           "pole": m["pole"] if m else None, "pc1": m["pc1"] if m else None, "pc2": m["pc2"] if m else None}
    # one honest read tying style + quality together
    elite = elo_pct >= 75
    if out["pole"] == "physical/direct":
        out["read"] = ("An elite side winning WITHOUT dominating the ball -- the off-possession route to the top."
                       if elite else "A physical, direct side: less of the ball, more duels and clearances.")
    else:
        out["read"] = ("A ball-dominant top side -- the textbook 'great team' style." if elite
                       else "A ball-dominant side: keeps the ball and builds, without (yet) elite results.")
    return out


def _print_card(c: dict) -> None:
    print(f"\n=== {c['team']} -- {c['season']} ===")
    print(f"Quality:   Elo {c['elo']} ({_ord(c['elo_pct'])} pct of all team-seasons) | "
          f"{c['points']} pts | GD {c['goal_diff']:+d} | finished {_ord(c['finish'])}")
    if c["pole"]:
        print(f"Style:     {c['pole']} pole | style-map position (PC1 {c['pc1']:+.1f}, PC2 {c['pc2']:+.1f})")
    print(f"Key stats: possession {c['possession']}% | pass accuracy {c['pass_acc']}% | "
          f"aerial-duel share {c['aerial_pct']}% | final-third entries {c['final_third']}/match")
    print(f"Read:      {c['read']}")


def match(home: str, away: str, season: str = "24/25", team_stats_path=DEFAULT_TEAM_STATS) -> dict:
    df = pd.read_csv(team_stats_path, low_memory=False)
    df = df[(df.comp_name == LALIGA) & (df.season == season)].copy()
    h_name, a_name = _resolve(home, df.team_name), _resolve(away, df.team_name)
    H = df[df.is_home][["match_id", "team_name", "goals", "expected_goals", "ball_possession", "total_shots"]]
    A = df[~df.is_home][["match_id", "team_name", "goals", "expected_goals", "ball_possession", "total_shots"]]
    m = H.merge(A, on="match_id", suffixes=("_h", "_a"))
    row = m[(m.team_name_h == h_name) & (m.team_name_a == a_name)]
    if row.empty:
        raise SystemExit(f"no {season} match with {h_name} at home to {a_name} "
                         f"(each pairing plays home once per season; try swapping, or --season)")
    r = row.iloc[0]
    has_xg = season in XG_SEASONS
    out = {"season": season, "home": h_name, "away": a_name, "score": f"{int(r.goals_h)}-{int(r.goals_a)}",
           "possession": (round(float(r.ball_possession_h), 1), round(float(r.ball_possession_a), 1)),
           "shots": (int(r.total_shots_h), int(r.total_shots_a)), "has_xg": has_xg}
    if has_xg:
        ph, pdr, pa = xpts.match_probs(float(r.expected_goals_h), float(r.expected_goals_a))
        xph, xpa = xpts.expected_points(float(r.expected_goals_h), float(r.expected_goals_a))
        out.update({"xg": (round(float(r.expected_goals_h), 2), round(float(r.expected_goals_a), 2)),
                    "p_home_draw_away": (round(ph, 2), round(pdr, 2), round(pa, 2)),
                    "xpts": (round(xph, 2), round(xpa, 2))})
        deserved = h_name if xph - xpa > 0.4 else (a_name if xpa - xph > 0.4 else "neither (even)")
        actual = h_name if r.goals_h > r.goals_a else (a_name if r.goals_a > r.goals_h else "draw")
        if deserved in (actual, "neither (even)") or actual == "draw":
            out["read"] = f"Chances and result broadly agree (deserved: {deserved}; result: {actual})."
        else:
            out["read"] = (f"The chances favoured {deserved}, but {actual} took the points -- a result driven by "
                           f"finishing/keeping, not chance creation. Single matches are noisy; xG decides on average.")
    return out


def _print_match(d: dict) -> None:
    print(f"\n=== {d['home']} {d['score']} {d['away']}  ({d['season']}) ===")
    if d["has_xg"]:
        print(f"Chances (xG):  {d['home']} {d['xg'][0]}  -  {d['xg'][1]} {d['away']}")
        print(f"Deserved pts:  {d['xpts'][0]}  -  {d['xpts'][1]}   "
              f"(win prob H/D/A: {d['p_home_draw_away'][0]}/{d['p_home_draw_away'][1]}/{d['p_home_draw_away'][2]})")
    else:
        print("Chances (xG):  not available before 22/23")
    print(f"Possession:    {d['possession'][0]}%  -  {d['possession'][1]}%      Shots: {d['shots'][0]} - {d['shots'][1]}")
    if "read" in d:
        print(f"Read:          {d['read']}")


def selfcheck(team_stats_path=DEFAULT_TEAM_STATS) -> None:
    c = card("Atletico Madrid", "24/25", team_stats_path)
    assert c["team"] == "Atlético Madrid" and c["pole"] == "physical/direct", f"Atletico card off: {c}"
    assert c["elo_pct"] >= 75 and "off-possession" in c["read"], "Atletico should read as an elite off-possession side"
    d = match("Real Madrid", "Barcelona", "24/25", team_stats_path)
    assert d["has_xg"] and d["score"].count("-") == 1 and len(d["xpts"]) == 2, f"match explain off: {d}"
    # case-insensitive substring resolution works
    assert card("barcelona", "24/25", team_stats_path)["team"] == "Barcelona", "name resolution failed"
    print("selfcheck PASSED")
    _print_card(c)
    _print_match(d)


def main() -> None:
    ap = argparse.ArgumentParser(description="La Liga analytics demo CLI (team fingerprint + match explainer).")
    ap.add_argument("--selfcheck", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    pc = sub.add_parser("card", help="a team's quality + style fingerprint")
    pc.add_argument("team"); pc.add_argument("--season", default=None)
    pm = sub.add_parser("match", help="explain a completed match (chances vs result)")
    pm.add_argument("home"); pm.add_argument("away"); pm.add_argument("--season", default="24/25")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
    elif args.cmd == "card":
        _print_card(card(args.team, args.season))
    elif args.cmd == "match":
        _print_match(match(args.home, args.away, args.season))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
