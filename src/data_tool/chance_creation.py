"""
Anatomy of chance creation (descriptive) -- what is "great xG" actually made of?

xG decides matches but is itself an output. This decomposes it. DESCRIPTIVE: uses each
completed match's own stats (like the explanatory layer; no leakage, no forecast). xG era only
(22/23+, ~3,000 team-matches). The wording below is deliberately conservative -- it reflects exactly what
the data supports (see notes).

Findings (with honest labels for what is MECHANICAL vs a real relation, and what is UNDERPOWERED):
  1. Two NEAR-ORTHOGONAL levers. xG = shots x xG-per-shot. corr(shots,xG) and corr(xG/shot,xG) are both
     strong (~0.7), but corr(shots, xG/shot) ~ +0.06 -- shooting a LOT and shooting WELL-LOCATED are
     near-separate skills. (Verified real, not a ratio artifact: a pure-ratio null forces this corr to
     -0.54; the observed +0.06 is the genuine shots-xG link cancelling the 1/shots drag.)
  2. Possession buys VOLUME, not QUALITY. possession / final-third entries / crosses correlate with xG,
     but the partial correlation GIVEN shot count collapses to ~0 -- they generate more shots, not better
     ones. (The most robust finding here: holds every season, and corroborated by corr(volume, xG/shot)~0.)
  3. Shot quality = LOCATION. xG-per-shot tracks shooting from inside the box (+) vs from distance (-),
     and almost nothing else. (Partly mechanical: xG is computed from shot location -- labelled as such.)
  4. Getting into the BOX is the discriminating step. Possession tracks reaching the final third (+0.56)
     AND getting into the box (+0.50); final-third presence on its own is a WEAKER, noisier indicator of
     actually getting into the box (+0.37). Box presence is what tracks chances. NB this is a correlational
     gradient, NOT a clean funnel (possession predicts box touches better than final-third entries do), and
     box->xG is largely a VOLUME channel (partial | shot count ~ +0.21), not a quality one.
  5. Set-pieces / penalties -- THE DATA CEILING. There is NO open-play / set-piece / counter split in this
     data (the API has no shot-event endpoint), so that analysis is out of reach. The only situational
     signal is penalties: ~10% of league xG (noisy: pen xG = xG - npxG is a difference-of-estimates).
     "Weaker teams lean on penalties more" is directionally suggestive but UNDERPOWERED (n=26 teams, p~0.06).

Usage:
    python -m src.data_tool.chance_creation
    python -m src.data_tool.chance_creation --selfcheck
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data_tool.team_profile import DEFAULT_TEAM_STATS, LALIGA, XG_SEASONS

DEFAULT_OUT = Path("data/processed/chance_creation.json")
LATE = {"24/25", "25/26"}                      # touches_in_penalty_area valid only here (regime break)


def _load(team_stats_path: Path) -> pd.DataFrame:
    df = pd.read_csv(team_stats_path, low_memory=False)
    df = df[(df.comp_name == LALIGA) & (df.season.isin(XG_SEASONS))].copy()
    # drop degenerate rows: a team with shots but zero final-third entries is corrupt (a few 25/26 rows
    # carry xG>0 with every count = 0 -- a raw-data glitch). Immaterial to results, but cleaner out.
    df = df[df.expected_goals.notna() & (df.total_shots > 0) & (df.pass_final_third_entries > 0)]
    df["xg_per_shot"] = df.expected_goals / df.total_shots
    df["pen_xg"] = (df.expected_goals - df.np_expected_goals).clip(lower=0)   # noisy proxy; never negative
    return df


def _rho(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    return round(float(spearmanr(a[m], b[m]).statistic), 2)


def _partial(df: pd.DataFrame, a: str, b: str, ctrl: str) -> float:
    s = df[[a, b, ctrl]].dropna()
    C = np.c_[np.ones(len(s)), s[ctrl].to_numpy(float)]
    res = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    return round(float(np.corrcoef(res(s[a].to_numpy(float)), res(s[b].to_numpy(float)))[0, 1]), 2)


def run(team_stats_path: Path = DEFAULT_TEAM_STATS) -> dict:
    df = _load(team_stats_path)
    late = df[df.season.isin(LATE)]

    two_levers = {  # xG = shots x xG/shot; near-orthogonal channels (verified not a ratio artifact)
        "corr_shots_xg": _rho(df.total_shots, df.expected_goals),
        "corr_quality_xg": _rho(df.xg_per_shot, df.expected_goals),
        "corr_shots_quality": _rho(df.total_shots, df.xg_per_shot),   # ~0 => near-separate skills
    }
    volume_not_quality = {v: {"corr_xg": _rho(df[v], df.expected_goals),
                              "partial_xg_given_shots": _partial(df, v, "expected_goals", "total_shots")}
                          for v in ["ball_possession", "pass_final_third_entries", "pass_accurate_crosses"]}
    quality_is_location = {"inside_box_vs_quality": _rho(df.sh_shots_inside_box, df.xg_per_shot),
                           "outside_box_vs_quality": _rho(df.sh_shots_outside_box, df.xg_per_shot)}
    # getting into the box (correlational gradient, NOT a clean funnel). On LATE (box touches valid there).
    box = {
        "possession_to_final_third": _rho(late.ball_possession, late.pass_final_third_entries),
        "possession_to_box": _rho(late.ball_possession, late.att_touches_in_penalty_area),
        "final_third_to_box": _rho(late.pass_final_third_entries, late.att_touches_in_penalty_area),
        "box_to_xg": _rho(late.att_touches_in_penalty_area, late.expected_goals),
        "box_to_xg_partial_given_shots": _partial(late, "att_touches_in_penalty_area", "expected_goals", "total_shots"),
        "note": "Possession tracks both reaching the final third (+) and getting into the box (+); final-third "
                "presence alone is a weaker, noisier indicator of box penetration. Box presence is what tracks "
                "chances -- but largely as a VOLUME channel (box->xG partial|shots ~ +0.2), not a clean funnel.",
    }
    # penalties = the ONLY situational split the data supports (no open-play/counter split exists)
    ts = df.groupby("team_name").agg(xg=("expected_goals", "mean"), penxg=("pen_xg", "mean")).reset_index()
    ts = ts[df.groupby("team_name").size().values >= 38]            # >= one full season of matches
    ts["pen_share"] = ts.penxg / ts.xg
    rho_strength, p_strength = spearmanr(ts.xg, ts.pen_share)
    set_pieces = {
        "situational_split_available": False,   # no /shots or /events endpoint -> no open-play/counter split
        "penalty_share_of_league_xg": round(float(df.pen_xg.sum() / df.expected_goals.sum()), 3),
        "penalty_share_note": "pen_xg = xG - npxG, a noisy difference-of-estimates (floored at 0).",
        "weaker_teams_lean_on_penalties": {
            "spearman_strength_vs_penshare": round(float(rho_strength), 2), "p": round(float(p_strength), 3),
            "n_teams": int(len(ts)),
            "verdict": "directionally negative but NOT established -- underpowered (n=26) and proxy-sensitive "
                       "(p moves ~0.02-0.06 with the penalty-xG definition; bootstrap CI crosses 0)"},
    }
    return {"n_team_matches": int(len(df)), "n_late": int(len(late)),
            "two_levers": two_levers, "volume_not_quality": volume_not_quality,
            "quality_is_location": quality_is_location, "box_penetration": box, "set_pieces": set_pieces}


def _print_md(r: dict) -> None:
    print(f"\n## Anatomy of chance creation (n={r['n_team_matches']} team-matches, xG era)")
    tl = r["two_levers"]
    print(f"\nxG = shots x xG/shot -- TWO NEAR-ORTHOGONAL LEVERS:")
    print(f"  volume:  corr(shots, xG)      = {tl['corr_shots_xg']:+.2f}")
    print(f"  quality: corr(xG/shot, xG)    = {tl['corr_quality_xg']:+.2f}")
    print(f"  link:    corr(shots, xG/shot) = {tl['corr_shots_quality']:+.2f}  (~0 => near-separate skills)")
    print("\nPossession buys VOLUME, not QUALITY (partial | shot count collapses to ~0) -- the robust core:")
    for v, d in r["volume_not_quality"].items():
        print(f"  {v:26} corr {d['corr_xg']:+.2f}  ->  partial|shots {d['partial_xg_given_shots']:+.2f}")
    ql = r["quality_is_location"]
    print(f"\nShot quality = LOCATION: inside-box {ql['inside_box_vs_quality']:+.2f}, "
          f"from distance {ql['outside_box_vs_quality']:+.2f} (vs xG/shot)")
    b = r["box_penetration"]
    print(f"\nGetting into the BOX (correlational gradient, not a funnel): possession->final-third "
          f"{b['possession_to_final_third']:+.2f}, possession->box {b['possession_to_box']:+.2f}, "
          f"final-third->box {b['final_third_to_box']:+.2f} (weaker); box->xG {b['box_to_xg']:+.2f} "
          f"(but partial|shots {b['box_to_xg_partial_given_shots']:+.2f} => mostly volume)")
    sp = r["set_pieces"]
    w = sp["weaker_teams_lean_on_penalties"]
    print(f"\nSet-pieces/penalties -- DATA CEILING: no open-play/counter split exists (API has no shot-event data).")
    print(f"  penalties ~ {sp['penalty_share_of_league_xg']*100:.0f}% of league xG ({sp['penalty_share_note']})")
    print(f"  weaker teams lean on penalties: {w['verdict']}")


def selfcheck(team_stats_path: Path = DEFAULT_TEAM_STATS) -> None:
    r = run(team_stats_path)
    assert 2500 < r["n_team_matches"] < 3500, f"unexpected n: {r['n_team_matches']}"
    tl = r["two_levers"]
    assert tl["corr_shots_xg"] > 0.5 and tl["corr_quality_xg"] > 0.5, "both levers should be strong"
    assert abs(tl["corr_shots_quality"]) < 0.25, f"levers should be near-orthogonal, got {tl['corr_shots_quality']}"
    # possession adds ~nothing to xG once shot count is fixed (the robust core finding)
    assert abs(r["volume_not_quality"]["ball_possession"]["partial_xg_given_shots"]) < 0.15, \
        "possession should buy shots not quality (partial|shots ~ 0)"
    # shot quality is location: inside-box positive, distance negative
    ql = r["quality_is_location"]
    assert ql["inside_box_vs_quality"] > 0 and ql["outside_box_vs_quality"] < 0, "quality should track location"
    # the robust box gap: final-third->box weaker than possession->final-third (verified CIs disjoint)
    b = r["box_penetration"]
    assert b["final_third_to_box"] < b["possession_to_final_third"], "box penetration should be the weaker step"
    # the honest data ceiling
    assert r["set_pieces"]["situational_split_available"] is False, "no /shots endpoint -> no situational split"
    print("selfcheck PASSED")
    _print_md(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Anatomy of chance creation (what great xG is made of).")
    ap.add_argument("--team-stats", type=Path, default=DEFAULT_TEAM_STATS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck(args.team_stats)
        return
    r = run(args.team_stats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, indent=2))
    _print_md(r)
    print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()
