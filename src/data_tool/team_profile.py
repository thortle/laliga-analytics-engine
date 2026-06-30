"""
Team-quality profile (descriptive) -- WHAT CHARACTERIZES A GREAT LA LIGA TEAM?

Flips the predictive question (the model hit a ceiling beating Elo) into a descriptive one:
use Elo as the TARGET and ask which team characteristics are most associated with being a
high-Elo team. No leakage concern (descriptive, not forecasting) and no power ceiling -- we use
all 6 seasons at the team-season level (~120 team-seasons).

The key methodological move (the circularity lesson): Elo is built FROM results, so the
"QUALITY CORE" stats (goals, xG, shots-on-target, big chances) correlate with Elo almost
tautologically -- great teams score, so of course high xG ~ high Elo. The INTERESTING answer is
the STYLE SIGNATURE: among non-scoring stats (possession, passing, pressing, duels, directness,
discipline, defending), which characterize great teams -- and which survive controlling for the
quality core (partial correlation given goal difference)?

Outputs:
  - ranked Spearman corr(stat, season-avg Elo), each labelled QUALITY_CORE / STYLE, with direction
  - partial corr(style stat, Elo | goal_diff) -- the style signature BEYOND just outscoring
  - tree (RF) permutation importance on STYLE stats only -- the joint style profile of quality
  - elite (top-quartile Elo) vs poor (bottom-quartile) mean-stat profile -- the presentable table
  - the possession paradox: corr(possession, Elo) [marks quality] vs ~0 within a match (doesn't decide a match)

Usage:
    python -m src.data_tool.team_profile             # profile -> JSON + markdown
    python -m src.data_tool.team_profile --selfcheck
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

DEFAULT_TEAM_STATS = Path("data/raw/thestatsapi/team_match_stats.csv")
DEFAULT_OUT = Path("data/processed/team_profile.json")
LALIGA = "La Liga"
XG_SEASONS = {"22/23", "23/24", "24/25", "25/26"}
SEED = 42

# stats built from / proxying the scoreline -> correlate with Elo ~tautologically (great teams score)
QUALITY_CORE = {"goals", "expected_goals", "np_expected_goals", "big_chances", "att_big_chances_missed",
                "shots_on_target", "total_shots", "sh_shots_on_target", "sh_total_shots",
                "sh_shots_inside_box", "goalkeeper_saves", "gk_saves", "gk_goals_prevented"}
# meta / keys -- never a profile feature (goals scored IS a profile trait, kept; QUALITY_CORE)
NON_FEATURE = {"match_id", "comp_name", "season", "utc_date", "is_home", "team_id", "team_name",
               "opp_team_id"}
# identical twins (keep one) -- else the ranking lists the same finding multiple times
DROP_REDUNDANT = {"sh_total_shots", "sh_shots_on_target", "pass_accurate_passes", "gk_saves",
                  "def_tackles", "np_expected_goals"}
# columns with a mid-dataset DEFINITION/COVERAGE break -> a season-mean across the break is garbage
# (verified: each is ~0 in early seasons then jumps x30-300). Drop, or gate to the valid era.
DROP_CONTAMINATED = {"def_ball_recoveries", "gk_high_claims", "gk_goals_prevented"}
LATE_ONLY = {"att_touches_in_penalty_area": {"24/25", "25/26"}}   # real only from 24/25
# the collinear ball-domination construct (one finding, not four) -> composite for the honest magnitude
BALL_DOMINATION = ["ball_possession", "passes", "pass_accuracy"]
QUALITY_FOR_RSQ = ["goals", "goal_diff"]   # the scoring core to measure style's increment OVER


def _elo_per_team_match(df: pd.DataFrame, k=32.0, hfa=100.0, start=1500.0) -> pd.DataFrame:
    """Running pre-match Elo, returned per (match_id, team_id)."""
    h = df[df.is_home][["match_id", "utc_date", "team_id", "goals"]].rename(columns={"team_id": "home_id", "goals": "hg"})
    a = df[~df.is_home][["match_id", "team_id", "goals"]].rename(columns={"team_id": "away_id", "goals": "ag"})
    m = h.merge(a, on="match_id").sort_values(["utc_date", "match_id"]).reset_index(drop=True)
    rating: dict[str, float] = defaultdict(lambda: start)
    rows = []
    for r in m.itertuples(index=False):
        hp, ap = rating[r.home_id], rating[r.away_id]
        rows.append((r.match_id, r.home_id, hp)); rows.append((r.match_id, r.away_id, ap))
        exp_h = 1.0 / (1.0 + 10.0 ** ((ap - (hp + hfa)) / 400.0))
        act_h = 1.0 if r.hg > r.ag else (0.0 if r.hg < r.ag else 0.5)
        d = k * (act_h - exp_h)
        rating[r.home_id] = hp + d; rating[r.away_id] = ap - d
    return pd.DataFrame(rows, columns=["match_id", "team_id", "elo_pre"])


def team_seasons(team_stats_path: Path = DEFAULT_TEAM_STATS) -> tuple[pd.DataFrame, list[str]]:
    """One row per (team, season): mean stats + mean pre-match Elo + points. xG stats NaN pre-22/23."""
    df = pd.read_csv(team_stats_path, low_memory=False)
    df = df[df.comp_name == LALIGA].copy()
    # opponent goals (self-join) for points + goal difference
    opp = df[["match_id", "team_id", "goals"]].rename(columns={"team_id": "opp_team_id", "goals": "opp_goals"})
    df = df.merge(opp, on=["match_id", "opp_team_id"], how="left")
    df["points"] = np.where(df.goals > df.opp_goals, 3, np.where(df.goals == df.opp_goals, 1, 0))
    df["goal_diff"] = df.goals - df.opp_goals
    # gate xG-core to the xG era (0 = missing pre-22/23)
    for c in ("expected_goals", "np_expected_goals", "big_chances", "att_big_chances_missed"):
        if c in df:
            df[c] = df[c].where(df.season.isin(XG_SEASONS))
    # gate columns with a mid-dataset regime break to their valid era (NaN before)
    for c, seasons in LATE_ONLY.items():
        if c in df:
            df[c] = df[c].where(df.season.isin(seasons))
    df = df.drop(columns=[c for c in DROP_REDUNDANT | DROP_CONTAMINATED if c in df], errors="ignore")
    df["pass_accuracy"] = df["accurate_passes"] / df["passes"].replace(0, np.nan)

    elo = _elo_per_team_match(df)
    df = df.merge(elo, on=["match_id", "team_id"], how="left")

    stat_cols = [c for c in df.columns if c not in NON_FEATURE | {"opp_goals", "points", "goal_diff", "elo_pre"}
                 and pd.api.types.is_numeric_dtype(df[c])]
    agg = {c: "mean" for c in stat_cols}
    agg.update({"elo_pre": "mean", "goal_diff": "mean", "points": "sum", "match_id": "count"})
    ts = df.groupby(["team_id", "team_name", "season"]).agg(agg).rename(columns={"match_id": "n_matches"}).reset_index()
    ts = ts[ts.n_matches >= 30]                       # full La Liga seasons only
    return ts, stat_cols


def _partial_corr(ts: pd.DataFrame, feat: str, target: str, ctrl: str) -> float | None:
    sub = ts[[feat, target, ctrl]].dropna()
    if len(sub) < 30:
        return None
    C = np.c_[np.ones(len(sub)), sub[ctrl].to_numpy(float)]
    res = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    return float(np.corrcoef(res(sub[feat].to_numpy(float)), res(sub[target].to_numpy(float)))[0, 1])


def run(team_stats_path: Path = DEFAULT_TEAM_STATS) -> dict:
    ts, stat_cols = team_seasons(team_stats_path)
    target = "elo_pre"

    ranking = []
    for c in stat_cols:
        sub = ts[[c, target, "goal_diff"]].dropna()
        if len(sub) < 30:
            continue
        rho = float(spearmanr(sub[c], sub[target]).statistic)
        kind = "QUALITY_CORE" if c in QUALITY_CORE else "STYLE"
        pr = _partial_corr(ts, c, target, "goal_diff") if kind == "STYLE" else None
        ranking.append({"stat": c, "kind": kind, "spearman_vs_elo": round(rho, 3),
                        "direction": "+" if rho >= 0 else "-",
                        "partial_vs_elo_given_goaldiff": None if pr is None else round(pr, 3),
                        "n": len(sub)})
    ranking.sort(key=lambda r: -abs(r["spearman_vs_elo"]))

    # STYLE-only RF importance (the joint style profile of quality). Exclude QUALITY_CORE and the
    # regime-gated LATE_ONLY cols (their NaN would collapse the frame to the late-era subset).
    style = [c for c in stat_cols if c not in QUALITY_CORE and c not in LATE_ONLY]
    s = ts[style + [target]].dropna()
    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
    rf.fit(s[style], s[target])
    pi = permutation_importance(rf, s[style], s[target], n_repeats=20, random_state=SEED, n_jobs=-1)
    style_importance = sorted([{"stat": f, "importance": round(float(pi.importances_mean[i]), 4)}
                               for i, f in enumerate(style)], key=lambda r: -r["importance"])[:12]

    # elite vs poor profile (top/bottom Elo quartile), ranked by Cohen's d (standardized gap --
    # NOT %-change, which blows up on near-zero-base columns and is denominator-sensitive)
    q_hi, q_lo = ts[target].quantile(0.75), ts[target].quantile(0.25)
    elite, poor = ts[ts[target] >= q_hi], ts[ts[target] <= q_lo]
    profile = []
    for c in stat_cols:
        e, p = elite[c].dropna(), poor[c].dropna()
        if len(e) < 5 or len(p) < 5:
            continue
        psd = np.sqrt((e.var(ddof=1) + p.var(ddof=1)) / 2)
        if not psd or np.isnan(psd):
            continue
        profile.append({"stat": c, "kind": "QUALITY_CORE" if c in QUALITY_CORE else "STYLE",
                        "elite_mean": round(float(e.mean()), 2), "poor_mean": round(float(p.mean()), 2),
                        "cohens_d": round(float((e.mean() - p.mean()) / psd), 2)})
    profile.sort(key=lambda r: -abs(r["cohens_d"]))

    # HONEST MAGNITUDE: how much does ball-domination add OVER the scoring core for explaining Elo?
    comp = ts.dropna(subset=BALL_DOMINATION + QUALITY_FOR_RSQ + [target]).copy()
    z = (comp[BALL_DOMINATION] - comp[BALL_DOMINATION].mean()) / comp[BALL_DOMINATION].std()
    comp["ball_domination"] = z.mean(axis=1)
    def _r2(cols):
        X = np.c_[np.ones(len(comp)), comp[cols].to_numpy(float)]
        beta, *_ = np.linalg.lstsq(X, comp[target].to_numpy(float), rcond=None)
        pred = X @ beta
        ss_res = ((comp[target] - pred) ** 2).sum(); ss_tot = ((comp[target] - comp[target].mean()) ** 2).sum()
        return float(1 - ss_res / ss_tot)
    r2_core = _r2(QUALITY_FOR_RSQ); r2_both = _r2(QUALITY_FOR_RSQ + ["ball_domination"])
    ball_dom = {"composite": "z-mean(possession, passes, pass_accuracy)",
                "spearman_vs_elo": round(float(spearmanr(comp["ball_domination"], comp[target]).statistic), 3),
                "partial_vs_elo_given_goaldiff": round(_partial_corr(comp.assign(bd=comp["ball_domination"]),
                                                                     "bd", target, "goal_diff") or 0, 3),
                "r2_quality_core": round(r2_core, 3), "r2_core_plus_balldom": round(r2_both, 3),
                "incremental_r2": round(r2_both - r2_core, 3)}

    poss = next((r for r in ranking if r["stat"] == "ball_possession"), None)
    return {"n_team_seasons": int(len(ts)), "target": "season-avg pre-match Elo",
            "ranking": ranking, "style_signature_importance": style_importance,
            "ball_domination_construct": ball_dom, "elite_vs_poor_profile": profile[:20],
            "possession_paradox": {
                "corr_possession_vs_elo": poss["spearman_vs_elo"] if poss else None,
                "note": "BETWEEN teams possession marks quality; WITHIN a match it does not predict the "
                        "result once quality is held fixed (partial corr possession_diff,margin|Elo "
                        "= -0.118). Possession is downstream of being the better side, not a cause of a win."}}


def _print_md(res: dict) -> None:
    print(f"\n## What characterizes a great La Liga team? (n={res['n_team_seasons']} team-seasons; target = {res['target']})")
    print("\nstat vs Elo (|Spearman| desc; STYLE also shows partial corr | goal-diff = signature beyond scoring):")
    print(f"{'stat':32} {'kind':12} {'rho':>6} {'partial|GD':>11}")
    for r in res["ranking"][:22]:
        pc = "" if r["partial_vs_elo_given_goaldiff"] is None else f"{r['partial_vs_elo_given_goaldiff']:+.2f}"
        print(f"{r['stat']:32} {r['kind']:12} {r['spearman_vs_elo']:+6.2f} {pc:>11}")
    print("\nSTYLE signature (RF permutation importance on non-scoring stats, predicting Elo):")
    for r in res["style_signature_importance"][:10]:
        print(f"  {r['stat']:30} {r['importance']:+.4f}")
    bd = res["ball_domination_construct"]
    print(f"\nball-domination construct ({bd['composite']}): corr {bd['spearman_vs_elo']}, partial|GD "
          f"{bd['partial_vs_elo_given_goaldiff']}; incremental R2 over scoring {bd['incremental_r2']} "
          f"({bd['r2_quality_core']} -> {bd['r2_core_plus_balldom']}) -- REAL but SMALL.")
    print("\nelite (top-Elo quartile) vs poor (bottom) -- biggest standardized gaps (Cohen's d):")
    for r in res["elite_vs_poor_profile"][:14]:
        print(f"  {r['stat']:30} {r['kind']:12} elite {r['elite_mean']:>8.2f}  poor {r['poor_mean']:>8.2f}  (d {r['cohens_d']:+.2f})")
    pp = res["possession_paradox"]
    print(f"\npossession paradox: corr(possession, Elo) = {pp['corr_possession_vs_elo']} (marks quality) "
          f"vs ~0 within a match.")


def selfcheck(team_stats_path: Path = DEFAULT_TEAM_STATS) -> None:
    res = run(team_stats_path)
    assert 90 < res["n_team_seasons"] < 130, f"expected ~120 team-seasons, got {res['n_team_seasons']}"
    rank = {r["stat"]: r for r in res["ranking"]}
    # sanity: quality core strongly +correlated with Elo (great teams score)
    assert rank["goals"]["spearman_vs_elo"] > 0.6, "goals should strongly track Elo"
    # possession should positively mark quality
    assert rank["ball_possession"]["spearman_vs_elo"] > 0.3, "possession should mark quality"
    print("selfcheck PASSED")
    _print_md(res)


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile what characterizes a great La Liga team.")
    ap.add_argument("--team-stats", type=Path, default=DEFAULT_TEAM_STATS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck(args.team_stats)
        return
    res = run(args.team_stats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    _print_md(res)
    print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()
