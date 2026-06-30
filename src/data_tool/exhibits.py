"""
Portfolio exhibits -- render the three headline findings as committed PNG figures so they
show on GitHub with zero setup. Each figure reuses an existing, already-verified analysis module; this
file only draws.

  1. what_decides_a_match.png -- explanatory permutation importance: chance quality (xG) dominates, possession ~0
  2. style_map.png            -- style map: a continuum (not archetypes); Atletico the off-possession outlier
  3. xg_expected_table.png    -- the xG "expected points" table: points earned vs points deserved (24/25)

Usage:
    python -m src.data_tool.exhibits            # write all three -> docs/figures/
    python -m src.data_tool.exhibits --selfcheck
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # headless: render to file, never open a window
import matplotlib.pyplot as plt

from src.data_tool import explanatory_rank, team_style, xpts

FIG_DIR = Path("docs/figures")
BLUE, RED, GREY = "#1f6feb", "#d1242f", "#9aa0a6"

# plain-language labels for the explanatory stat diffs (the brief: simple terminology). The raw columns
# carry the provider's SECTION prefix (pass_/sh_/def_/gk_/duel_/att_/gene_) which is not a real word --
# the fallback strips it so unmapped stats still read cleanly.
NICE = {
    "expected_goals_diff": "Chance quality (xG)", "ball_possession_diff": "Possession",
    "total_shots_diff": "Shots", "sh_shots_outside_box_diff": "Shots from distance",
    "sh_shots_off_target_diff": "Shots off target", "pass_accurate_crosses_diff": "Accurate crosses",
    "pass_throw_ins_diff": "Throw-ins", "def_ball_recoveries_diff": "Ball recoveries",
    "pass_final_third_entries_diff": "Final-third entries", "corner_kicks_diff": "Corners",
    "fouls_diff": "Fouls", "duel_aerial_duels_percentage_diff": "Aerial-duel share",
    "att_touches_in_penalty_area_diff": "Touches in the box", "pass_accuracy_diff": "Pass accuracy",
    "accurate_passes_diff": "Passing", "tackles_diff": "Tackles", "def_interceptions_diff": "Interceptions",
}
# xG (the one real driver) vs a handful of RECOGNIZABLE stats for the headline chart. Showing the literal
# next-ranked stats would dignify noise: every non-xG score is ~0.01-0.06 (10-50x below xG) and their
# order among themselves isn't meaningful.
WHAT_DECIDES_SHOW = ["expected_goals_diff", "total_shots_diff", "accurate_passes_diff",
                     "tackles_diff", "ball_possession_diff"]
_SECTION = ("pass_", "sh_", "def_", "gk_", "duel_", "att_", "gene_")
def nice(f: str) -> str:
    if f in NICE:
        return NICE[f]
    s = f.removesuffix("_diff")
    for p in _SECTION:
        if s.startswith(p):
            s = s[len(p):]
            break
    return s.replace("_", " ").capitalize()


def fig_what_decides(out: Path) -> None:
    rank = {r["feature"]: r["mean_perm_logloss_drop"] for r in explanatory_rank.run()["frame_C_clean"]["ranking"]}
    rows = sorted([(f, rank[f]) for f in WHAT_DECIDES_SHOW if f in rank], key=lambda kv: kv[1])  # biggest on top
    labels = [nice(f) for f, _ in rows]
    vals = [v for _, v in rows]
    colors = [RED if f == "expected_goals_diff" else (GREY if f == "ball_possession_diff" else BLUE) for f, _ in rows]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    ax.barh(labels, vals, color=colors)
    ax.set_xlabel("Predictive importance  (drop in log-loss when the stat is shuffled, held-out test)")
    fig.suptitle("What decides a La Liga match?", fontsize=14, fontweight="bold")
    ax.set_title("Chance quality (xG) dwarfs every common stat — possession, shots, passing are all close to zero.\n"
                 "(Stats that merely restate the score, e.g. shots on target, are excluded as circular.)",
                 fontsize=9, color="#555")
    for y, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, y, f"{v:.3f}", va="center", fontsize=8.5, color="#333")
    ax.margins(x=0.12)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_style_map(out: Path) -> None:
    res = team_style.run()
    pts = res["map"]
    poles = sorted({p["pole"] for p in pts})
    cmap = {poles[0]: BLUE, poles[1]: RED} if len(poles) == 2 else {p: BLUE for p in poles}

    fig, ax = plt.subplots(figsize=(8.5, 7), layout="constrained")
    ax.axhline(0, color="#ddd", lw=1, zorder=0); ax.axvline(0, color="#ddd", lw=1, zorder=0)
    for pole in poles:
        xs = [p["pc1"] for p in pts if p["pole"] == pole]
        ys = [p["pc2"] for p in pts if p["pole"] == pole]
        ax.scatter(xs, ys, s=22, c=cmap[pole], alpha=0.55, label=pole, edgecolors="none")

    # annotate a few recognizable team-seasons at the extremes + the elite outliers
    want = {("Barcelona", "20/21"), ("Barcelona", "24/25"), ("Real Madrid", "24/25"),
            ("Atlético Madrid", "23/24"), ("Atlético Madrid", "24/25"), ("Getafe", "21/22"),
            ("Girona FC", "22/23"), ("Real Sociedad", "21/22")}
    for p in pts:
        if (p["team"], p["season"]) in want:
            ax.annotate(f"{p['team']} {p['season']}", (p["pc1"], p["pc2"]), fontsize=7.5,
                        xytext=(4, 4), textcoords="offset points", color="#222")
    ax.set_xlabel("PC1:  physical / aerial  ←——→  possession / passing / dribbling")
    ax.set_ylabel("PC2:  deep / reactive  ←——→  territorial penetration")
    fig.suptitle("La Liga team-season style map (style at fixed quality)", fontsize=14, fontweight="bold")
    ax.set_title("No discrete archetypes — a continuum. Atletico is the elite off-possession outlier.",
                 fontsize=9.5, color="#555")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_xg_table(out: Path, season: str = "24/25") -> None:
    t = xpts.season_xg_table(season).sort_values("xpts").reset_index(drop=True)  # ascending -> top of chart = best
    y = range(len(t))
    fig, ax = plt.subplots(figsize=(8.5, 8), layout="constrained")
    for i, r in enumerate(t.itertuples()):
        col = RED if r.over_perf < -3 else (BLUE if r.over_perf > 3 else GREY)
        ax.plot([r.xpts, r.pts], [i, i], color=col, lw=2, zorder=1)
    ax.scatter(t.xpts, y, s=40, color="#222", zorder=2, label="Deserved (xPts)")
    ax.scatter(t.pts, y, s=40, facecolors="white", edgecolors="#222", zorder=3, label="Actual points")
    ax.set_yticks(list(y)); ax.set_yticklabels(t.team, fontsize=8.5)
    ax.set_xlabel("League points")
    fig.suptitle(f"La Liga {season}: points earned vs points deserved (xG)", fontsize=14, fontweight="bold")
    ax.set_title("Dot = points deserved by chances (Poisson on xG); circle = actual. Red = under-, blue = over-performed.\n"
                 "For top teams the gap is largely REPEATABLE finishing skill (correlates with strength, persists yearly), not just luck.",
                 fontsize=8.5, color="#555")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.margins(y=0.02)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def build_all(fig_dir: Path = FIG_DIR) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    jobs = [("what_decides_a_match.png", fig_what_decides), ("style_map.png", fig_style_map),
            ("xg_expected_table.png", fig_xg_table)]
    out = []
    for name, fn in jobs:
        p = fig_dir / name
        fn(p)
        print(f"wrote {p}  ({p.stat().st_size // 1024} KB)")
        out.append(p)
    return out


def selfcheck() -> None:
    paths = build_all()
    for p in paths:
        assert p.exists() and p.stat().st_size > 5_000, f"figure not rendered or too small: {p}"
    print("selfcheck PASSED")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the portfolio figures.")
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    build_all(args.out_dir)


if __name__ == "__main__":
    main()
