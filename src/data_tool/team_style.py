"""
Style fingerprints (descriptive). The honest version, after correcting an initial over-claim.

The team-quality profile (see team_profile.py) found the AVERAGE great La Liga team
dominates the ball, but flagged a caveat: at a fixed Elo band, teams range from 34%-possession direct
sides to 60%-possession sides. This module set out to cluster team-seasons into named STYLE archetypes
and show "multiple routes to greatness". A closer look overturned that framing, and the code now reports the
finding that actually holds:

  1. STYLE IS A CONTINUUM, NOT ARCHETYPES. The best k-means split (k=2) scores silhouette 0.197 -- but
     so does clustering matched-covariance Gaussian noise (~0.187, p~0.26); Hopkins ~0.61 (< 0.75 =
     not clusterable); the silhouette curve declines monotonically for k>2. There are no discrete tribes.
     What IS real is a low-dimensional CONTINUOUS structure (~6 effective dims; a dominant possession/
     passing-vs-aerial axis) -- a 2-D style MAP, not clusters. (The map's axes are seed- and
     leave-one-out-stable and sit 20+ SD above a feature-shuffle null -- the coordinate system is real.)
  2. "MULTIPLE ROUTES TO GREATNESS" WAS CIRCULAR. Residualizing style on Elo forces clusters to be
     quality-balanced, so "elite teams appear in every style region" is guaranteed by construction, not
     a finding (chi-square elite-vs-style p~0.4-0.5; a random relabel fills every bin too). The honest
     statement: at fixed quality, residual style is ~orthogonal to quality -- no single style marks or
     makes a great team. Caveat at the extreme: the two super-clubs (Madrid/Barca) both play possession.
  3. THE ONE ROBUST SPECIFIC FINDING: Atletico Madrid is a genuinely elite side built off a
     NON-possession identity (strongly negative possession axis every season) -- the standout exception.

Pipeline (the map is honest; the clustering is reported as the null result it is):
  team_seasons() [reused] -> z-score WITHIN season (kill league-wide drift) -> residualize on Elo
  (style at fixed quality) -> standardize -> PCA (the 2-D map) -> k-means + clusterability nulls.

Usage:
    python -m src.data_tool.team_style              # -> JSON + markdown
    python -m src.data_tool.team_style --selfcheck
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.data_tool.team_profile import DEFAULT_TEAM_STATS, team_seasons

DEFAULT_OUT = Path("data/processed/team_style.json")
SEED = 42

# Recognizable STYLE dimensions, all fully populated across the 6 seasons (verified: min per-season
# non-null = 1.00). QUALITY_CORE scoring stats and the regime-break / late-only columns are excluded.
# NOTE (verified): these 13 carry ~6 effective dimensions, dominated by one possession/passing/dribbling
# -vs-aerial construct -- so this is "~6 independent style dims", not 13 distinct signals.
STYLE_FEATURES = [
    "ball_possession",              # possession
    "pass_accuracy",                # ball retention
    "pass_accurate_long_balls",     # directness (going long)
    "gk_goal_kicks",                # directness (long from the back, skips build-up)
    "duel_aerial_duels_percentage", # aerial reliance
    "pass_accurate_crosses",        # width / crossing
    "pass_final_third_entries",     # territorial penetration
    "att_fouled_in_final_third",    # final-third presence (drawing fouls high)
    "def_clearances",               # low-block / reactive defending
    "def_interceptions",            # interception-based defending
    "duel_dribbles_percentage",     # on-ball dribbling
    "fouls",                        # defensive aggression
    "corner_kicks",                 # sustained attacking pressure
]


def _residualize_on_elo(feat_df: pd.DataFrame, elo: np.ndarray) -> pd.DataFrame:
    """Each feature minus its OLS fit on Elo -> the STYLE part, orthogonal to quality."""
    X = np.c_[np.ones(len(feat_df)), np.asarray(elo, float)]
    out = {}
    for f in feat_df.columns:
        y = feat_df[f].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out[f] = y - X @ beta
    return pd.DataFrame(out, index=feat_df.index)


def _silhouette_curve(Z: np.ndarray, kmin=2, kmax=6) -> dict[int, float]:
    return {k: float(silhouette_score(Z, KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(Z)))
            for k in range(kmin, kmax + 1)}


def _matched_noise_silhouette(Z: np.ndarray, k: int, reps=80) -> np.ndarray:
    """Silhouette of k-means on Gaussian noise with the SAME mean/covariance -- the clusterability null.
    If the real silhouette isn't above this, the 'clusters' are no better than slicing structureless noise."""
    rng = np.random.default_rng(SEED)
    mean, cov = Z.mean(0), np.cov(Z.T)
    out = []
    for _ in range(reps):
        Y = rng.multivariate_normal(mean, cov, len(Z))
        out.append(silhouette_score(Y, KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Y)))
    return np.array(out)


def _hopkins(Z: np.ndarray, m_frac=0.2) -> float:
    """Hopkins statistic: ~0.5 = no cluster tendency (uniform), ->1 = clusterable. <0.75 = not clusterable."""
    rng = np.random.default_rng(SEED)
    n = len(Z)
    m = max(20, int(m_frac * n))
    idx = rng.choice(n, m, replace=False)
    rand_pts = rng.uniform(Z.min(0), Z.max(0), size=(m, Z.shape[1]))
    tree = cKDTree(Z)
    u = tree.query(rand_pts, k=1)[0]              # random point -> nearest real point
    w = tree.query(Z[idx], k=2)[0][:, 1]          # real point  -> nearest OTHER real point
    return float(u.sum() / (u.sum() + w.sum()))


def _participation_ratio(eigvals: np.ndarray) -> float:
    """Effective dimensionality: (sum lambda)^2 / sum(lambda^2). 13 features -> how many real dims?"""
    return float(eigvals.sum() ** 2 / (eigvals ** 2).sum())


def run(team_stats_path: Path = DEFAULT_TEAM_STATS) -> dict:
    ts, _ = team_seasons(team_stats_path)
    feats = [f for f in STYLE_FEATURES if f in ts.columns]
    ts = ts.dropna(subset=feats + ["elo_pre"]).reset_index(drop=True)

    # (1) within-season z -> style vs that season's league norm (kills league-wide provider/era drift:
    #     interceptions ~2 SD, clearances/long-balls ~1.7 SD). Without it k-means recovers SEASON not STYLE.
    zs = ts.groupby("season")[feats].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    # (2) residualize on Elo -> the style component at fixed quality
    resid = _residualize_on_elo(zs, ts["elo_pre"].to_numpy())
    Zr = StandardScaler().fit_transform(resid.to_numpy(float))

    pca = PCA(random_state=SEED).fit(Zr)
    pcs = pca.transform(Zr)
    # anchor signs: PC1+ = possession, PC2+ = territorial penetration (else a refresh could flip labels)
    for axis, anchor in ((0, "ball_possession"), (1, "pass_final_third_entries")):
        if pca.components_[axis][feats.index(anchor)] < 0:
            pca.components_[axis] *= -1
            pcs[:, axis] *= -1
    evr = pca.explained_variance_ratio_
    n_pcs = int(np.searchsorted(np.cumsum(evr), 0.90) + 1)
    Zc = pcs[:, :n_pcs]

    # --- clusterability: is there ANY discrete structure, or is it a continuum? ---
    sil_curve = _silhouette_curve(Zc)
    best_k = max(sil_curve, key=sil_curve.get)
    obs_sil = sil_curve[best_k]
    noise = _matched_noise_silhouette(Zc, best_k)
    noise_p = float((noise >= obs_sil).mean())          # p that structureless noise clusters as well
    hopkins = _hopkins(Zc)
    is_continuum = noise_p > 0.05 or hopkins < 0.75
    km = KMeans(n_clusters=best_k, n_init=20, random_state=SEED).fit(Zc)
    ts["cluster"] = km.labels_

    # --- the 2-D style MAP (the real, robust deliverable): name the two axes by their loadings ---
    load = lambda a: dict(sorted(zip(feats, np.round(pca.components_[a], 2)), key=lambda kv: -abs(kv[1])))
    style_map = {
        "PC1_axis": "possession/passing/dribbling (+) <-> aerial/physical (-)",
        "PC2_axis": "territorial penetration (+) <-> deep/reactive clearing (-)",
        "PC1_var": round(float(evr[0]), 3), "PC2_var": round(float(evr[1]), 3),
        "PC1_loadings": load(0), "PC2_loadings": load(1),
        "effective_dimensions": round(_participation_ratio(pca.explained_variance_), 1),
        "cum_var": {k: round(float(np.cumsum(evr)[k - 1]), 2) for k in (2, 5, 9)},
    }

    # --- style-quality orthogonality (the honest replacement for "multiple routes") ---
    # by construction corr(PC, Elo)~0; the live question is whether any style predicts
    # points BEYOND Elo (punches above its rating). Residualize season points on Elo, correlate with PC1.
    pe = np.c_[np.ones(len(ts)), ts["elo_pre"].to_numpy(float)]
    pts_resid = ts["points"].to_numpy(float) - pe @ np.linalg.lstsq(pe, ts["points"].to_numpy(float), rcond=None)[0]
    over_rho, over_p = spearmanr(pcs[:, 0], pts_resid)
    orthogonality = {
        "corr_PC1_elo": round(float(np.corrcoef(pcs[:, 0], ts["elo_pre"])[0, 1]), 3),
        "corr_PC2_elo": round(float(np.corrcoef(pcs[:, 1], ts["elo_pre"])[0, 1]), 3),
        "style_predicts_points_beyond_elo": {  # does the possession axis buy points over your rating?
            "spearman_PC1_vs_points_residual": round(float(over_rho), 3), "p": round(float(over_p), 3),
            "verdict": "no style over/under-performs its Elo" if over_p > 0.05 else "SIGNAL -- investigate"},
    }

    # --- the two poles of the dominant axis (DESCRIPTIVE labels on a continuum, not discovered groups) ---
    zr_df = pd.DataFrame(Zr, columns=feats)
    poles = []
    for c in range(best_k):
        mask = (ts["cluster"] == c).to_numpy()
        means = zr_df[mask].mean().sort_values()
        d = np.linalg.norm(Zc[mask] - km.cluster_centers_[c], axis=1)
        rep = ts[mask].assign(_d=d).nsmallest(4, "_d")
        poles.append({
            "pole": "ball-dominant" if pcs[mask, 0].mean() > 0 else "physical/direct",
            "n": int(mask.sum()), "mean_elo": round(float(ts.loc[mask, "elo_pre"].mean()), 1),
            "high": [(f, round(float(means[f]), 2)) for f in means.index[-3:][::-1]],
            "low": [(f, round(float(means[f]), 2)) for f in means.index[:3]],
            "examples": [f"{r.team_name} {r.season}" for r in rep.itertuples()],
        })
    pole_elo = [p["mean_elo"] for p in poles]

    # --- the robust specific finding: high-Elo teams at the extremes of the possession axis ---
    elite = ts[ts["elo_pre"] >= ts["elo_pre"].quantile(0.75)].assign(pc1=pcs[ts["elo_pre"] >= ts["elo_pre"].quantile(0.75), 0])
    exemplars = {
        "elite_most_off_possession": [f"{r.team_name} {r.season} (PC1 {r.pc1:+.1f})" for r in elite.nsmallest(4, "pc1").itertuples()],
        "elite_most_possession": [f"{r.team_name} {r.season} (PC1 {r.pc1:+.1f})" for r in elite.nlargest(4, "pc1").itertuples()],
        "note": "Atletico is the robust high-Elo NON-possession exemplar (strongly negative PC1 every season).",
    }

    return {
        "n_team_seasons": int(len(ts)), "features": feats,
        "headline": "La Liga team style is a low-dimensional CONTINUUM, not discrete archetypes; at fixed "
                    "quality it is ~orthogonal to greatness (no style makes a team great). Robust exception: "
                    "Atletico, an elite off-possession side.",
        "style_map": style_map,
        "clusterability": {
            "best_k": int(best_k), "best_silhouette": round(obs_sil, 3),
            "silhouette_curve": {k: round(v, 3) for k, v in sil_curve.items()},
            "matched_noise_silhouette_mean": round(float(noise.mean()), 3),
            "matched_noise_p": round(noise_p, 3), "hopkins": round(hopkins, 3),
            "verdict": "CONTINUUM (no discrete archetypes)" if is_continuum else "discrete clusters present",
        },
        "style_quality_orthogonality": orthogonality,
        "poles": poles,
        "pole_elo_gap": round(abs(pole_elo[0] - pole_elo[1]), 1),
        "robust_exemplars": exemplars,
        "season_balance": {  # the within-season-z guard: clusters must not be season tiers
            "min_cluster_per_season": int(pd.crosstab(ts["cluster"], ts["season"]).min().min())},
        "map": [{"team": r.team_name, "season": r.season, "elo": round(float(r.elo_pre), 1),
                 "pc1": round(float(pcs[i, 0]), 2), "pc2": round(float(pcs[i, 1]), 2), "pole": poles[int(r.cluster)]["pole"]}
                for i, r in enumerate(ts.itertuples())],
    }


def _print_md(res: dict) -> None:
    print(f"\n## Style fingerprints -- the honest read (n={res['n_team_seasons']} team-seasons)")
    print(f"\n{res['headline']}\n")
    sm = res["style_map"]
    print(f"STYLE MAP (the robust deliverable): {sm['effective_dimensions']} effective dims; "
          f"cum var {sm['cum_var']}")
    print(f"  PC1 ({sm['PC1_var']*100:.0f}% var) = {sm['PC1_axis']}")
    print("    " + ", ".join(f"{f} {v:+.2f}" for f, v in list(sm["PC1_loadings"].items())[:5]))
    print(f"  PC2 ({sm['PC2_var']*100:.0f}% var) = {sm['PC2_axis']}")
    print("    " + ", ".join(f"{f} {v:+.2f}" for f, v in list(sm["PC2_loadings"].items())[:5]))
    cl = res["clusterability"]
    print(f"\nARE THERE ARCHETYPES? -> {cl['verdict']}")
    print(f"  best k={cl['best_k']} silhouette {cl['best_silhouette']} vs matched-noise {cl['matched_noise_silhouette_mean']} "
          f"(p={cl['matched_noise_p']}); Hopkins {cl['hopkins']} (<0.75 = not clusterable)")
    print(f"  silhouette curve {cl['silhouette_curve']} (declining = continuum, no natural k)")
    o = res["style_quality_orthogonality"]
    print(f"\nSTYLE vs QUALITY: corr(PC1,Elo)={o['corr_PC1_elo']}, corr(PC2,Elo)={o['corr_PC2_elo']} (~0 by design). "
          f"\n  Does style buy points beyond Elo? PC1 vs points|Elo: rho={o['style_predicts_points_beyond_elo']['spearman_PC1_vs_points_residual']} "
          f"(p={o['style_predicts_points_beyond_elo']['p']}) -> {o['style_predicts_points_beyond_elo']['verdict']}")
    print(f"\nTWO POLES of the dominant axis (DESCRIPTIVE cuts of a continuum, not groups; Elo gap {res['pole_elo_gap']}):")
    for p in res["poles"]:
        print(f"  {p['pole']:16} n={p['n']:3d} mean Elo {p['mean_elo']:.0f}  high: "
              + ", ".join(f"{f}{v:+.1f}" for f, v in p["high"]) + "  e.g. " + ", ".join(p["examples"][:3]))
    ex = res["robust_exemplars"]
    print(f"\nROBUST SPECIFIC FINDING -- elite teams at the possession-axis extremes:")
    print(f"  off-possession elite: {', '.join(ex['elite_most_off_possession'])}")
    print(f"  possession elite:     {', '.join(ex['elite_most_possession'])}")
    print(f"  {ex['note']}")


def selfcheck(team_stats_path: Path = DEFAULT_TEAM_STATS) -> None:
    res = run(team_stats_path)
    # data completeness
    assert 90 < res["n_team_seasons"] < 130, f"expected ~120 team-seasons, got {res['n_team_seasons']}"
    # the within-season-z guard: clusters must NOT be season tiers (the confound we removed)
    assert res["season_balance"]["min_cluster_per_season"] >= 2, \
        f"a cluster nearly vanishes in some season ({res['season_balance']['min_cluster_per_season']}) -- season confound"
    # residualization implemented correctly: the map axes are orthogonal to quality
    assert abs(res["style_quality_orthogonality"]["corr_PC1_elo"]) < 0.05, "PC1 not orthogonal to Elo -- residualization broke"
    # the HONEST conclusion is locked in: we report a continuum, not invented archetypes
    assert res["clusterability"]["verdict"].startswith("CONTINUUM"), \
        f"clusterability flipped to discrete -- re-examine before claiming archetypes ({res['clusterability']})"
    print("selfcheck PASSED")
    _print_md(res)


def main() -> None:
    ap = argparse.ArgumentParser(description="Map La Liga team-season style at fixed quality (continuum, not archetypes).")
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
