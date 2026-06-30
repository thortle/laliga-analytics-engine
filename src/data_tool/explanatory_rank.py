"""
Explanatory ranking -- which in-match team stats decided the
1X2 outcome of completed La Liga matches? This is EXPLANATION, not prediction:
each match is described by its own (home-minus-away) stat diffs, so there is no
leakage concern. The output ranks candidate stats; the predictive layer
rolls the winners to T-1 and re-tests by predictive importance.

Method (the honest core):
  - Three frames (no fillna): A = full 6 seasons, non-xG diffs only (RAW); B = xG-era
    (22/23+), all diffs (RAW -- shows circular stats out-ranking xG); C = xG-era CLEAN,
    score-reconstructors (OUTCOME_EMBEDDING) removed -- the HONEST drivers table where xG is #1.
  - Chronological holdout per frame (train on earlier seasons, test on the latest 2).
  - Three tree families (XGB / LGBM / RF) for stability -- a driver is trustworthy
    only if the families agree.
  - Importance = PERMUTATION importance on the HELD-OUT test set, scored by
    neg-log-loss (proper score), n_repeats=10. Consensus = mean rank across families.
  - SHAP = XGBoost native TreeSHAP (pred_contribs) -- mean|contrib|, no `shap` pkg.
  - marginal_direction = sign of Spearman corr(feature, goal margin) -- UNCONTROLLED, can flip.
  - partial_r_vs_margin_given_xG = the clean/circular separator: a CLEAN driver adds info beyond
    chance quality (xG); a CIRCULAR symptom only restates the score (gk_goals_prevented = 0.76).
  - outcome_embedding flag = the roll-forward BLACKLIST; collinearity cluster map printed.
  - Deferred: grouped/drop-cluster importance + bootstrap CIs on perm drops (tail past
    rank ~6 is within noise). Add if a later hardening pass needs construct-level credit; the clean
    frame + partial-r already give the honest ranking.

Target: 1X2 (3-class, ordered A<D<H), scored by log-loss + RPS vs a class-prior baseline.

Usage:
    python -m src.data_tool.explanatory_rank              # rank -> JSON + markdown
    python -m src.data_tool.explanatory_rank --selfcheck
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss
from xgboost import XGBClassifier, DMatrix
from lightgbm import LGBMClassifier

DEFAULT_DATA = Path("data/processed/explanatory_match_stats.csv")
DEFAULT_OUT = Path("data/processed/explanatory_ranking.json")
XG_SEASONS = ["22/23", "23/24", "24/25", "25/26"]
TEST_SEASONS = ["24/25", "25/26"]           # chronological holdout (latest 2 seasons)
XG_FEATS = ["expected_goals_diff", "big_chances_diff", "att_big_chances_missed_diff"]
CLASSES = ["A", "D", "H"]                    # ordinal for RPS
SEED = 42

# Stats that mechanically EMBED the realized score (verified):
#   - GK family: goals_prevented = post-shot xG - goals conceded; a save = an on-target
#     shot NOT scored -> both carry goals-against.
#   - SOT / big_chances / shots-inside-box / big-chances-missed: a goal IS an on-target /
#     big chance, so these partially restate goals-for.
#   - def_clearances: lead-protection reverse causation (winners defending a lead clear more).
# They make the model reconstruct the score rather than explain it. Excluded from the CLEAN
# driver frame, and the roll-forward BLACKLIST for the predictive layer (rolling them
# to T-1 would smuggle realized results into "predictive" features).
OUTCOME_EMBEDDING = [
    "gk_goals_prevented_diff", "goalkeeper_saves_diff", "gk_goal_kicks_diff", "gk_high_claims_diff",
    "shots_on_target_diff", "big_chances_diff", "att_big_chances_missed_diff",
    "sh_shots_inside_box_diff", "def_clearances_diff",
]


def rps(proba: np.ndarray, y_int: np.ndarray) -> float:
    """Ranked Probability Score for ordered 1X2 (A<D<H). 0 = perfect, lower better."""
    onehot = np.eye(len(CLASSES))[y_int]
    cp, co = np.cumsum(proba, axis=1), np.cumsum(onehot, axis=1)
    return float(np.mean(np.sum((cp - co) ** 2, axis=1) / (len(CLASSES) - 1)))


def models():
    return {
        "XGB": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                             eval_metric="mlogloss", n_jobs=-1),
        "LGBM": LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                               n_jobs=-1, verbose=-1),
        "RF": RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=5,
                                     random_state=SEED, n_jobs=-1),
    }


def xgb_treeshap(model: XGBClassifier, X: pd.DataFrame) -> dict:
    """Native TreeSHAP mean|contrib| per feature (multiclass), no `shap` package."""
    contribs = model.get_booster().predict(DMatrix(X, feature_names=list(X.columns)),
                                            pred_contribs=True)
    contribs = np.asarray(contribs)
    if contribs.ndim == 3:                    # (n, n_class, n_feat+1)
        mabs = np.abs(contribs[:, :, :-1]).mean(axis=(0, 1))
    else:                                     # (n, n_feat+1) binary-style fallback
        mabs = np.abs(contribs[:, :-1]).mean(axis=0)
    return dict(zip(X.columns, mabs.tolist()))


def partial_r_vs_margin(df: pd.DataFrame, feat: str, ctrl: str) -> float | None:
    """corr(feat, goal margin) after residualising both on `ctrl` -- separates a CLEAN
    driver (adds info beyond chance quality) from a CIRCULAR symptom (only restates it)."""
    if feat == ctrl or ctrl not in df.columns:
        return None
    sub = df[[feat, ctrl, "home_goals", "away_goals"]].dropna()
    if len(sub) < 50:
        return None
    margin = (sub["home_goals"] - sub["away_goals"]).to_numpy(float)
    C = np.c_[np.ones(len(sub)), sub[ctrl].to_numpy(float)]
    res = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    rx, rm = res(sub[feat].to_numpy(float)), res(margin)
    return float(np.corrcoef(rx, rm)[0, 1])


def rank_frame(df: pd.DataFrame, feats: list[str], name: str) -> dict:
    df = df.dropna(subset=feats).copy()
    y = df["FTR"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    is_test = df["season"].isin(TEST_SEASONS).to_numpy()
    Xtr, ytr = df.loc[~is_test, feats], y[~is_test]
    Xte, yte = df.loc[is_test, feats], y[is_test]

    # class-prior baseline (train priors applied to every test row)
    prior = np.bincount(ytr, minlength=len(CLASSES)) / len(ytr)
    base_proba = np.tile(prior, (len(yte), 1))
    baseline = {"log_loss": float(log_loss(yte, base_proba, labels=[0, 1, 2])),
                "rps": rps(base_proba, yte)}

    perm_rank = {f: [] for f in feats}        # per-model rank (1=most important)
    perm_imp = {f: [] for f in feats}         # per-model log-loss increase
    scores, shap = {}, {}
    for mname, model in models().items():
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)
        scores[mname] = {"log_loss": float(log_loss(yte, proba, labels=[0, 1, 2])),
                         "rps": rps(proba, yte),
                         "accuracy": float((proba.argmax(1) == yte).mean())}
        r = permutation_importance(model, Xte, yte, scoring="neg_log_loss",
                                   n_repeats=10, random_state=SEED, n_jobs=-1)
        order = pd.Series(r.importances_mean, index=feats).rank(ascending=False)
        for f in feats:
            perm_rank[f].append(float(order[f]))
            perm_imp[f].append(float(r.importances_mean[feats.index(f)]))
        if mname == "XGB":
            shap = xgb_treeshap(model, Xte)

    # MARGINAL direction: sign of Spearman corr(feature, goal margin) -- uncontrolled, can
    # flip once chance quality is held fixed (e.g. att_big_chances_missed, possession).
    margin = (df["home_goals"] - df["away_goals"]).to_numpy()
    direction = {f: ("+" if spearmanr(df[f], margin).statistic >= 0 else "-") for f in feats}
    # partial-r vs margin | xG (vs SOT for xG itself): the clean/circular separator.
    have_xg = "expected_goals_diff" in feats
    partial = {f: (partial_r_vs_margin(df, f, "shots_on_target_diff" if f == "expected_goals_diff"
                                       else "expected_goals_diff") if have_xg else None) for f in feats}

    consensus = sorted(
        feats,
        key=lambda f: np.mean(perm_rank[f]),   # lower mean rank = more important
    )
    table = [{
        "feature": f,
        "consensus_rank": i + 1,
        "mean_perm_rank": round(float(np.mean(perm_rank[f])), 1),
        "perm_logloss_drop": {m: round(perm_imp[f][j], 4) for j, m in enumerate(models())},
        "mean_perm_logloss_drop": round(float(np.mean(perm_imp[f])), 4),
        "shap_xgb_meanabs": round(float(shap.get(f, 0.0)), 4),
        "marginal_direction": direction[f],
        "partial_r_vs_margin_given_xG": None if partial[f] is None else round(partial[f], 3),
        "outcome_embedding": f in OUTCOME_EMBEDDING,       # circular -> do NOT roll to T-1
        "noise": abs(float(np.mean(perm_imp[f]))) < 0.01,  # drop within noise of zero
    } for i, f in enumerate(consensus)]

    return {"frame": name, "n_train": int((~is_test).sum()), "n_test": int(is_test.sum()),
            "n_features": len(feats), "baseline": baseline, "holdout_scores": scores,
            "ranking": table}


def correlation_clusters(df: pd.DataFrame, feats: list[str], top: list[str]) -> dict:
    """Pearson corr among the top features -- shows where permutation under-credits."""
    sub = df[top].dropna()
    c = sub.corr().abs()
    pairs = [(a, b, round(float(c.loc[a, b]), 2))
             for i, a in enumerate(top) for b in top[i + 1:] if c.loc[a, b] >= 0.5]
    return {"high_corr_pairs(|r|>=0.5)": sorted(pairs, key=lambda x: -x[2])}


def run(data_path: Path = DEFAULT_DATA) -> dict:
    df = pd.read_csv(data_path, low_memory=False)
    era = df[df["season"].isin(XG_SEASONS)].copy()
    all_feats = [c for c in df.columns if c.endswith("_diff")]
    non_xg = [c for c in all_feats if c not in XG_FEATS]
    clean = [c for c in all_feats if c not in OUTCOME_EMBEDDING]  # drop score-reconstructors

    # A = raw full frame (no xG); B = raw xG-era (shows the circular stats out-ranking xG);
    # C = CLEAN xG-era (the HONEST drivers table -- score-reconstructors removed, xG #1).
    frame_a = rank_frame(df, non_xg, "A_full_6seasons_no_xG_RAW")
    frame_b = rank_frame(era, all_feats, "B_xG_era_RAW_incl_circular")
    frame_c = rank_frame(era, clean, "C_xG_era_CLEAN_drivers")

    top_b = [r["feature"] for r in frame_b["ranking"][:12]]
    clusters = correlation_clusters(era, all_feats, top_b)
    return {"frame_C_clean": frame_c, "frame_B_raw": frame_b, "frame_A_full": frame_a,
            "collinearity_frame_B_top12": clusters}


def _print_md(res: dict) -> None:
    for key in ("frame_C_clean", "frame_B_raw", "frame_A_full"):
        fr = res[key]
        print(f"\n### {fr['frame']}  (train={fr['n_train']}, test={fr['n_test']}, feats={fr['n_features']})")
        b = fr["baseline"]
        print(f"baseline (class prior): log-loss {b['log_loss']:.4f}, RPS {b['rps']:.4f}")
        for m, s in fr["holdout_scores"].items():
            print(f"  {m:5} holdout: log-loss {s['log_loss']:.4f}  RPS {s['rps']:.4f}  acc {s['accuracy']:.3f}")
        print(f"{'rank':>4}  {'feature':32} {'mdir':>4} {'pr|xG':>6} {'perm_drop':>10}  flags")
        for r in fr["ranking"][:12]:
            flags = ",".join(f for f, on in [("CIRCULAR", r["outcome_embedding"]), ("noise", r["noise"])] if on)
            pr = "" if r["partial_r_vs_margin_given_xG"] is None else f"{r['partial_r_vs_margin_given_xG']:+.2f}"
            print(f"{r['consensus_rank']:>4}  {r['feature']:32} {r['marginal_direction']:>4} {pr:>6} "
                  f"{r['mean_perm_logloss_drop']:>10.4f}  {flags}")
    print("\n### collinearity (frame B top-12, |r|>=0.5) -- permutation under-credits these:")
    for a, b, r in res["collinearity_frame_B_top12"]["high_corr_pairs(|r|>=0.5)"]:
        print(f"  {r:>4}  {a}  ~  {b}")


def selfcheck(data_path: Path = DEFAULT_DATA) -> None:
    res = run(data_path)
    # CLEAN frame is the honest drivers table: xG must be #1 once score-reconstructors are gone.
    clean = res["frame_C_clean"]["ranking"]
    assert clean[0]["feature"] == "expected_goals_diff", \
        f"clean-frame #1 should be xG, got {clean[0]['feature']}"
    assert clean[0]["mean_perm_logloss_drop"] > 3 * clean[1]["mean_perm_logloss_drop"], \
        "xG should dominate the clean frame (>3x the #2 driver)"
    eg = next(r for r in clean if r["feature"] == "expected_goals_diff")
    assert eg["marginal_direction"] == "+" and not eg["outcome_embedding"], "xG must be a clean + driver"
    # possession is the confirmed null
    poss = next(r for r in res["frame_A_full"]["ranking"] if r["feature"] == "ball_possession_diff")
    assert abs(poss["mean_perm_logloss_drop"]) < 0.01, "possession should be ~noise (confirmed null)"
    print("selfcheck PASSED")
    _print_md(res)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank which in-match stats decide the 1X2 outcome.")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck(args.data)
        return
    res = run(args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    _print_md(res)
    print(f"\nWrote ranking -> {args.out}")


if __name__ == "__main__":
    main()
