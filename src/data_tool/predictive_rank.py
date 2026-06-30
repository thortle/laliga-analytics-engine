"""
Predictive evaluation -- does the clean explanatory driver (rolling xG)
add real PREDICTIVE lift over an Elo-only / Elo+results benchmark, out-of-sample?

The explanatory -> predictive bet: explanatory importance ranks
candidates; PREDICTIVE importance on a chronological holdout decides what ships
We test a nested feature ladder on a shared T-1 frame, score with
RPS + multiclass log-loss (proper scores), and bootstrap the RPS difference for the
key contrasts so the verdict is significance-tested, not a point estimate.

Tree ensembles only: XGB primary, LGBM as a family-robustness check.

Splits (chronological, no leakage):
  - Shared ladder frame = xG-era rows with ALL 9 T-1 features present (xG only 22/23+).
    train = 22/23+23/24, test = 24/25+25/26. Same rows for every rung -> a fair ablation.
  - Elo-only-fullhist benchmark trains on ALL pre-holdout seasons (20/21-23/24) for Elo's
    fair best, but is scored on the SAME shared test rows.

Usage:
    python -m src.data_tool.predictive_rank             # evaluate -> JSON + markdown
    python -m src.data_tool.predictive_rank --selfcheck
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data_tool.predictive_dataset import build, FEATURES, XG_SEASONS

CLASSES = ["A", "D", "H"]
SEED = 42
TEST_SEASONS = ["24/25", "25/26"]
TRAIN_SHARED = ["22/23", "23/24"]
PRE_HOLDOUT = ["20/21", "21/22", "22/23", "23/24"]

# nested feature ladder -- tells the recent-form vs long-horizon-strength story
_ELO = ["Elo_Diff"]
_DWA = _ELO + ["DWA_Goal_Diff", "DWA_SOT_Diff"]
_XGFORM = _DWA + ["xG_Diff", "xGA_Diff", "netxG_Diff"]              # 5-match xG = recent form
_XGSTR = _DWA + ["xGstr_Diff", "xGAstr_Diff", "netxGstr_Diff"]     # long-horizon xG = strength
LADDER = {
    "Elo_only": _ELO,
    "Elo+DWA_form": _DWA,
    "Elo+DWA+xG_form(5m)": _XGFORM,
    "Elo+DWA+xG_strength(long)": _XGSTR,
    "All": FEATURES,
}


def _xgb():
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, random_state=SEED, eval_metric="mlogloss", n_jobs=-1)


def _lgbm():
    # bagging_freq must be set or subsample is silently ignored (LightGBM quirk).
    # Without it the model was fully deterministic across seeds.
    return LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                          bagging_freq=1, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)


def rps_vec(proba: np.ndarray, y_int: np.ndarray) -> np.ndarray:
    """Per-match RPS for ordered 1X2 (A<D<H)."""
    onehot = np.eye(len(CLASSES))[y_int]
    cp, co = np.cumsum(proba, axis=1), np.cumsum(onehot, axis=1)
    return np.sum((cp - co) ** 2, axis=1) / (len(CLASSES) - 1)


def _y(df: pd.DataFrame) -> np.ndarray:
    return df["FTR"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()


def _score(model, Xtr, ytr, Xte, yte) -> dict:
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    return {"log_loss": float(log_loss(yte, proba, labels=[0, 1, 2])),
            "rps": float(rps_vec(proba, yte).mean()),
            "accuracy": float((proba.argmax(1) == yte).mean()),
            "_proba": proba}


def bootstrap_rps_diff(proba_a: np.ndarray, proba_b: np.ndarray, yte: np.ndarray,
                       n: int = 2000) -> dict:
    """95% CI on mean per-match RPS(a) - RPS(b). Negative => a (the richer model) is better."""
    ra, rb = rps_vec(proba_a, yte), rps_vec(proba_b, yte)
    d = ra - rb
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    verdict = "richer_better" if hi < 0 else ("richer_worse" if lo > 0 else "tie")
    # minimum detectable effect at 80% power + n needed for the OBSERVED effect: a "tie" with
    # MDE >> |effect| means UNDERPOWERED ("can't resolve this small"), NOT "no effect".
    sd = float(d.std(ddof=1))
    mde80 = 2.80 * sd / np.sqrt(len(d))                      # (z.975 + z.80) * se
    eff = abs(float(d.mean()))
    n_needed = int((2.80 * sd / eff) ** 2) if eff > 1e-9 else None
    return {"mean_rps_diff": round(float(d.mean()), 5),       # neg => richer model better
            "ci95": [round(float(lo), 5), round(float(hi), 5)], "verdict": verdict,
            "mde80": round(mde80, 5), "n_needed_80": n_needed,
            "underpowered": bool(verdict == "tie" and eff < mde80)}


def run(team_stats_path: Path | None = None) -> dict:
    df = build() if team_stats_path is None else build(team_stats_path)
    shared = df.dropna(subset=FEATURES).copy()              # all 9 features present (=> xG era)
    test = shared[shared.season.isin(TEST_SEASONS)]
    train = shared[shared.season.isin(TRAIN_SHARED)]
    yte, ytr = _y(test), _y(train)

    # class-prior baseline
    prior = np.bincount(ytr, minlength=3) / len(ytr)
    base_proba = np.tile(prior, (len(yte), 1))
    baseline = {"log_loss": float(log_loss(yte, base_proba, labels=[0, 1, 2])),
                "rps": float(rps_vec(base_proba, yte).mean())}

    # nested ladder (XGB + LGBM family-robustness)
    ladder = {}
    proba = {}
    for name, feats in LADDER.items():
        sx = _score(_xgb(), train[feats], ytr, test[feats], yte)
        sl = _score(_lgbm(), train[feats], ytr, test[feats], yte)
        proba[name] = sx.pop("_proba"); sl.pop("_proba")
        ladder[name] = {"n_features": len(feats), "XGB": sx, "LGBM": sl}

    # Elo-only-fullhist benchmark (Elo's fair best: trained on all pre-holdout history)
    full = df[df.season.isin(PRE_HOLDOUT)].dropna(subset=["Elo_Diff"])
    elo_fh = _score(_xgb(), full[["Elo_Diff"]], _y(full), test[["Elo_Diff"]], yte)
    proba["Elo_only_fullhist"] = elo_fh.pop("_proba")
    ladder["Elo_only_fullhist"] = {"n_features": 1, "n_train": int(len(full)), "XGB": elo_fh}

    # significance on the key contrasts (XGB proba)
    contrasts = {
        "xG_form(5m)_adds_over_Elo+DWA": bootstrap_rps_diff(proba["Elo+DWA+xG_form(5m)"], proba["Elo+DWA_form"], yte),
        "xG_strength(long)_adds_over_Elo+DWA": bootstrap_rps_diff(proba["Elo+DWA+xG_strength(long)"], proba["Elo+DWA_form"], yte),
        "xG_strength(long)_adds_over_Elo_only": bootstrap_rps_diff(proba["Elo+DWA+xG_strength(long)"], proba["Elo_only"], yte),
        "best_beats_Elo_fullhist": bootstrap_rps_diff(proba["Elo+DWA+xG_strength(long)"], proba["Elo_only_fullhist"], yte),
    }

    # PREDICTIVE importance on the full model
    fm = _xgb().fit(train[FEATURES], ytr)
    pi = permutation_importance(fm, test[FEATURES], yte, scoring="neg_log_loss",
                                n_repeats=20, random_state=SEED, n_jobs=-1)
    importance = sorted(
        [{"feature": f, "perm_logloss_drop": round(float(pi.importances_mean[i]), 4),
          "std": round(float(pi.importances_std[i]), 4)} for i, f in enumerate(FEATURES)],
        key=lambda r: -r["perm_logloss_drop"])

    # CONDITIONAL pocket (post-hoc, report as promising-not-established): on toss-up matches
    # (|Elo_Diff| small) Elo is weakest -- does xG-strength help most there?
    thr = float(test["Elo_Diff"].abs().quantile(1 / 3))
    tu = test["Elo_Diff"].abs() <= thr
    tossup = {"elo_diff_thr": round(thr, 1), "n": int(tu.sum()),
              "xGstr_vs_Elo": bootstrap_rps_diff(proba["Elo+DWA+xG_strength(long)"][tu.to_numpy()],
                                                 proba["Elo_only"][tu.to_numpy()], yte[tu.to_numpy()])}

    return {"n_train": int(len(train)), "n_test": int(len(test)),
            "baseline": baseline, "ladder": ladder, "contrasts": contrasts,
            "predictive_importance": importance, "tossup_segment": tossup,
            "walk_forward": walk_forward(df)}


def walk_forward(df: pd.DataFrame) -> dict:
    """Expanding-window CV (less variance-prone than one split): does xG-strength's edge over
    Elo_only GROW as training data accrues? Both models train on the SAME shared-frame rows
    (fair ablation), test on each successive season."""
    shared = df.dropna(subset=FEATURES).copy()
    out = {}
    for test_season, train_seasons in [("23/24", ["22/23"]),
                                       ("24/25", ["22/23", "23/24"]),
                                       ("25/26", ["22/23", "23/24", "24/25"])]:
        tr = shared[shared.season.isin(train_seasons)]
        te = shared[shared.season == test_season]
        if len(tr) < 50 or len(te) < 50:
            continue
        ytr2, yte2 = _y(tr), _y(te)
        p_elo = _xgb().fit(tr[_ELO], ytr2).predict_proba(te[_ELO])
        p_str = _xgb().fit(tr[_XGSTR], ytr2).predict_proba(te[_XGSTR])
        out[test_season] = {"n_train": int(len(tr)), "n_test": int(len(te)),
                            "Elo_rps": round(float(rps_vec(p_elo, yte2).mean()), 4),
                            "xGstr_rps": round(float(rps_vec(p_str, yte2).mean()), 4),
                            "dRPS_xGstr_minus_Elo": round(float(rps_vec(p_str, yte2).mean()
                                                               - rps_vec(p_elo, yte2).mean()), 5)}
    return out


def _print_md(res: dict) -> None:
    print(f"\n## Predictive 1X2 (train={res['n_train']}, test={res['n_test']}, chrono holdout)")
    b = res["baseline"]
    print(f"class-prior baseline: log-loss {b['log_loss']:.4f}, RPS {b['rps']:.4f}\n")
    print(f"{'model':26} {'feats':>5} {'XGB_logloss':>12} {'XGB_RPS':>9} {'XGB_acc':>8} {'LGBM_RPS':>9}")
    for name, d in res["ladder"].items():
        lg = f"{d['LGBM']['rps']:.4f}" if "LGBM" in d else "   -"
        print(f"{name:26} {d['n_features']:>5} {d['XGB']['log_loss']:>12.4f} {d['XGB']['rps']:>9.4f} "
              f"{d['XGB']['accuracy']:>8.3f} {lg:>9}")
    print("\nsignificance (bootstrap 95% CI on mean RPS diff; neg => richer model better):")
    for k, c in res["contrasts"].items():
        up = " [UNDERPOWERED: MDE %.4f > |eff|]" % c["mde80"] if c.get("underpowered") else ""
        print(f"  {k:38} dRPS {c['mean_rps_diff']:+.5f} CI {c['ci95']} -> {c['verdict']}{up}")
    wf = res["walk_forward"]
    print("\nwalk-forward (expanding train; does xG-strength's edge over Elo GROW with data?):")
    for s, d in wf.items():
        print(f"  test {s} (train n={d['n_train']:>4}): Elo {d['Elo_rps']:.4f}  xGstr {d['xGstr_rps']:.4f}  "
              f"dRPS {d['dRPS_xGstr_minus_Elo']:+.5f}")
    tu = res["tossup_segment"]
    print(f"\ntoss-up pocket (|Elo_Diff|<={tu['elo_diff_thr']}, n={tu['n']}; POST-HOC, XGB): "
          f"xGstr vs Elo dRPS {tu['xGstr_vs_Elo']['mean_rps_diff']:+.5f} CI {tu['xGstr_vs_Elo']['ci95']} "
          f"-> {tu['xGstr_vs_Elo']['verdict']}")
    print("\npredictive importance (permutation, neg-log-loss on holdout):")
    for r in res["predictive_importance"]:
        print(f"  {r['feature']:18} {r['perm_logloss_drop']:+.4f} +/- {r['std']:.4f}")


def selfcheck(team_stats_path: Path | None = None) -> None:
    res = run(team_stats_path)
    # Elo must carry most of the signal (results-based strength dominates single-match prediction)
    assert res["ladder"]["Elo_only"]["XGB"]["rps"] < res["baseline"]["rps"], "Elo-only worse than prior"
    # the long-horizon strength model must at least beat the class-prior baseline
    assert res["ladder"]["Elo+DWA+xG_strength(long)"]["XGB"]["rps"] < res["baseline"]["rps"], "model worse than prior"
    print("selfcheck PASSED")
    _print_md(res)


def main() -> None:
    ap = argparse.ArgumentParser(description="Predictive evaluation (does T-1 xG add lift over Elo?).")
    ap.add_argument("--out", type=Path, default=Path("data/processed/predictive_eval.json"))
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    res = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    _print_md(res)
    print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()
