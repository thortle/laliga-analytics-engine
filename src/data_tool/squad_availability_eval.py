"""
Squad-availability evaluation (pre-registered).
Does the T-1 available-squad rating (announced XI x prior player ratings) add incremental
predictive value over FULL-HISTORY Elo for La Liga 1X2?

Unlike rolling xG (22/23+ only), player `rating` spans all 6 seasons, so Elo and Elo+Squad
train on the SAME full history -- no data-volume handicap, better powered. The decision rule is
the locked TWO-TIER DIRECTIONAL criterion: continue on a consistent positive walk-forward lift
(+ toss-up signal); "proven" only on significance; do NOT gate on significance (underpowered).

Usage:
    python -m src.data_tool.squad_availability_eval             # evaluate -> JSON + markdown
    python -m src.data_tool.squad_availability_eval --selfcheck
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss

from src.data_tool.predictive_dataset import build as build_spine
from src.data_tool.player_features import build as build_squad
from src.data_tool.predictive_rank import (_xgb, _lgbm, _y, rps_vec, bootstrap_rps_diff,
                                           TEST_SEASONS, PRE_HOLDOUT, SEED)

WF_FOLDS = [("22/23", ["20/21", "21/22"]),
            ("23/24", ["20/21", "21/22", "22/23"]),
            ("24/25", ["20/21", "21/22", "22/23", "23/24"]),
            ("25/26", ["20/21", "21/22", "22/23", "23/24", "24/25"])]
ELO = ["Elo_Diff"]
ELO_SQUAD = ["Elo_Diff", "SquadRating_Diff"]


def _data() -> pd.DataFrame:
    spine = build_spine()
    squad = build_squad()[["match_id", "SquadRating_Diff"]]
    df = spine.merge(squad, on="match_id", how="left")
    return df.dropna(subset=["Elo_Diff", "SquadRating_Diff"]).copy()  # both present (fair, full history)


def _fit_score(feats, tr, te):
    ytr, yte = _y(tr), _y(te)
    p_x = _xgb().fit(tr[feats], ytr).predict_proba(te[feats])
    p_l = _lgbm().fit(tr[feats], ytr).predict_proba(te[feats])
    return p_x, p_l


def run() -> dict:
    df = _data()
    train = df[df.season.isin(PRE_HOLDOUT)]
    test = df[df.season.isin(TEST_SEASONS)]
    yte = _y(test)

    # main holdout: full-history Elo vs Elo+Squad (same rows)
    px_elo, pl_elo = _fit_score(ELO, train, test)
    px_sq, pl_sq = _fit_score(ELO_SQUAD, train, test)
    prior = np.bincount(_y(train), minlength=3) / len(train)
    base = np.tile(prior, (len(yte), 1))
    holdout = {
        "n_train": int(len(train)), "n_test": int(len(test)),
        "baseline_rps": round(float(rps_vec(base, yte).mean()), 4),
        "Elo_rps": round(float(rps_vec(px_elo, yte).mean()), 4),
        "Elo+Squad_rps": round(float(rps_vec(px_sq, yte).mean()), 4),
        "Elo_logloss": round(float(log_loss(yte, px_elo, labels=[0, 1, 2])), 4),
        "Elo+Squad_logloss": round(float(log_loss(yte, px_sq, labels=[0, 1, 2])), 4),
        "Elo+Squad_rps_LGBM": round(float(rps_vec(pl_sq, yte).mean()), 4),
        "Elo_rps_LGBM": round(float(rps_vec(pl_elo, yte).mean()), 4),
    }
    contrast = bootstrap_rps_diff(px_sq, px_elo, yte)  # neg => Squad helps

    # walk-forward (expanding) -- does the Squad edge over Elo grow / stay positive?
    wf = {}
    for test_season, train_seasons in WF_FOLDS:
        tr, te = df[df.season.isin(train_seasons)], df[df.season == test_season]
        if len(tr) < 50 or len(te) < 50:
            continue
        yt = _y(te)
        pe, _ = _fit_score(ELO, tr, te)
        ps, _ = _fit_score(ELO_SQUAD, tr, te)
        wf[test_season] = {"n_train": int(len(tr)),
                           "Elo_rps": round(float(rps_vec(pe, yt).mean()), 4),
                           "Elo+Squad_rps": round(float(rps_vec(ps, yt).mean()), 4),
                           "dRPS": round(float(rps_vec(ps, yt).mean() - rps_vec(pe, yt).mean()), 5)}

    # toss-up interaction: band threshold fixed on TRAIN, evaluated on TEST band
    thr = float(train["Elo_Diff"].abs().quantile(1 / 3))
    band = (test["Elo_Diff"].abs() <= thr).to_numpy()
    tossup = {"elo_diff_thr_from_train": round(thr, 1), "n": int(band.sum()),
              "Squad_vs_Elo": bootstrap_rps_diff(px_sq[band], px_elo[band], yte[band])}

    # is SquadRating's predictive importance positive on the holdout?
    fm = _xgb().fit(train[ELO_SQUAD], _y(train))
    pi = permutation_importance(fm, test[ELO_SQUAD], yte, scoring="neg_log_loss",
                                n_repeats=20, random_state=SEED, n_jobs=-1)
    importance = {f: round(float(pi.importances_mean[i]), 4) for i, f in enumerate(ELO_SQUAD)}

    # REDUNDANCY: raw permutation importance double-counts variance Squad shares with Elo.
    # Report corr + the Elo-RESIDUALIZED importance (the honest unique contribution).
    corr = float(np.corrcoef(df["Elo_Diff"], df["SquadRating_Diff"])[0, 1])
    b1, b0 = np.polyfit(train["Elo_Diff"], train["SquadRating_Diff"], 1)   # fit Squad~Elo on TRAIN only
    tr_r = train.assign(SquadResid=train["SquadRating_Diff"] - (b1 * train["Elo_Diff"] + b0))
    te_r = test.assign(SquadResid=test["SquadRating_Diff"] - (b1 * test["Elo_Diff"] + b0))
    fmr = _xgb().fit(tr_r[["Elo_Diff", "SquadResid"]], _y(train))
    pir = permutation_importance(fmr, te_r[["Elo_Diff", "SquadResid"]], yte, scoring="neg_log_loss",
                                 n_repeats=20, random_state=SEED, n_jobs=-1)
    redundancy = {"corr_squad_elo": round(corr, 3), "r2_explained_by_elo": round(corr ** 2, 3),
                  "squad_raw_importance": importance["SquadRating_Diff"],
                  "squad_elo_residualized_importance": round(float(pir.importances_mean[1]), 4)}

    return {"holdout": holdout, "squad_vs_elo_holdout": contrast, "walk_forward": wf,
            "tossup": tossup, "predictive_importance": importance, "redundancy": redundancy}


def _verdict(res: dict) -> str:
    """Apply the LOCKED two-tier criterion VERBATIM.
    The pre-registration has exactly TWO directional tiers -- no 'weak/caution' middle tier (an
    earlier version invented one AFTER seeing the results; pre-registration caught that
    goalpost move). Honest mapping of the locked words:
      PROVEN    = significant lift over Elo.
      CONTINUE  = CONSISTENT positive walk-forward lift (every fold) AND a positive toss-up signal.
      KILL      = sign inconsistent/negative across folds, OR negligible-and-not-improving.
    Anything that fails CONTINUE without being a clean KILL is reported as NOT-PASSED, never 'continue'."""
    dr = [f["dRPS"] for f in res["walk_forward"].values()]
    all_pos = all(d < 0 for d in dr)
    toss_pos = res["tossup"]["Squad_vs_Elo"]["mean_rps_diff"] < 0   # neg => squad helps the band
    sig = res["squad_vs_elo_holdout"]["verdict"] == "richer_better"
    if sig:
        return "PROVEN (significant lift over Elo on the holdout)"
    if all_pos and toss_pos:
        return "CONTINUE (consistent positive walk-forward lift on every fold + positive toss-up signal)"
    reasons = []
    if not all_pos:
        reasons.append("walk-forward sign inconsistent (>=1 fold reversed) = the pre-reg's named KILL trigger")
    if not toss_pos:
        reasons.append("toss-up niche unconfirmed/wrong-sign = the core squad-availability hypothesis failed")
    return "PRE-REGISTERED TEST NOT PASSED (inconclusive / weak-KILL): " + "; ".join(reasons) + \
           ". Any 'continue' from here is a DISCLOSED off-criterion judgment call, not the locked verdict."


def _print_md(res: dict) -> None:
    h = res["holdout"]
    print(f"\n## Squad availability -- available-squad rating vs full-history Elo (train={h['n_train']}, test={h['n_test']})")
    print(f"baseline RPS {h['baseline_rps']}  |  Elo {h['Elo_rps']}  Elo+Squad {h['Elo+Squad_rps']}  "
          f"(LGBM: Elo {h['Elo_rps_LGBM']}  Elo+Squad {h['Elo+Squad_rps_LGBM']})")
    c = res["squad_vs_elo_holdout"]
    up = " [UNDERPOWERED: MDE %.4f]" % c["mde80"] if c.get("underpowered") else ""
    print(f"Squad vs Elo (holdout): dRPS {c['mean_rps_diff']:+.5f} CI {c['ci95']} -> {c['verdict']}{up}")
    print("\nwalk-forward (expanding; neg dRPS = Squad beats Elo):")
    for s, d in res["walk_forward"].items():
        print(f"  test {s} (train n={d['n_train']:>4}): Elo {d['Elo_rps']:.4f}  Elo+Squad {d['Elo+Squad_rps']:.4f}  dRPS {d['dRPS']:+.5f}")
    tu = res["tossup"]
    print(f"\ntoss-up band (|Elo_Diff|<={tu['elo_diff_thr_from_train']} from TRAIN, n={tu['n']}): "
          f"Squad vs Elo dRPS {tu['Squad_vs_Elo']['mean_rps_diff']:+.5f} CI {tu['Squad_vs_Elo']['ci95']} -> {tu['Squad_vs_Elo']['verdict']}")
    print(f"\npredictive importance (raw): " + "  ".join(f"{k} {v:+.4f}" for k, v in res["predictive_importance"].items()))
    rd = res["redundancy"]
    print(f"redundancy: corr(Squad,Elo)={rd['corr_squad_elo']} (R2 {rd['r2_explained_by_elo']} = Elo-redundant); "
          f"Squad importance raw {rd['squad_raw_importance']:+.4f} -> Elo-RESIDUALIZED {rd['squad_elo_residualized_importance']:+.4f} (the honest unique contribution)")
    print(f"\nLOCKED-CRITERION VERDICT: {_verdict(res)}")


def selfcheck() -> None:
    res = run()
    assert res["holdout"]["Elo_rps"] < res["holdout"]["baseline_rps"], "Elo worse than baseline?!"
    assert set(res["predictive_importance"]) == {"Elo_Diff", "SquadRating_Diff"}
    assert len(res["walk_forward"]) >= 3, "need >=3 walk-forward folds"
    print("selfcheck PASSED")
    _print_md(res)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-registered eval (squad rating vs Elo).")
    ap.add_argument("--out", type=Path, default=Path("data/processed/squad_availability_eval.json"))
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
