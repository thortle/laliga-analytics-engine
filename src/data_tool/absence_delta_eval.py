"""
Absence-delta evaluation (pre-registered). Does the
ABSENCE DELTA (today's XI vs the team's own recent norm) add incremental predictive lift over
full-history Elo -- the orthogonal "who is missing today" signal the squad-rating LEVEL could not
isolate? Applies the SAME locked two-tier directional criterion VERBATIM (no invented tiers).
This is the final feature experiment of the predictive layer.

Usage:
    python -m src.data_tool.absence_delta_eval             # evaluate -> JSON + markdown
    python -m src.data_tool.absence_delta_eval --selfcheck
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
from src.data_tool.absence_delta_features import build as build_absence
from src.data_tool.predictive_rank import (_xgb, _lgbm, _y, rps_vec, bootstrap_rps_diff,
                                           TEST_SEASONS, PRE_HOLDOUT, SEED)
from src.data_tool.squad_availability_eval import _verdict, WF_FOLDS

ELO = ["Elo_Diff"]
ELO_ABS = ["Elo_Diff", "AbsenceDelta_Diff"]
ELO_SQUAD = ["Elo_Diff", "SquadRating_Diff"]
ELO_SQUAD_ABS = ["Elo_Diff", "SquadRating_Diff", "AbsenceDelta_Diff"]
REQUIRED = ["Elo_Diff", "SquadRating_Diff", "AbsenceDelta_Diff"]


def _data() -> pd.DataFrame:
    df = build_spine().merge(build_squad()[["match_id", "SquadRating_Diff"]], on="match_id", how="left")
    df = df.merge(build_absence()[["match_id", "AbsenceDelta_Diff"]], on="match_id", how="left")
    return df.dropna(subset=REQUIRED).copy()


def _rps(feats, tr, te, yte, model=_xgb):
    return rps_vec(model().fit(tr[feats], _y(tr)).predict_proba(te[feats]), yte)


def _proba(feats, tr, te, model=_xgb):
    return model().fit(tr[feats], _y(tr)).predict_proba(te[feats])


def run() -> dict:
    df = _data()
    train, test = df[df.season.isin(PRE_HOLDOUT)], df[df.season.isin(TEST_SEASONS)]
    yte = _y(test)

    p_elo = _proba(ELO, train, test); p_abs = _proba(ELO_ABS, train, test)
    p_sq = _proba(ELO_SQUAD, train, test); p_sqabs = _proba(ELO_SQUAD_ABS, train, test)
    prior = np.bincount(_y(train), minlength=3) / len(train)
    base = np.tile(prior, (len(yte), 1))
    holdout = {
        "n_train": int(len(train)), "n_test": int(len(test)),
        "baseline_rps": round(float(rps_vec(base, yte).mean()), 4),
        "Elo_rps": round(float(rps_vec(p_elo, yte).mean()), 4),
        "Elo+Abs_rps": round(float(rps_vec(p_abs, yte).mean()), 4),
        "Elo+Squad_rps": round(float(rps_vec(p_sq, yte).mean()), 4),
        "Elo+Squad+Abs_rps": round(float(rps_vec(p_sqabs, yte).mean()), 4),
        "Elo+Abs_rps_LGBM": round(float(_rps(ELO_ABS, train, test, yte, _lgbm).mean()), 4),
        "Elo_rps_LGBM": round(float(_rps(ELO, train, test, yte, _lgbm).mean()), 4),
    }
    primary = bootstrap_rps_diff(p_abs, p_elo, yte)                 # Abs vs Elo (PRIMARY)
    secondary = bootstrap_rps_diff(p_sqabs, p_sq, yte)             # Abs beyond the level

    wf = {}
    for test_season, train_seasons in WF_FOLDS:
        tr, te = df[df.season.isin(train_seasons)], df[df.season == test_season]
        if len(tr) < 50 or len(te) < 50:
            continue
        yt = _y(te)
        d = float(_rps(ELO_ABS, tr, te, yt).mean() - _rps(ELO, tr, te, yt).mean())
        wf[test_season] = {"n_train": int(len(tr)), "dRPS": round(d, 5)}

    thr = float(train["Elo_Diff"].abs().quantile(1 / 3))
    band = (test["Elo_Diff"].abs() <= thr).to_numpy()
    tossup = {"elo_diff_thr_from_train": round(thr, 1), "n": int(band.sum()),
              "Abs_vs_Elo": bootstrap_rps_diff(p_abs[band], p_elo[band], yte[band])}

    fm = _xgb().fit(train[ELO_SQUAD_ABS], _y(train))
    pi = permutation_importance(fm, test[ELO_SQUAD_ABS], yte, scoring="neg_log_loss",
                                n_repeats=20, random_state=SEED, n_jobs=-1)
    importance = {f: round(float(pi.importances_mean[i]), 4) for i, f in enumerate(ELO_SQUAD_ABS)}
    redundancy = {"corr_abs_elo": round(float(np.corrcoef(df.Elo_Diff, df.AbsenceDelta_Diff)[0, 1]), 3),
                  "corr_abs_squad": round(float(np.corrcoef(df.SquadRating_Diff, df.AbsenceDelta_Diff)[0, 1]), 3)}

    # verdict via the LOCKED criterion (reuse the squad-availability _verdict; map keys it expects)
    verdict = _verdict({"walk_forward": wf, "tossup": {"Squad_vs_Elo": tossup["Abs_vs_Elo"]},
                        "squad_vs_elo_holdout": primary})
    return {"holdout": holdout, "primary_abs_vs_elo": primary, "secondary_abs_beyond_squad": secondary,
            "walk_forward": wf, "tossup": tossup, "predictive_importance": importance,
            "redundancy": redundancy, "verdict": verdict}


def _print_md(res: dict) -> None:
    h = res["holdout"]
    print(f"\n## Absence delta (today's XI vs own norm) vs Elo (train={h['n_train']}, test={h['n_test']})")
    print(f"baseline {h['baseline_rps']} | Elo {h['Elo_rps']}  Elo+Abs {h['Elo+Abs_rps']}  "
          f"Elo+Squad {h['Elo+Squad_rps']}  Elo+Squad+Abs {h['Elo+Squad+Abs_rps']}  "
          f"(LGBM: Elo {h['Elo_rps_LGBM']} Elo+Abs {h['Elo+Abs_rps_LGBM']})")
    pr, se = res["primary_abs_vs_elo"], res["secondary_abs_beyond_squad"]
    up = " [UNDERPOWERED MDE %.4f]" % pr["mde80"] if pr.get("underpowered") else ""
    print(f"PRIMARY Abs vs Elo:           dRPS {pr['mean_rps_diff']:+.5f} CI {pr['ci95']} -> {pr['verdict']}{up}")
    print(f"SECONDARY Abs beyond +Squad:  dRPS {se['mean_rps_diff']:+.5f} CI {se['ci95']} -> {se['verdict']}")
    print("walk-forward (neg dRPS = Abs beats Elo):")
    for s, d in res["walk_forward"].items():
        print(f"  test {s} (train n={d['n_train']:>4}): dRPS {d['dRPS']:+.5f}")
    tu = res["tossup"]
    print(f"toss-up band (|Elo_Diff|<={tu['elo_diff_thr_from_train']}, n={tu['n']}): "
          f"dRPS {tu['Abs_vs_Elo']['mean_rps_diff']:+.5f} CI {tu['Abs_vs_Elo']['ci95']} -> {tu['Abs_vs_Elo']['verdict']}")
    rd = res["redundancy"]
    print(f"redundancy: corr(Abs,Elo)={rd['corr_abs_elo']} (target: LOW)  corr(Abs,Squad)={rd['corr_abs_squad']}")
    print("predictive importance: " + "  ".join(f"{k} {v:+.4f}" for k, v in res["predictive_importance"].items()))
    print(f"\nLOCKED-CRITERION VERDICT: {res['verdict']}")


def selfcheck() -> None:
    res = run()
    assert res["holdout"]["Elo_rps"] < res["holdout"]["baseline_rps"], "Elo worse than baseline?!"
    assert "AbsenceDelta_Diff" in res["predictive_importance"]
    assert len(res["walk_forward"]) >= 3
    print("selfcheck PASSED")
    _print_md(res)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-registered eval (absence delta vs Elo).")
    ap.add_argument("--out", type=Path, default=Path("data/processed/absence_delta_eval.json"))
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
