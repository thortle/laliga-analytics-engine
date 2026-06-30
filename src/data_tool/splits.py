"""
Chronological splits with a QUARANTINED final season -- so we can experiment with as many
features as we like WITHOUT the one real cost of repeated testing: multiple comparisons / the
garden of forking paths on an underpowered holdout (the exact mechanism behind the betting
chapter's phantom +9.67%). Each new feature scored against the SAME test set inflates the
family-wise false-positive rate; reusing a test set 20 times guarantees a fluke "winner".

Policy (use this for ALL new feature exploration):
  - DEV (explore freely): train = 20/21-23/24, validation = 24/25. Try anything here, as many
    times as you want, select features/windows/hyperparams ONLY on these.
  - SEALED final: 25/26 -- touch ONCE, for the single final confirmation of whatever survived DEV.
    `sealed_final()` refuses to return it without an explicit one-shot flag.

CAVEAT (honesty): 25/26 was already used in aggregate by earlier evaluations, so it is
*semi*-sealed -- the truly-pristine final test is the next live season (26/27) when it exists.
From here on, new exploration must stay in DEV; 25/26 is reserved.

    python -m src.data_tool.splits --selfcheck
"""
from __future__ import annotations

import argparse

import pandas as pd

SEASONS = ["20/21", "21/22", "22/23", "23/24", "24/25", "25/26"]
DEV_TRAIN = ["20/21", "21/22", "22/23", "23/24"]   # fit on these
DEV_VAL = ["24/25"]                                 # select/compare on this (feature choice, windows)
SEALED = ["25/26"]                                  # final confirmation -- touch ONCE


def dev_split(df: pd.DataFrame, season_col: str = "season") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, validation) for free, repeated exploration. Never returns the sealed final set."""
    return df[df[season_col].isin(DEV_TRAIN)].copy(), df[df[season_col].isin(DEV_VAL)].copy()


def sealed_final(df: pd.DataFrame, season_col: str = "season",
                 i_am_running_the_one_shot_final: bool = False) -> pd.DataFrame:
    """The SEALED final set (25/26). Refuses unless you explicitly confirm the one-shot.
    Explore on dev_split(); only call this once, at the very end, for the feature(s) that survived."""
    if not i_am_running_the_one_shot_final:
        raise RuntimeError(
            "SEALED final set (25/26) is quarantined. Explore on dev_split() instead. "
            "Pass i_am_running_the_one_shot_final=True ONLY for the single, final confirmation.")
    print("WARNING: touching the SEALED final set (25/26). This should happen ONCE, at the very end.")
    return df[df[season_col].isin(SEALED)].copy()


def selfcheck() -> None:
    assert set(DEV_TRAIN) | set(DEV_VAL) | set(SEALED) == set(SEASONS), "splits must cover all seasons"
    assert not (set(DEV_TRAIN) & set(DEV_VAL)) and not (set(DEV_VAL) & set(SEALED)), "splits must be disjoint"
    df = pd.DataFrame({"season": SEASONS * 10})
    tr, va = dev_split(df)
    assert set(tr.season.unique()) == set(DEV_TRAIN) and set(va.season.unique()) == set(DEV_VAL)
    try:
        sealed_final(df)
        raise AssertionError("sealed_final must refuse without the explicit one-shot flag")
    except RuntimeError:
        pass
    fin = sealed_final(df, i_am_running_the_one_shot_final=True)
    assert set(fin.season.unique()) == set(SEALED)
    print(f"selfcheck PASSED: DEV train {DEV_TRAIN} + val {DEV_VAL}; SEALED {SEALED} (quarantined, one-shot only)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Chronological splits with a quarantined final season.")
    ap.add_argument("--selfcheck", action="store_true")
    ap.parse_args()
    selfcheck()


if __name__ == "__main__":
    main()
