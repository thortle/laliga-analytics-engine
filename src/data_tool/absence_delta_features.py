"""
Absence-delta feature (pre-registered) -- the
ABSENCE DELTA: how much today's announced XI deviates from the team's own recent norm.

The squad-rating LEVEL is ~57% redundant with Elo (corr 0.755). The orthogonal
"who is missing today" signal lives in the DEVIATION, not the level:

  XIDeviation(team, match) = SquadRating(today's XI)  -  TypicalXI(team's T-1 rolling mean SquadRating)
  AbsenceDelta_Diff        = XIDeviation(home) - XIDeviation(away)

XIDeviation < 0 => weaker-than-usual XI today (absences / rotation). Strictly pre-match: today's
SquadRating uses the announced XI (public ~1h pre-kickoff) x prior player ratings; TypicalXI uses
only the team's PRIOR matches (shift(1)). No fillna.

Usage:
    python -m src.data_tool.absence_delta_features            # build -> CSV
    python -m src.data_tool.absence_delta_features --selfcheck
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_tool.player_features import team_squad_long, PRIOR_WINDOW, MIN_PRIOR

DEFAULT_OUT = Path("data/processed/absence_delta.csv")


def build() -> pd.DataFrame:
    ts = team_squad_long().sort_values(["team_id", "Date", "match_id"]).reset_index(drop=True)
    # TypicalXI = team's T-1 rolling mean of its OWN past squad ratings (the normal-XI baseline)
    shifted = ts.groupby("team_id", group_keys=False)["squad_rating"].shift(1)
    ts["typical_xi"] = (
        shifted.groupby(ts["team_id"], group_keys=False)
        .rolling(PRIOR_WINDOW, min_periods=MIN_PRIOR).mean()
        .reset_index(level=0, drop=True)
    )
    ts["xi_deviation"] = ts["squad_rating"] - ts["typical_xi"]   # <0 = weaker-than-usual XI (absences)

    home = ts[ts["is_home"]][["match_id", "xi_deviation"]].rename(columns={"xi_deviation": "home_xidev"})
    away = ts[~ts["is_home"]][["match_id", "xi_deviation"]].rename(columns={"xi_deviation": "away_xidev"})
    out = home.merge(away, on="match_id", how="inner")
    out["AbsenceDelta_Diff"] = out["home_xidev"] - out["away_xidev"]
    return out[["match_id", "home_xidev", "away_xidev", "AbsenceDelta_Diff"]]


def selfcheck() -> None:
    out = build()
    assert out["match_id"].is_unique, "duplicate match_id"
    cov = out["AbsenceDelta_Diff"].notna().mean()
    assert cov > 0.85, f"AbsenceDelta coverage too low: {cov:.3f}"

    # deviation should be centered near 0 (it's today minus the team's own recent mean)
    dev = pd.concat([out["home_xidev"], out["away_xidev"]]).dropna()
    assert abs(dev.mean()) < 0.1, f"xi_deviation not centered: mean {dev.mean():.3f}"

    # LEAKAGE: a team's FIRST squad-rated match has no prior -> typical_xi NaN -> xi_deviation NaN.
    ts = team_squad_long().sort_values(["team_id", "Date", "match_id"])
    shifted = ts.groupby("team_id")["squad_rating"].shift(1)
    typ = (shifted.groupby(ts["team_id"]).rolling(PRIOR_WINDOW, min_periods=MIN_PRIOR).mean()
           .reset_index(level=0, drop=True))
    first_idx = ts.groupby("team_id").head(1).index
    assert typ.loc[first_idx].isna().all(), "team's first match typical_xi must be NaN (T-1 leak!)"

    print(f"selfcheck PASSED: {len(out)} matches, AbsenceDelta coverage {cov*100:.1f}%, "
          f"xi_deviation mean {dev.mean():+.3f} sd {dev.std():.3f} (range {dev.min():.2f}..{dev.max():.2f})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the absence-delta feature.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    out = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} matches (AbsenceDelta_Diff) -> {args.out}")


if __name__ == "__main__":
    main()
