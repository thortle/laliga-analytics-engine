"""Ingest TheStatsAPI -> raw JSONL (resumable). Cross-competition pull.

Captures RAW API responses verbatim (schema-agnostic, append-only JSONL) so the
scarce, rate-limited API calls are never wasted on a re-fetch when feature
columns change later. Flattening to feature CSVs is a separate offline step
(`flatten`) we can iterate on for free without touching the network.

Commands:
  matches   -> data/raw/thestatsapi/matches.csv (all comps x our 6 years, tagged
               involves_laliga_team). Cheap (paginated list calls only).
  stats [N] -> team_stats.jsonl + player_stats.jsonl for finished matches that
               involve a La Liga team. RESUMABLE: skips match_ids already
               captured. Optional N = stop after N matches (smoke test).
  status    -> progress counts (targets / captured / remaining).
  flatten   -> tidy CSVs (team_match_stats.csv long, player_match_stats.csv).
  selfcheck -> offline logic tests.

JSONL raw-capture decouples the slow rate-limited pull from column
choices; resume is just the done-set read back from the JSONL. No DB, no state
file. Lineups/odds endpoints skipped -- player-stats already carries
started/minutes (derives the XI); odds are not used by the analytics engine.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thestatsapi as api  # noqa: E402

OUT = api.ROOT / "data" / "raw" / "thestatsapi"
MATCHES_CSV = OUT / "matches.csv"
TEAM_JSONL = OUT / "team_stats.jsonl"
PLAYER_JSONL = OUT / "player_stats.jsonl"

# Newest first; xG seasons (22/23+) sort ahead of 21/22 & 20/21 so the most
# valuable data lands first if the pull is interrupted.
SEASON_ORDER = {s: i for i, s in enumerate(
    ["25/26", "24/25", "23/24", "22/23", "21/22", "20/21"])}
YEARS = set(api.LALIGA_SEASONS)  # our 6 covered years, as "YY/YY"

MATCH_FIELDS = [
    "match_id", "competition_id", "comp_name", "season", "season_id", "matchday",
    "stage_name", "status", "utc_date", "home_team_id", "home_team_name",
    "away_team_id", "away_team_name", "score_home", "score_away",
    "xg_available", "odds_available", "involves_laliga_team",
]


def _match_row(m, comp_name, year, involves):
    ht, at = m.get("home_team") or {}, m.get("away_team") or {}
    sc = m.get("score") or {}
    return {
        "match_id": m["id"], "competition_id": m.get("competition_id"),
        "comp_name": comp_name, "season": year, "season_id": m.get("season_id"),
        "matchday": m.get("matchday"), "stage_name": m.get("stage_name"),
        "status": m.get("status"), "utc_date": m.get("utc_date"),
        "home_team_id": ht.get("id"), "home_team_name": ht.get("name"),
        "away_team_id": at.get("id"), "away_team_name": at.get("name"),
        "score_home": sc.get("home"), "score_away": sc.get("away"),
        "xg_available": m.get("xg_available"), "odds_available": m.get("odds_available"),
        "involves_laliga_team": involves,
    }


def cmd_matches():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    # 1) La Liga (all 6 seasons) -> also defines the La Liga team-id universe.
    ll_team_ids = set()
    for year, sid in api.LALIGA_SEASONS.items():
        ms = api.matches(api.LALIGA, sid)
        print(f"La Liga {year}: {len(ms)} matches")
        for m in ms:
            rows.append(_match_row(m, "La Liga", year, True))
            for side in ("home_team", "away_team"):
                t = m.get(side) or {}
                if t.get("id"):
                    ll_team_ids.add(t["id"])
    print(f"La Liga team universe: {len(ll_team_ids)} distinct team ids")

    # 2) Other comps -> keep only matches involving a La Liga team.
    for comp_name, cid in api.OTHER_COMPS.items():
        st, _, b = api.get(f"/football/competitions/{cid}/seasons")
        seasons = api._data(b) if st == 200 else []
        kept_total = 0
        for s in seasons:
            if str(s.get("year")) not in YEARS:
                continue
            ms = api.matches(cid, s["id"])
            kept = 0
            for m in ms:
                hid = (m.get("home_team") or {}).get("id")
                aid = (m.get("away_team") or {}).get("id")
                if hid in ll_team_ids or aid in ll_team_ids:
                    rows.append(_match_row(m, comp_name, str(s.get("year")), True))
                    kept += 1
            kept_total += kept
        print(f"{comp_name}: kept {kept_total} matches involving a La Liga team")

    with MATCHES_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MATCH_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWROTE {MATCHES_CSV} ({len(rows)} rows)")


def _read_matches(path=MATCHES_CSV):
    if not path.exists():
        sys.exit(f"No {path}. Run `matches` first.")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _done_ids(path):
    """Set of match_ids already captured in a JSONL (tolerates blank/bad lines)."""
    ids = set()
    if not path.exists():
        return ids
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["match_id"])
            except Exception:  # noqa: BLE001 - skip corrupt trailing line
                pass
    return ids


def _is_quota_error(resp):
    """True if the API returned the monthly usage-limit error (vs genuine no-data).

    Trial/plan has a MONTHLY request cap separate from the 12/60s rate limit; when
    hit, every endpoint returns {'error': {'code': 'USAGE_LIMIT_EXCEEDED', ...}}.
    Treating this as 'no stats' silently corrupts a pull, so callers must halt."""
    return (isinstance(resp, dict) and isinstance(resp.get("error"), dict)
            and (resp["error"].get("code") == "USAGE_LIMIT_EXCEEDED"
                 or resp["error"].get("status_code") == 429))


def _targets(rows):
    t = [r for r in rows
         if r["status"] == "finished" and r["involves_laliga_team"] == "True"]
    t.sort(key=lambda r: (r["comp_name"] != "La Liga",
                          SEASON_ORDER.get(r["season"], 9),
                          r.get("utc_date") or ""))
    return t


def _fetch_stats(targets, team_jsonl, player_jsonl, limit=None):
    """Resumable team+player stats pull for a list of target match rows.

    Shared by the La Liga (cmd_stats) and international (cmd_intl_stats) paths.
    Skips match_ids already in the JSONL; appends + flushes per match so a kill
    loses nothing. Validates the response shape so transient/no-data responses
    are simply left for a later re-run rather than written as garbage.
    """
    done_t, done_p = _done_ids(team_jsonl), _done_ids(player_jsonl)
    todo = [r for r in targets
            if r["match_id"] not in done_t or r["match_id"] not in done_p]
    if limit:
        todo = todo[:limit]
    print(f"targets={len(targets)} to_fetch={len(todo)}  "
          f"(~{len(todo) * 2} calls, ~{len(todo) * 2 * 5 / 60:.0f} min)")

    team_jsonl.parent.mkdir(parents=True, exist_ok=True)
    no_stats = 0
    with team_jsonl.open("a") as tf, player_jsonl.open("a") as pf:
        for i, r in enumerate(todo, 1):
            mid = r["match_id"]
            try:
                if mid not in done_t:
                    st = api.match_stats(mid)
                    if _is_quota_error(st):
                        print("  !! MONTHLY USAGE LIMIT EXCEEDED — halting. "
                              "Resume after the plan's quota resets (this is NOT no-data; "
                              "uncaptured matches are quota-blocked).")
                        break
                    if isinstance(st, dict) and "overview" in st:
                        tf.write(json.dumps({"match_id": mid, "stats": st}) + "\n")
                        tf.flush()
                        done_t.add(mid)
                    else:
                        no_stats += 1
                if mid not in done_p:
                    ps = api.player_stats(mid)
                    if _is_quota_error(ps):
                        print("  !! MONTHLY USAGE LIMIT EXCEEDED — halting.")
                        break
                    if isinstance(ps, list) and ps:
                        pf.write(json.dumps({"match_id": mid, "players": ps}) + "\n")
                        pf.flush()
                        done_p.add(mid)
            except Exception as e:  # noqa: BLE001 - never lose a multi-hour run
                print(f"  !! {mid} ({r['comp_name']} {r['season']}): {e}")
            if i % 25 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] last={r['comp_name']} {r['season']} "
                      f"{r['utc_date'][:10] if r.get('utc_date') else ''}")
    print(f"DONE. team={len(done_t)} player={len(done_p)} no_stats_responses={no_stats}")


def cmd_stats(limit=None):
    _fetch_stats(_targets(_read_matches()), TEAM_JSONL, PLAYER_JSONL, limit)


def cmd_status():
    rows = _read_matches()
    targets = _targets(rows)
    done_t, done_p = _done_ids(TEAM_JSONL), _done_ids(PLAYER_JSONL)
    by_comp = {}
    for r in targets:
        by_comp[r["comp_name"]] = by_comp.get(r["comp_name"], 0) + 1
    print(f"matches.csv rows: {len(rows)}")
    print(f"targets (finished + La Liga team): {len(targets)}")
    for c, n in sorted(by_comp.items(), key=lambda x: -x[1]):
        print(f"   {c:28s} {n}")
    rem = len([r for r in targets if r["match_id"] not in done_t or r["match_id"] not in done_p])
    print(f"team_stats captured:   {len(done_t)}")
    print(f"player_stats captured: {len(done_p)}")
    print(f"remaining to fetch:    {rem}")


# --- offline flatten: raw JSONL -> tidy feature CSVs (re-runnable, no network) ---
def _flatten_side(stats, side):
    """One team's row from a match_stats dict. Every section is {all,..}->{home,away}."""
    out = {}
    for sec, prefix in [("overview", ""), ("shots", "sh_"), ("attack", "att_"),
                        ("passes", "pass_"), ("duels", "duel_"),
                        ("defending", "def_"), ("goalkeeping", "gk_")]:
        for k, v in (stats.get(sec) or {}).items():
            leaf = v.get("all") if isinstance(v, dict) else None
            if isinstance(leaf, dict):
                out[f"{prefix}{k}"] = leaf.get(side)
    npxg = (stats.get("np_expected_goals") or {}).get("all")
    if isinstance(npxg, dict):
        out["np_expected_goals"] = npxg.get(side)
    return out


_PLAYER_GROUPS = ["passing", "shooting", "duels", "defending", "goalkeeping", "general"]


def _flatten_player(p):
    out = {k: p.get(k) for k in
           ("player_id", "player_name", "team_id", "position", "rating",
            "started", "played", "minutes_played")}
    for g in _PLAYER_GROUPS:
        for k, v in (p.get(g) or {}).items():
            out[f"{g[:4]}_{k}"] = v
    return out


def cmd_flatten(matches_csv=MATCHES_CSV, team_jsonl=TEAM_JSONL,
                player_jsonl=PLAYER_JSONL, out_dir=OUT):
    rows = {r["match_id"]: r for r in _read_matches(matches_csv)}
    # team_match_stats.csv (long: 2 rows/match)
    team_rows = []
    for o in _iter_jsonl(team_jsonl):
        mid, st = o["match_id"], o["stats"]
        m = rows.get(mid, {})
        for side in ("home", "away"):
            base = {
                "match_id": mid, "comp_name": m.get("comp_name"),
                "season": m.get("season"), "utc_date": m.get("utc_date"),
                "is_home": side == "home",
                "team_id": m.get(f"{side}_team_id"),
                "team_name": m.get(f"{side}_team_name"),
                "opp_team_id": m.get(f"{'away' if side == 'home' else 'home'}_team_id"),
                "goals": m.get(f"score_{side}"),
            }
            base.update(_flatten_side(st, side))
            team_rows.append(base)
    _write_csv(out_dir / "team_match_stats.csv", team_rows)

    # player_match_stats.csv (~46 rows/match)
    player_rows = []
    for o in _iter_jsonl(player_jsonl):
        mid = o["match_id"]
        m = rows.get(mid, {})
        for p in o["players"]:
            base = {"match_id": mid, "comp_name": m.get("comp_name"),
                    "season": m.get("season"), "utc_date": m.get("utc_date")}
            base.update(_flatten_player(p))
            player_rows.append(base)
    _write_csv(out_dir / "player_match_stats.csv", player_rows)


# --- International / national-team ingestion (whole-competition, no team filter) ---
INTL_DIR = api.ROOT / "data" / "raw" / "thestatsapi_intl"
INTL_MATCHES_CSV = INTL_DIR / "matches.csv"
INTL_TEAM_JSONL = INTL_DIR / "team_stats.jsonl"
INTL_PLAYER_JSONL = INTL_DIR / "player_stats.jsonl"

# Men's national-team competitions relevant to a World Cup model (verified flags
# 2026-06-25). Non-xG comps still carry team/player stats + odds. Women's/club
# comps deliberately excluded. xg=yes: WC, WC-Qual-UEFA, EURO, Copa America, both
# Nations Leagues, AFCON-Qual, Friendlies. xg=no (stats only): other WC quals,
# EURO-Qual, AFCON, Gold Cup.
NATIONAL_TEAM_COMPS = {
    "FIFA World Cup": "comp_6107",
    "World Cup Qual. UEFA": "comp_2954", "World Cup Qual. CONMEBOL": "comp_4682",
    "World Cup Qual. CONCACAF": "comp_0836", "World Cup Qual. CAF": "comp_5720",
    "World Cup Qual. AFC": "comp_8973", "World Cup Qual. OFC": "comp_7363",
    "EURO": "comp_2949", "EURO Qual.": "comp_3759", "Copa America": "comp_5749",
    "UEFA Nations League": "comp_574977", "CONCACAF Nations League": "comp_193547",
    "Africa Cup of Nations": "comp_1554", "AFCON Qual.": "comp_83579",
    "CONCACAF Gold Cup": "comp_1376", "International Friendly Games": "comp_29967",
}
WC_ONLY = {"FIFA World Cup": "comp_6107"}


def _season_year(year_str):
    y = str(year_str)
    if "/" in y:  # "24/25" -> 2024
        try:
            return 2000 + int(y.split("/")[0])
        except ValueError:
            return 0
    try:
        return int(y)
    except ValueError:
        return 0


def cmd_intl_matches(comps, year_min=0):
    """Build the international matches list for `comps` (whole competition, no
    team filter), keeping seasons with year >= year_min. Rewrites INTL_MATCHES_CSV."""
    INTL_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, cid in comps.items():
        st, _, b = api.get(f"/football/competitions/{cid}/seasons")
        seasons = api._data(b) if st == 200 else []
        kept = 0
        for s in seasons:
            if _season_year(s.get("year")) < year_min:
                continue
            for m in api.matches(cid, s["id"]):
                rows.append(_match_row(m, name, str(s.get("year")), True))
                kept += 1
        print(f"{name}: {kept} matches (seasons >= {year_min})")
    with INTL_MATCHES_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MATCH_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"WROTE {INTL_MATCHES_CSV} ({len(rows)} rows)")


def cmd_intl_stats(limit=None):
    if not INTL_MATCHES_CSV.exists():
        sys.exit("No intl matches.csv. Run intl-matches first.")
    rows = list(csv.DictReader(INTL_MATCHES_CSV.open(newline="")))
    targets = [r for r in rows if r["status"] == "finished"]
    targets.sort(key=lambda r: r.get("utc_date") or "", reverse=True)  # newest first
    _fetch_stats(targets, INTL_TEAM_JSONL, INTL_PLAYER_JSONL, limit)


def cmd_wc():
    """Core convenience: pull the FIFA World Cup (all seasons) end to end."""
    cmd_intl_matches(WC_ONLY, year_min=0)
    cmd_intl_stats()


def _iter_jsonl(path):
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def cmd_audit(matches_csv=MATCHES_CSV, team_jsonl=TEAM_JSONL, player_jsonl=PLAYER_JSONL):
    """Completeness/sanity audit of captured raw JSONL (offline, snapshot-safe).

    Answers the questions that de-risk feature-building: is xG actually dense for
    22/23+? are player rows complete? what's null? does the data agree with scores?
    """
    from collections import defaultdict
    rows = {r["match_id"]: r for r in _read_matches(matches_csv)}

    # --- team stats: per (comp, season) capture + xG density + sanity ---
    cap = defaultdict(lambda: {"team": 0, "xg_nz": 0, "poss_ok": 0, "goals_ok": 0})
    KEY_FIELDS = ["expected_goals", "ball_possession", "passes", "tackles"]
    null_ct = defaultdict(int)
    team_total = 0
    for o in _iter_jsonl(team_jsonl):
        m = rows.get(o["match_id"], {})
        st = o["stats"]
        ov = st.get("overview") or {}
        d = cap[(m.get("comp_name"), m.get("season"))]
        d["team"] += 1
        team_total += 1
        xg = (ov.get("expected_goals") or {}).get("all") or {}
        if (xg.get("home") or 0) or (xg.get("away") or 0):
            d["xg_nz"] += 1
        poss = (ov.get("ball_possession") or {}).get("all") or {}
        ph, pa = poss.get("home"), poss.get("away")
        if ph is not None and pa is not None and 95 <= ph + pa <= 105:
            d["poss_ok"] += 1
        # goals: overview has no goals; compare nothing here (score lives in matches.csv)
        sh, sa = m.get("score_home"), m.get("score_away")
        if sh not in (None, "") and sa not in (None, ""):
            d["goals_ok"] += 1
        for kf in KEY_FIELDS:
            v = (ov.get(kf) or {}).get("all") or {}
            if v.get("home") is None and v.get("away") is None:
                null_ct[kf] += 1

    # --- player stats: rows/match, rating/minutes presence ---
    permatch, rating_present, minutes_present, ptotal, pmatches = [], 0, 0, 0, 0
    for o in _iter_jsonl(player_jsonl):
        ps = o["players"]
        permatch.append(len(ps))
        pmatches += 1
        for p in ps:
            ptotal += 1
            if p.get("rating") is not None:
                rating_present += 1
            if p.get("minutes_played") is not None:
                minutes_present += 1

    print(f"=== CAPTURE + xG DENSITY (team_stats: {team_total} matches) ===")
    print(f"{'comp / season':34s} {'team':>5} {'xG!=0':>6} {'poss_ok':>8}")
    order = sorted(cap, key=lambda k: (k[0] or '', SEASON_ORDER.get(k[1], 9)))
    for k in order:
        d = cap[k]
        t = d["team"] or 1
        print(f"{(str(k[0])+' '+str(k[1])):34s} {d['team']:>5} "
              f"{100*d['xg_nz']/t:>5.0f}% {100*d['poss_ok']/t:>7.0f}%")
    print(f"\n=== TEAM KEY-FIELD NULL RATE (of {team_total}) ===")
    for kf in KEY_FIELDS:
        print(f"  {kf:20s} {100*null_ct[kf]/(team_total or 1):.1f}% null")
    if permatch:
        permatch.sort()
        print(f"\n=== PLAYER STATS ({pmatches} matches, {ptotal} player-rows) ===")
        print(f"  players/match: min={permatch[0]} median={permatch[len(permatch)//2]} max={permatch[-1]}")
        print(f"  rating present:  {100*rating_present/(ptotal or 1):.1f}%")
        print(f"  minutes present: {100*minutes_present/(ptotal or 1):.1f}%")
    # team vs player match-set agreement
    tset = {o["match_id"] for o in _iter_jsonl(team_jsonl)}
    pset = {o["match_id"] for o in _iter_jsonl(player_jsonl)}
    print(f"\n=== COVERAGE PARITY ===")
    print(f"  team-only (no players): {len(tset - pset)}   player-only (no team): {len(pset - tset)}")


def _write_csv(path, rows):
    if not rows:
        print(f"  (no rows for {path.name})")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))  # union, stable order
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"WROTE {path} ({len(rows)} rows, {len(fields)} cols)")


def _selfcheck():
    # done-set parsing tolerates a corrupt trailing line
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write('{"match_id":"mt_1","stats":{}}\n{"match_id":"mt_2"}\n{bad\n')
        tmp = Path(f.name)
    assert _done_ids(tmp) == {"mt_1", "mt_2"}, _done_ids(tmp)
    tmp.unlink()

    # target filter: finished + involves a La Liga team only; sort puts La Liga xG first
    fake = [
        {"match_id": "a", "status": "finished", "involves_laliga_team": "True",
         "comp_name": "La Liga", "season": "24/25", "utc_date": "2025-01-01"},
        {"match_id": "b", "status": "scheduled", "involves_laliga_team": "True",
         "comp_name": "La Liga", "season": "24/25", "utc_date": "2025-02-01"},
        {"match_id": "c", "status": "finished", "involves_laliga_team": "False",
         "comp_name": "UEFA Champions League", "season": "24/25", "utc_date": "2025-01-01"},
        {"match_id": "d", "status": "finished", "involves_laliga_team": "True",
         "comp_name": "UEFA Champions League", "season": "24/25", "utc_date": "2025-01-01"},
    ]
    t = _targets(fake)
    assert [r["match_id"] for r in t] == ["a", "d"], [r["match_id"] for r in t]

    # team flatten pulls the 'all' branch per side
    st = {"overview": {"expected_goals": {"all": {"home": 1.2, "away": 3.49}}},
          "shots": {"total_shots": {"all": {"home": 9, "away": 13}}},
          "np_expected_goals": {"all": {"home": 1.22, "away": 2.73}}}
    h = _flatten_side(st, "home")
    assert h == {"expected_goals": 1.2, "sh_total_shots": 9, "np_expected_goals": 1.22}, h

    # player flatten prefixes nested groups
    p = {"player_id": "pl_1", "rating": 7.3, "started": True,
         "shooting": {"goals": 1, "expected_goals": 0.4}, "general": {"touches": 100}}
    fp = _flatten_player(p)
    assert fp["shoo_goals"] == 1 and fp["gene_touches"] == 100 and fp["rating"] == 7.3, fp
    print("selfcheck OK")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "matches":
        cmd_matches()
    elif cmd == "stats":
        cmd_stats(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif cmd == "status":
        cmd_status()
    elif cmd == "flatten":
        cmd_flatten()
    elif cmd == "audit":
        cmd_audit()
    elif cmd == "wc":
        cmd_wc()
    elif cmd == "intl-matches":
        cmd_intl_matches(NATIONAL_TEAM_COMPS,
                         int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    elif cmd == "intl-stats":
        cmd_intl_stats(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif cmd == "intl-flatten":
        cmd_flatten(INTL_MATCHES_CSV, INTL_TEAM_JSONL, INTL_PLAYER_JSONL, INTL_DIR)
    elif cmd == "intl-audit":
        cmd_audit(INTL_MATCHES_CSV, INTL_TEAM_JSONL, INTL_PLAYER_JSONL)
    elif cmd == "selfcheck":
        _selfcheck()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
