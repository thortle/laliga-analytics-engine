"""TheStatsAPI client + endpoint discovery.

Key loading order: env THESTATSAPI_KEY, else `THESTATSAPI_KEY=...` in the
project-root `.env` (gitignored). The key is NEVER hardcoded or printed.

CLI:
  # interactive: hit any endpoint, pretty-print the JSON
  python src/data_tool/thestatsapi.py get /football/matches date_from=2025-02-01 date_to=2025-02-05
  # sweep candidate endpoints, map what the API actually exposes
  python src/data_tool/thestatsapi.py discover
  python src/data_tool/thestatsapi.py selfcheck   # offline

stdlib only (urllib), no requests/dotenv deps. This is the data-
ingestion client; the discover sweep is how we learn the real schema before
writing any feature code against it.
"""
from __future__ import annotations

import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

BASE = "https://api.thestatsapi.com/api"
ROOT = Path(__file__).resolve().parents[2]

# macOS system Python often can't find root CAs; use certifi's bundle if present.
# (We verify certificates — never disable verification.)
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:  # noqa: BLE001
    _SSL_CTX = ssl.create_default_context()


def load_key() -> str:
    k = os.environ.get("THESTATSAPI_KEY")
    if k:
        return k.strip()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("THESTATSAPI_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("No THESTATSAPI_KEY in env or .env")


# Rate limit is 12 requests / 60s. Track the last response so bulk callers
# self-throttle instead of hitting 429.
_RL = {"remaining": None, "reset": 0}


def _respect_rate_limit():
    import time
    if _RL["remaining"] is not None and _RL["remaining"] <= 1:
        wait = max(0, _RL["reset"] - int(time.time())) + 2
        if wait:
            time.sleep(wait)
        _RL["remaining"] = None


def get(path: str, key: str | None = None, pace: bool = True, **params):
    """GET an endpoint. Returns (status:int|None, headers:dict, body:json|str).

    pace=True self-throttles against the 12/60s limit using the prior response's
    X-Ratelimit headers (set pace=False for one-off calls / discovery sweeps).
    """
    import time
    key = key or load_key()
    if pace:
        _respect_rate_limit()
    url = BASE + path
    params = {k: v for k, v in params.items() if v is not None}
    if params:
        url += "?" + urlencode(params)
    # Cloudflare (error 1010) rejects the default urllib UA; send a real browser one.
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    })
    try:
        with urllib.request.urlopen(req, timeout=40, context=_SSL_CTX) as r:
            headers = dict(r.headers)
            _update_rl(headers)
            return r.status, headers, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        headers = dict(e.headers)
        _update_rl(headers)
        body = e.read().decode()[:800]
        try:
            body = json.loads(body)
        except Exception:
            pass
        return e.code, headers, body
    except Exception as e:  # noqa: BLE001 - discovery, surface anything
        return None, {}, str(e)


def _update_rl(headers: dict):
    import time
    rem = headers.get("X-Ratelimit-Remaining")
    if rem is not None:
        _RL["remaining"] = int(rem)
        _RL["reset"] = int(headers.get("X-Ratelimit-Reset", time.time() + 60))


def shape(obj, depth=2):
    """Compact structural summary: keys + types, first list element only."""
    if isinstance(obj, dict):
        if depth <= 0:
            return f"{{...{len(obj)} keys...}}"
        return {k: shape(v, depth - 1) for k, v in list(obj.items())[:40]}
    if isinstance(obj, list):
        return [shape(obj[0], depth - 1), f"...len={len(obj)}"] if obj else []
    return type(obj).__name__


def rate_limit_headers(headers: dict) -> dict:
    return {k: v for k, v in headers.items()
            if "rate" in k.lower() or "limit" in k.lower() or "remaining" in k.lower()}


# --- La Liga constants ---
LALIGA = "comp_8814"
COPA_DEL_REY = "comp_7915"          # for fatigue/Rest_Diff cup fixtures
# Other competitions La Liga teams appear in (cross-competition fixtures/fatigue).
# Verified by NAME (the API's `confederation` field is unreliable). No team-centric
# matches endpoint exists -> pull each comp's matches and filter by La Liga team_ids.
OTHER_COMPS = {
    "Copa del Rey": "comp_7915", "Supercopa de España": "comp_5843",  # Supercopa: no xG
    "UEFA Champions League": "comp_3498", "UEFA Europa League": "comp_7739",
    "UEFA Conference League": "comp_408698", "Club World Championship": "comp_3872",
}
# year -> season_id. Only 6 seasons exist; xG is populated for 22/23+ only.
LALIGA_SEASONS = {
    "25/26": "sn_7246390", "24/25": "sn_5761468", "23/24": "sn_606099",
    "22/23": "sn_709252", "21/22": "sn_543625", "20/21": "sn_262755",
}
XG_SEASONS = {"22/23", "23/24", "24/25", "25/26"}  # 21/22 & 20/21 return xG=0


def _data(b):
    return b.get("data", b) if isinstance(b, dict) else b


def paginate(path, per_page=100, **params):
    """Yield every row across all pages of a paginated endpoint."""
    page = 1
    while True:
        st, _, b = get(path, page=page, per_page=per_page, **params)
        if st != 200:
            raise RuntimeError(f"{path} HTTP {st}: {str(b)[:200]}")
        yield from _data(b)
        meta = b.get("meta") if isinstance(b, dict) else None
        if not meta or page >= meta.get("total_pages", page):
            return
        page += 1


# Named helpers with the CORRECT discovered paths (stats live at /stats, not
# /statistics; player stats at /player-stats). Saves the next session re-probing.
def matches(competition_id, season_id):
    return list(paginate("/football/matches",
                         competition_id=competition_id, season_id=season_id))


def match_stats(match_id):      # team stats: overview/shots/attack/passes/duels/defending/np_expected_goals
    return _data(get(f"/football/matches/{match_id}/stats")[2])


def player_stats(match_id):     # 46 players: rating, passing, shooting(xg/xa), duels, defending, general
    return _data(get(f"/football/matches/{match_id}/player-stats")[2])


def lineups(match_id):          # formation + starting_xi + substitutes
    return _data(get(f"/football/matches/{match_id}/lineups")[2])


# Candidate endpoints to sweep. We don't know which exist; the sweep tells us.
CANDIDATES = [
    "/football/competitions", "/football/leagues", "/football/seasons",
    "/football/countries", "/football/teams", "/football/players",
    "/football/standings", "/football/matches",
    "/football/odds", "/football/coaches", "/football/venues",
    "/football", "/", "/openapi.json", "/docs",
]


def discover():
    key = load_key()
    print(f"BASE={BASE}\n")
    first_headers = None
    for path in CANDIDATES:
        status, headers, body = get(path, key=key)
        if first_headers is None and headers:
            first_headers = headers
        tag = "OK " if status == 200 else "   "
        print(f"[{tag}] {status} GET {path}")
        if status == 200:
            print("        shape:", json.dumps(shape(body), default=str)[:300])
        elif status and status != 404:
            print("        body:", json.dumps(body, default=str)[:200])
    if first_headers:
        rl = rate_limit_headers(first_headers)
        if rl:
            print("\nrate-limit headers:", rl)
    print("\nNext: drill into the 200s with `get <path> k=v ...` "
          "(esp. find La Liga id + a match with stats).")


def _cli_get(argv):
    path = argv[0]
    params = dict(a.split("=", 1) for a in argv[1:] if "=" in a)
    status, headers, body = get(path, **params)
    print(f"HTTP {status}")
    rl = rate_limit_headers(headers)
    if rl:
        print("rate-limit:", rl)
    print(json.dumps(body, indent=2, default=str)[:6000])


def _selfcheck():
    s = shape({"a": [1, 2, 3], "b": {"c": "x"}})
    assert s == {"a": ["int", "...len=3"], "b": {"c": "str"}}, s
    assert rate_limit_headers({"X-RateLimit-Remaining": "9"}) == {"X-RateLimit-Remaining": "9"}
    print("selfcheck OK")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    if cmd == "discover":
        discover()
    elif cmd == "get":
        _cli_get(sys.argv[2:])
    elif cmd == "selfcheck":
        _selfcheck()
    else:
        sys.exit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
