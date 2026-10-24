"""Settle "can he play?" from the latest news when data sources disagree.

Sources, freshest first:
  - ESPN fantasy news feed (Rotowire notes citing beat reporters / Schefter etc.)
  - ESPN's live athlete status
  - Sleeper's player status
Claude Haiku reads them and returns a verdict; keyword rules take over if the API fails.
"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import requests
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-5-5"
HEADERS = {"User-Agent": "Mozilla/5.0"}
NEWS_URL = "https://site.api.espn.com/apis/fantasy/v2/games/ffl/news/players?playerId={}&limit=5"
ATHLETE_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/athletes/{}"
SLEEPER_URL = "https://api.sleeper.app/v1/players/nfl"

PRICE_PER_M = {"input": 0.10, "output": 0.50}   # claude-haiku-5-5, $ per million tokens

# token counts from every LLM call, so the agent can enforce its budget cap
USAGE = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
LLM_ENABLED = True   # the agent turns this off when the budget cap is hit
_cache = {}          # same inputs -> same verdict, no new LLM call (saves money, avoids flip-flops)


def spend():
    """Dollars spent on LLM calls so far this run."""
    return (USAGE["input_tokens"] * PRICE_PER_M["input"]
            + USAGE["output_tokens"] * PRICE_PER_M["output"]) / 1e6

_sleeper = None
SLEEPER_CACHE = Path(__file__).parent / ".cache" / "sleeper_players.json"
SLEEPER_CACHE_S = 6 * 3600   # Sleeper statuses are a backup source; 6 h old is fine
SLEEPER_DEADLINE_S = 20      # give up on the download after this long


def _get(url, timeout=15):
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _download_sleeper(deadline_s=SLEEPER_DEADLINE_S):
    """Sleeper's ~5 MB all-players file, abandoned if it takes longer than the deadline
    (requests' timeout is per read, so a slow trickle could otherwise take minutes)."""
    start, chunks = time.time(), []
    with requests.get(SLEEPER_URL, headers=HEADERS, timeout=10, stream=True) as r:
        r.raise_for_status()
        for chunk in r.iter_content(64 * 1024):
            chunks.append(chunk)
            if time.time() - start > deadline_s:
                raise TimeoutError(f"Sleeper download over {deadline_s}s")
    players = json.loads(b"".join(chunks))
    # keep only what we use, so the disk cache is small
    return {v["espn_id"]: {"status": v.get("status"), "injury_status": v.get("injury_status")}
            for v in players.values() if v.get("espn_id")}


def _sleeper_by_espn_id():
    """Sleeper status keyed by ESPN id: disk cache (6 h) -> download -> empty (it's only a backup source)."""
    global _sleeper
    if _sleeper is None:
        data = None
        try:
            if time.time() - SLEEPER_CACHE.stat().st_mtime < SLEEPER_CACHE_S:
                data = json.loads(SLEEPER_CACHE.read_text())
        except (OSError, ValueError):
            pass
        if data is None:
            try:
                data = _download_sleeper()
                SLEEPER_CACHE.parent.mkdir(exist_ok=True)
                SLEEPER_CACHE.write_text(json.dumps(data))
            except Exception:
                data = {}
        _sleeper = {int(k): v for k, v in data.items()}
    return _sleeper


def gather(espn_id):
    """Everything we know about one player's availability, with timestamps."""
    out = {"headlines": [], "espn_status": None, "sleeper_status": None}
    # the two ESPN requests don't depend on each other: run them at the same time
    with ThreadPoolExecutor(max_workers=2) as pool:
        news_req = pool.submit(_get, NEWS_URL.format(espn_id))
        status_req = pool.submit(_get, ATHLETE_URL.format(espn_id))
    try:
        for item in news_req.result().get("feed", [])[:5]:
            out["headlines"].append({
                "time": item.get("published") or item.get("lastModified"),
                "source": item.get("type"),
                "text": item.get("headline", ""),
            })
    except Exception as e:
        out["news_error"] = type(e).__name__
    try:
        out["espn_status"] = (status_req.result().get("status") or {}).get("name")
    except Exception:
        pass
    s = _sleeper_by_espn_id().get(espn_id)
    if s:
        out["sleeper_status"] = s.get("status")
        if s.get("injury_status"):
            out["sleeper_status"] += f" ({s['injury_status']})"
    return out


def _ask_claude(name, team, espn_injury, nfl_roster, info):
    client = anthropic.Anthropic()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    news = "\n".join(f"- [{h['time']}] ({h['source']}) {h['text']}" for h in info["headlines"]) or "- none"
    prompt = f"""It is {now}. Will {name} ({team}) play in this week's NFL game?

Sources (they may disagree; newer, more specific reports beat older status fields):
- ESPN fantasy injury tag: {espn_injury}
- Official NFL weekly roster file (can lag by days): {nfl_roster or "unknown"}
- ESPN live athlete status: {info['espn_status'] or "unknown"}
- Sleeper status: {info['sleeper_status'] or "unknown"}
Latest news, newest first:
{news}

Reply with JSON only:
{{"verdict": "PLAYING" | "OUT" | "UNCERTAIN", "play_prob": <0-1>, "reason": "<under 25 words: the deciding fact, who reported it, and when>"}}"""
    msg = client.messages.create(
        model=MODEL, max_tokens=1000,
        output_config={"effort": "low"},  # a short classification: little thinking needed
        messages=[{"role": "user", "content": prompt}],
    )
    USAGE["calls"] += 1
    USAGE["input_tokens"] += msg.usage.input_tokens
    USAGE["output_tokens"] += msg.usage.output_tokens
    text = next(b.text for b in msg.content if getattr(b, "type", "") == "text")
    result = json.loads(text[text.find("{"): text.rfind("}") + 1])
    result["play_prob"] = max(0.0, min(1.0, float(result["play_prob"])))
    result["via"] = "claude"
    return result


OUT_WORDS = r"\bruled out\b|\bwill not play\b|\bwon't play\b|\bplaced on (injured reserve|ir)\b|\bout for\b|\binactive\b"
IN_WORDS = r"\bactivated\b|\bexpected to play\b|\bwill play\b|\bremoved\b.*\b(exempt|list)\b|\bcleared\b|\bno injury designation\b|\bon track to play\b"


def _rules(info):
    """Fallback: the newest headline that clearly says in or out decides."""
    for h in info["headlines"]:
        t = h["text"].lower()
        if re.search(OUT_WORDS, t):
            return {"verdict": "OUT", "play_prob": 0.05, "reason": f"news {h['time']}: {h['text'][:120]}", "via": "rules"}
        if re.search(IN_WORDS, t):
            return {"verdict": "PLAYING", "play_prob": 0.9, "reason": f"news {h['time']}: {h['text'][:120]}", "via": "rules"}
    return {"verdict": "UNCERTAIN", "play_prob": 0.5, "reason": "no clear news either way", "via": "rules"}


def resolve(name, team, espn_id, espn_injury, nfl_roster=None):
    info = gather(espn_id)
    key = (espn_id, espn_injury, nfl_roster, info["espn_status"], info["sleeper_status"],
           tuple(h["time"] for h in info["headlines"]))
    if key in _cache:
        return dict(_cache[key], cached=True)
    if not LLM_ENABLED:
        result = _rules(info)
        result["reason"] += " (LLM off: budget cap reached)"
    else:
        try:
            result = _ask_claude(name, team, espn_injury, nfl_roster, info)
        except Exception as e:
            result = _rules(info)
            result["reason"] += f" (LLM unavailable: {type(e).__name__})"
    result["latest_news_time"] = info["headlines"][0]["time"] if info["headlines"] else None
    if result["via"] == "claude":
        _cache[key] = result
    return result


def resolve_many(cases):
    """cases: list of dicts with name, team, espn_id, espn_injury, nfl_roster. Runs in parallel."""
    _sleeper_by_espn_id()  # load once before the threads start
    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(lambda c: resolve(**c), cases))


if __name__ == "__main__":
    for r in resolve_many([
        dict(name="Josh Jacobs", team="GB", espn_id=4047365, espn_injury="DAY_TO_DAY", nfl_roster="exempt list"),
        dict(name="Jordan Mason", team="MIN", espn_id=4360569, espn_injury="ACTIVE", nfl_roster="reserve/IR"),
    ]):
        print(r)
    print(USAGE, f"${spend():.5f}")
