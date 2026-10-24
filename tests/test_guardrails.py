"""Guardrail tests for agent.py on made-up players (no network, no LLM).

Run: python3 tests/test_guardrails.py
"""
import copy
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import agent
import news
from data import Matchup, Player

LOG = os.path.join(tempfile.mkdtemp(), "test_log.jsonl")
KICK = datetime(2026, 10, 25, 10, 0)
ELIG = {"QB": ["QB"], "RB": ["RB", "RB/WR/TE"], "WR": ["WR", "RB/WR/TE"], "TE": ["TE", "RB/WR/TE"],
        "D/ST": ["D/ST"], "K": ["K"]}
_id = iter(range(1, 1000))


def P(name, pos, slot, med, spread=6, team=None, **kw):
    return Player(id=next(_id), name=name, position=pos, pro_team=team or name[:3].upper(), opp_team="OPP",
                  slot=slot, eligible=ELIG[pos], injury="ACTIVE", on_bye=False, projected=med,
                  floor=max(0, med - spread), median=med, ceiling=med + 1.8 * spread, game_time=KICK, **kw)


def roster(bench_wr_med=12.0):
    return [P("Qb One", "QB", "QB", 18), P("Qb Two", "QB", "BE", 14),
            P("Rb One", "RB", "RB", 15), P("Rb Two", "RB", "RB", 12), P("Rb Three", "RB", "BE", 8),
            P("Wr One", "WR", "WR", 14), P("Wr Two", "WR", "WR", 12), P("Wr Bench", "WR", "BE", bench_wr_med),
            P("Te One", "TE", "TE", 9), P("Flex Guy", "WR", "RB/WR/TE", 10),
            P("Dst", "D/ST", "D/ST", 7), P("Kick", "K", "K", 8)]


def opponent():
    return [P("O Qb", "QB", "QB", 18), P("O Rb1", "RB", "RB", 14), P("O Rb2", "RB", "RB", 12),
            P("O Wr1", "WR", "WR", 14), P("O Wr2", "WR", "WR", 12), P("O Te", "TE", "TE", 9),
            P("O Flex", "WR", "RB/WR/TE", 10), P("O Dst", "D/ST", "D/ST", 7), P("O K", "K", "K", 8)]


def run(script, loops, start=datetime(2026, 10, 25, 8, 0), **kw):
    """script(loop_number) -> Matchup or raises. Returns (agent, alerts sent)."""
    sent, clock = [], {"t": start}
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return script(calls["n"])

    def sleep(s):
        clock["t"] += timedelta(seconds=s)

    a = agent.Agent(fetch=fetch, send=sent.append, clock=lambda: clock["t"], sleep=sleep,
                    loop_seconds=900, max_loops=loops, log_file=LOG, **kw)
    a.run()
    return a, sent


def M(me, opp=None):
    return Matchup(7, me, opp or opponent(), time.time())


OPP = opponent()
results = []


def check(name, cond):
    results.append((name, cond))
    print(("PASS " if cond else "FAIL ") + name)


# 1. flip-flop: bench WR is a hair better than Flex Guy -> resisted, no switch, no swap alert
base = roster(bench_wr_med=10.3)
a, sent = run(lambda n: M(copy.deepcopy(base), copy.deepcopy(OPP)), loops=3)
check("tiny edge is resisted (no switch)", a.stats["switches"] == 0 and a.stats["resisted"] >= 1)
check("coin-flip alert sent ONCE across 3 loops",
      sum("Coin flip" in s for s in sent) == 1)
print("   ", next(s for s in sent if "Coin flip" in s))

# 2. clear upgrade -> switch with explanation
base = roster(bench_wr_med=17)
a, sent = run(lambda n: M(copy.deepcopy(base), copy.deepcopy(OPP)), loops=2)
check("clear upgrade switches once (not every loop)", a.stats["switches"] == 1)
check("switch alert names the swap", any("Start Wr Bench over" in s for s in sent))

# 3. forced: a starter is ruled OUT on loop 2 -> switch even if margin is small
base3 = roster(bench_wr_med=9)
def script3(n):
    me = copy.deepcopy(base3)
    if n >= 2:
        next(p for p in me if p.name == "Wr Two").injury = "OUT"
    return M(me, copy.deepcopy(OPP))
a, sent = run(script3, loops=3)
check("starter ruled OUT forces exactly one switch, explained",
      a.stats["switches"] == 1 and any("Wr Two can't play" in l for l in open(LOG).read().splitlines()[-30:]))
check("forced lineup no longer has the OUT player", all(p.name != "Wr Two" for p in agent.flat(a.current)))

# 4. stale: fetch fails 3x in a row -> stuck alert and stop
base4 = roster()
def script4(n):
    if n >= 2:
        raise ConnectionError("ESPN down")
    return M(copy.deepcopy(base4), copy.deepcopy(OPP))
a, sent = run(script4, loops=10)
check("3 failed fetches -> stop as stuck", a.loops == 4 and any("stuck" in s for s in sent))

# 5. stale: data older than 30 min is not acted on
old = M(copy.deepcopy(roster(bench_wr_med=17)), copy.deepcopy(OPP))
old.fetched_at -= 3600
a, sent = run(lambda n: old, loops=2)
check("old data -> held, no switch", a.stats["held_stale"] == 2 and a.stats["switches"] == 0)

# 5b. incomplete data (e.g. this season's stats failed to download) -> held, no switch
bad = M(copy.deepcopy(roster(bench_wr_med=17)), copy.deepcopy(OPP))
bad.critical = ["multi-season load failed; using 2024-2025 only"]
a, sent = run(lambda n: bad, loops=2)
check("critical data gap -> held, no switch", a.stats["held_stale"] == 2 and a.stats["switches"] == 0)

# 6. budget cap: spend over cap -> LLM turned off, one alert
news.USAGE.update(input_tokens=10_000_000, output_tokens=0)   # $1.00
a, sent = run(lambda n: M(copy.deepcopy(roster()), copy.deepcopy(OPP)), loops=2, budget=0.25)
check("budget cap turns LLM off and alerts once",
      news.LLM_ENABLED is False and sum("budget" in s for s in sent) == 1)
news.USAGE.update(input_tokens=0, output_tokens=0); news.LLM_ENABLED = True

# 7. hard stop at kickoff, with a final report
a, sent = run(lambda n: M(copy.deepcopy(roster()), copy.deepcopy(OPP)), loops=None,
              start=datetime(2026, 10, 25, 9, 20))
check("stops at kickoff with report", "kickoff" in sent[-1] and a.loops <= 4)

# 8. locked players: a starter whose game is final can't be benched even for a better bench player
base8 = roster(bench_wr_med=25)
def script8(n):
    me = copy.deepcopy(base8)
    flex = next(p for p in me if p.name == "Flex Guy")
    flex.game_progress, flex.points = 100, 3.0
    wr2 = next(p for p in me if p.name == "Wr Two")
    wr2.game_progress, wr2.points = 100, 2.0
    return M(me, copy.deepcopy(OPP))
a, sent = run(script8, loops=1)
names = {p.name for p in agent.flat(a.current)}
check("locked starters stay put; swap uses an unlocked slot",
      "Flex Guy" in names and "Wr Two" in names and "Wr Bench" in names)

print(f"\n{sum(c for _, c in results)}/{len(results)} passed")
