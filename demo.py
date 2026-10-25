"""3-minute live demo: real week-7 data, a demo clock, and three SIMULATED events.

   8:00 AM  normal check: agent keeps your lineup (and the real coin-flip alert fires)
   8:15 AM  SIMULATED: a practice report nudges a bench QB up a hair -> agent RESISTS the flip-flop
   8:31 AM  SIMULATED: inactives are out, a starting RB is ruled OUT -> agent swaps + alerts
  10:00 AM  kickoff: hard stop, final report

Usage:
  python3 demo.py              fetch live data (saves a snapshot for offline use)
  python3 demo.py --offline    use the saved snapshot (no network for data)
  python3 demo.py --fast       no pauses (for testing)
  python3 demo.py --no-discord print alerts instead of sending them
"""
import argparse
import copy
import pickle
import time
from datetime import datetime, timedelta

import alerts
import news
from agent import SWITCH_MARGIN, Agent, flat, ids, legal_lineups, rank_lineups
from data import get_matchup
from sim import Sim, current_lineup

SNAPSHOT = "demo_snapshot.pkl"
DAY = datetime(2026, 10, 25)          # Sunday of week 7
PT_TO_ET = timedelta(hours=0)         # ESPN game times are Pacific, and so is the demo clock
SCHEDULE = [DAY.replace(hour=8), DAY.replace(hour=8, minute=15), DAY.replace(hour=8, minute=31)]
KICKOFF = DAY.replace(hour=10)   # Sunday early slate, Pacific
OUT_PLAYER = "Josh Jacobs"            # the starter "ruled inactive" at 8:31 (inactives ~90 min before kickoff)
NUDGE_PLAYER = "Bryce Young"          # bench QB who gets a small, not-enough bump at 8:15

BOLD, RED, GREEN, YELLOW, DIM, END = "\033[1m", "\033[91m", "\033[92m", "\033[93m", "\033[2m", "\033[0m"


def banner(clock, text, color=YELLOW):
    print(f"\n{color}{BOLD}{'=' * 78}\n  {clock:%I:%M %p}  {text}\n{'=' * 78}{END}\n")


def load(offline):
    if offline:
        with open(SNAPSHOT, "rb") as f:
            m = pickle.load(f)
        print(f"{DIM}(offline: using snapshot saved {datetime.fromtimestamp(m.fetched_at):%a %I:%M %p}){END}")
    else:
        print(f"{DIM}fetching live data from ESPN + nflverse...{END}")
        m = get_matchup()
        with open(SNAPSHOT, "wb") as f:
            pickle.dump(m, f)
    for p in m.my_team + m.opponent:   # kickoffs on the demo clock (Pacific)
        if p.game_time:
            p.game_time = p.game_time + PT_TO_ET
    return m


def by_name(m, name):
    return next(p for p in m.my_team if p.name == name)


def find_nudge(base):
    """Smallest bump to the bench QB that makes him the top pick by LESS than the
    switch margin: the textbook flip-flop the agent should resist. None if not possible."""
    now = SCHEDULE[1]
    for pct in range(1, 40):
        m = copy.deepcopy(base)
        p = by_name(m, NUDGE_PLAYER)
        p.floor, p.median, p.ceiling = (v * (1 + pct / 100) for v in (p.floor, p.median, p.ceiling))
        sim = Sim(m)
        ranked = rank_lineups(sim, legal_lineups(m.my_team, now, m.slots))
        cur = ids(current_lineup(m.my_team))
        cur_win = next(w for w, lu in ranked if ids(lu) == cur)
        top_win, top = ranked[0]
        if p.id in ids(top) and 0.003 < top_win - cur_win < SWITCH_MARGIN * 0.8:
            return pct
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-discord", action="store_true")
    a = ap.parse_args()
    pause = 0 if a.fast else 4

    base = load(a.offline)
    nudge = find_nudge(base)
    out_id = by_name(base, OUT_PLAYER).id

    def scenario(n):
        """The matchup the agent 'fetches' on loop n."""
        m = copy.deepcopy(base)
        m.fetched_at = time.time()
        if n >= 2 and nudge:
            p = by_name(m, NUDGE_PLAYER)
            p.floor, p.median, p.ceiling = (v * (1 + nudge / 100) for v in (p.floor, p.median, p.ceiling))
        if n >= 3:
            p = next(p for p in m.my_team if p.id == out_id)
            p.injury, p.play_prob = "OUT", 0.0
        return m

    clock = {"t": SCHEDULE[0]}
    step = {"n": 0}
    events = {
        2: (f"SIMULATED practice report: {NUDGE_PLAYER} full participant, projection +{nudge}%"
            if nudge else "routine re-check (no news)", YELLOW),
        3: (f"SIMULATED inactives list: {OUT_PLAYER} ruled OUT", RED),
    }

    def fetch():
        step["n"] += 1
        n = step["n"]
        if n == 1:
            banner(clock["t"], "Lockbox wakes up: real week-7 matchup, 20,000 simulations per check", GREEN)
        elif n in events:
            banner(clock["t"], *events[n])
        time.sleep(pause / 2)
        return scenario(n)

    def sleep(_seconds):
        time.sleep(pause)
        i = step["n"]
        clock["t"] = SCHEDULE[i] if i < len(SCHEDULE) else KICKOFF
        if clock["t"] == KICKOFF:
            banner(KICKOFF, "KICKOFF: lineup locks, agent stops", GREEN)

    def label(text):
        # demo alerts land in the same Discord channel as real ones: never let them be mistaken
        title, _, body = text.partition("\n")
        note = (f"🧪 Demo scenario: real week-7 data, staged events "
                f"({OUT_PLAYER} is not actually ruled out).")
        return f"🧪 DEMO · {title}\n{body}\n\n{note}" if body else f"🧪 DEMO · {title}\n{note}"

    if a.no_discord:
        send = lambda text: print(f"{BOLD}[ALERT]{END} {alerts._plain(label(text))}")
    else:
        send = lambda text: alerts.send_alert(label(text))
    print(f"{BOLD}LOCKBOX demo{END}: real data, demo clock (Pacific), simulated events are labeled SIMULATED.")
    Agent(fetch=fetch, send=send, clock=lambda: clock["t"], sleep=sleep,
          stop_at=KICKOFF, log_file="demo_log.jsonl").run()
    print(f"\n{DIM}LLM spend this run: ${news.spend():.4f}{END}")


if __name__ == "__main__":
    main()
