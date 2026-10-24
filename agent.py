"""Lockbox agent: watches this week's matchup until kickoff and keeps the lineup
with the best chance to WIN (not the most projected points).

Every loop: fetch fresh data -> simulate -> try every legal lineup -> decide.
Guardrails:
  - flip-flop control: only switch if the new lineup wins by a clear margin; log every resisted switch
  - confidence escalation: if the top two lineups are a coin flip, send ONE alert with a lean
  - stale data: if a fetch fails or is too old, don't act; after repeated failures, stop
  - budget cap: hard limit on LLM spend per run; shown every loop; LLM turns off at the cap
  - hard stop: lock and stop at the last kickoff, then send a short report

ESPN doesn't let apps set lineups, so "switch" means: alert the user to make the swap.
"""
import argparse
import itertools
import json
import time
from datetime import datetime, timedelta

import alerts
import news
from data import NON_STARTING, STARTER_SLOTS, connect, get_matchup
from sim import Sim, current_lineup, starters

SWITCH_MARGIN = 0.015    # a new lineup must win 1.5 points more often to replace the current one
COINFLIP = 0.02          # top two lineups within 2 points -> ask the user
STALE_SECONDS = 30 * 60  # never act on data older than this
MAX_FAILURES = 3         # this many failed fetches in a row -> stuck: alert and stop
BUDGET_USD = 0.25        # hard cap on LLM spend per run
LOOP_SECONDS = 15 * 60
LOG_FILE = "agent_log.jsonl"
FLEX = "RB/WR/TE"
SLOT_ORDER = ["QB", "RB", "WR", "TE", FLEX, "D/ST", "K"]   # display order; other slots go last


# ---------- lineup search ----------

def legal_lineups(players, now, slot_counts=STARTER_SLOTS):
    """Every legal lineup as {slot: [players]}. Players whose game has started are
    frozen: a locked starter keeps his slot, a locked bench player stays benched."""
    fixed = {}
    for p in players:
        if p.locked(now) and p.starting:
            fixed.setdefault(p.slot, []).append(p)
    pool = [p for p in players if not p.locked(now) and not p.unavailable and p.play_prob > 0]

    def options(slot, count, taken):
        locked_here = fixed.get(slot, [])
        need = count - len(locked_here)
        cands = [p for p in pool if slot in p.eligible and p.id not in taken]
        for combo in itertools.combinations(cands, need):
            yield locked_here + list(combo)

    seen, results = set(), []
    locked_ids = {p.id for ps in fixed.values() for p in ps}

    def build(slots, lineup, taken):
        if not slots:
            key = frozenset(p.id for ps in lineup.values() for p in ps)
            if key not in seen:
                seen.add(key)
                results.append({k: list(v) for k, v in lineup.items()})
            return
        slot, rest = slots[0], slots[1:]
        for chosen in options(slot, slot_counts[slot], taken):
            lineup[slot] = chosen
            build(rest, lineup, taken | {p.id for p in chosen})
        lineup.pop(slot, None)

    build(list(slot_counts), {}, set(locked_ids))
    return results


def flat(lineup):
    return [p for ps in lineup.values() for p in ps]


def ids(lineup):
    return frozenset(p.id for p in flat(lineup))


def rank_lineups(sim, lineups):
    opp = sim.total(starters(sim.matchup.opponent))
    scored = []
    for lu in lineups:
        mine = sim.total(flat(lu))
        win = float((mine > opp).mean() + 0.5 * (mine == opp).mean())
        scored.append((win, lu))
    scored.sort(key=lambda x: -x[0])
    return scored


# ---------- explanations ----------

def describe_swap(new, old):
    """Pairs of (player in, player out) between two lineups."""
    ins = [p for p in flat(new) if p.id not in ids(old)]
    outs = [p for p in flat(old) if p.id not in ids(new)]
    pairs = []
    for i in ins:  # pair same positions first
        match = next((o for o in outs if o.position == i.position), outs[0] if outs else None)
        if match:
            outs.remove(match)
        pairs.append((i, match))
    return pairs


def explain(new, old, win_new, win_old):
    """One line per swap, saying why in terms of this matchup."""
    underdog = win_old < 0.5
    lines = []
    for i, o in describe_swap(new, old):
        if o is None:
            lines.append(f"Start {i.name}")
            continue
        if o.unavailable or o.play_prob == 0:
            why = f"{o.name} is {'on bye' if o.on_bye else 'ruled ' + o.injury}"
        elif underdog and i.ceiling > o.ceiling:
            why = f"higher ceiling ({i.ceiling:.0f} vs {o.ceiling:.0f}) and you're the underdog, so upside wins"
        elif not underdog and i.floor > o.floor:
            why = f"safer floor ({i.floor:.0f} vs {o.floor:.0f}) and you're favored, so avoid a dud"
        elif i.play_prob > o.play_prob + 0.05:
            why = f"{o.name} is only {o.play_prob:.0%} to play"
        else:
            why = f"better outcome mix (median {i.median:.0f} vs {o.median:.0f})"
        lines.append(f"Start {i.name} over {o.name}: {why}")
    # players staying in the lineup but changing slots (e.g. FLEX -> RB) are a step in the ESPN app too
    old_slot = {p.id: s for s, ps in old.items() for p in ps}
    for s, ps in new.items():
        for p in ps:
            if p.id in old_slot and old_slot[p.id] != s:
                lines.append(f"Move {p.name} from {_label(old_slot[p.id])} to {_label(s)}")
    return lines, f"win {win_old:.1%} -> {win_new:.1%}"


# ---------- slot-by-slot report ----------

def _label(slot):
    return "FLEX" if slot == FLEX else slot


def _ordered(slots):
    rank = {s: i for i, s in enumerate(SLOT_ORDER)}
    return sorted(slots, key=lambda s: rank.get(s, len(SLOT_ORDER)))


def lineup_report(sim, ranked, current, players, now, slots):
    """The recommended lineup slot by slot (with a reason each), the FLEX decision,
    and every change vs. the lineup currently set in ESPN. ranked = rank_lineups() output."""
    opp = sim.total(starters(sim.matchup.opponent))

    def win(lineup_players):
        t = sim.total(lineup_players)
        return float((t > opp).mean() + 0.5 * (t == opp).mean())

    best_win, best = ranked[0]
    base = flat(best)
    cur_win = win(flat(current))
    underdog = best_win < 0.5
    taken = ids(best)
    pool = [p for p in players if not p.locked(now) and not p.unavailable and p.play_prob > 0]
    bench_eligible = [p for p in pool if p.slot in NON_STARTING]
    excluded = [f"{p.name} ({'bye' if p.on_bye else p.injury.lower()})" for p in players if p.unavailable]

    out = [f"Searched {len(ranked)} legal lineups from your full roster "
           f"({len(bench_eligible)} healthy bench players included). "
           f"You're {'the underdog' if underdog else 'favored'}: win {best_win:.1%}."]
    if excluded:
        out.append(f"Not available: {', '.join(excluded)}")
    out += ["", f"  {'SLOT':6}{'PLAYER':22}{'MED':>6}  REASON"]
    for slot in _ordered(slots):
        for p in best.get(slot, []):
            # his value = best win% WITH him minus best win% of any lineup WITHOUT him
            # (the full search may reshuffle several slots to cover for him)
            reason = f"floor {p.floor:.0f} / ceil {p.ceiling:.0f}"
            alt = next(((w, lu) for w, lu in ranked if p.id not in ids(lu)), None)
            if p.locked(now):
                reason += "; locked (game started)"
            elif alt is None:
                reason += f"; no one else on your roster can fill {_label(slot)}"
            else:
                alt_win, alt_lu = alt
                subs = [q for q in flat(alt_lu) if q.id not in taken]
                q = next((s for s in subs if s.position == p.position), subs[0] if subs else None)
                edge = ""
                if q:
                    edge = (" - higher ceiling, and upside matters as the underdog" if underdog and p.ceiling > q.ceiling
                            else " - safer floor, and you're favored" if not underdog and p.floor > q.floor
                            else "")
                reason += (f"; without him, best is {' + '.join(s.name for s in subs)} "
                           f"at {alt_win:.1%} ({alt_win - best_win:+.1%}){edge}")
            tag = ""
            if p.slot in NON_STARTING:
                tag = "  <-- FROM BENCH"
            elif p.slot != slot:
                tag = f"  <-- moved from {_label(p.slot)}"
            out.append(f"  {_label(slot):6}{p.name[:21]:22}{p.median:6.1f}  {reason}{tag}")

    if FLEX in slots:
        others = taken - {p.id for p in best.get(FLEX, [])}
        cands = sorted(((win([q if x.id in {f.id for f in best[FLEX]} else x for x in base]), q)
                        for q in pool if FLEX in q.eligible and q.id not in others),
                       key=lambda a: -a[0])
        out += ["", "  FLEX decision (every eligible RB/WR/TE, rest of lineup fixed):"]
        for w, q in cands[:6]:
            where = "bench" if q.slot in NON_STARTING else _label(q.slot)
            pick = "  <-- PICK" if q.id in {f.id for f in best[FLEX]} else ""
            out.append(f"    {q.name[:21]:22}{q.position:4} med {q.median:5.1f}  ceil {q.ceiling:5.1f}  "
                       f"win {w:.1%}  (now: {where}){pick}")

    lines, change = explain(best, current, best_win, cur_win)
    moves = [f"Move {p.name} from {_label(p.slot)} to {_label(s)}"
             for s, ps in best.items() for p in ps
             if p.slot not in NON_STARTING and p.slot != s and p.id in ids(current)
             and p.slot != s and not (s in ("RB", "WR") and p.slot == s)]
    out += ["", "  Changes vs. your ESPN lineup:"]
    if not lines and not moves:
        out.append(f"    none: your current lineup is already the best (win {cur_win:.1%})")
    out += [f"    {l}" for l in lines] + [f"    {mv}" for mv in moves]
    if lines:
        out.append(f"    ({change})")
        if best_win - cur_win < SWITCH_MARGIN:
            out.append(f"    Gain is under the {SWITCH_MARGIN:.1%} switch margin: the running agent "
                       f"keeps your lineup (too close to call; this is a coin flip).")
    return "\n".join(out)


# ---------- the agent ----------

class Agent:
    def __init__(self, fetch=get_matchup, send=alerts.send_alert, clock=datetime.now,
                 sleep=time.sleep, stop_at=None, loop_seconds=LOOP_SECONDS, budget=BUDGET_USD,
                 max_loops=None, log_file=LOG_FILE):
        self.fetch, self.send, self.clock, self.sleep = fetch, send, clock, sleep
        self.stop_at, self.loop_seconds, self.budget = stop_at, loop_seconds, budget
        self.max_loops, self.log_file = max_loops, log_file
        self.current = None          # the lineup we're recommending (starts as the user's own)
        self.current_win = None
        self.escalated_for = None    # the lineup we last sent a coin-flip alert about (ONE per lineup)
        self.switched_this_loop = False
        self.failures = 0
        self.loops = 0
        self.stats = {"switches": 0, "resisted": 0, "alerts": 0, "held_stale": 0}
        self.budget_hit = False

    # -- logging --
    def log(self, kind, msg, **extra):
        stamp = self.clock().strftime("%a %H:%M:%S")
        print(f"[{stamp}] {kind.upper():9} {msg}")
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps({"time": stamp, "kind": kind, "msg": msg, **extra}, default=str) + "\n")
        except OSError:
            pass

    def alert(self, text):
        self.stats["alerts"] += 1
        self.send(text)

    # -- guardrails --
    def check_budget(self):
        spent = news.spend()
        if spent >= self.budget and not self.budget_hit:
            self.budget_hit = True
            news.LLM_ENABLED = False
            self.log("budget", f"${spent:.4f} >= ${self.budget:.2f} cap: LLM off, news falls back to keyword rules")
            self.alert(f"⚠️ Lockbox hit its ${self.budget:.2f} LLM budget\n"
                       f"Still running. News checks now use keyword rules instead of the LLM.")
        return spent

    def get_fresh(self):
        """Fetch data; return None (and don't act) if it failed or is stale."""
        try:
            m = self.fetch()
        except Exception as e:
            self.failures += 1
            self.stats["held_stale"] += 1
            self.log("stale", f"fetch failed ({type(e).__name__}: {str(e)[:80]}); holding last decision "
                              f"[{self.failures}/{MAX_FAILURES}]")
            return None
        age = time.time() - m.fetched_at
        if age > STALE_SECONDS:
            self.failures += 1
            self.stats["held_stale"] += 1
            self.log("stale", f"data is {age / 60:.0f} min old; not acting on it [{self.failures}/{MAX_FAILURES}]")
            return None
        if m.critical:
            # e.g. this season's stats failed to download: projections would be quietly wrong
            self.failures += 1
            self.stats["held_stale"] += 1
            self.log("stale", f"incomplete data ({'; '.join(m.critical)}); not acting on it "
                              f"[{self.failures}/{MAX_FAILURES}]")
            return None
        self.failures = 0
        for w in m.warnings:
            self.log("warn", w)
        return m

    def kickoff_deadline(self, m):
        """Hard stop = last kickoff among my players whose games haven't started."""
        if self.stop_at:
            return self.stop_at
        times = [p.game_time for p in m.my_team if p.game_time and p.game_progress == 0]
        return max(times) if times else self.clock()

    # -- one decision --
    def step(self, m):
        now = self.clock()
        self.switched_this_loop = False
        sim = Sim(m)
        lineups = legal_lineups(m.my_team, now, m.slots)
        if not lineups:
            self.log("error", "no legal lineup (not enough healthy players?)")
            return
        ranked = rank_lineups(sim, lineups)
        best_win, best = ranked[0]

        if self.current is None:
            self.current = current_lineup(m.my_team)
            self.log("start", f"{len(lineups)} legal lineups; your current lineup is the baseline")
            print(lineup_report(sim, ranked, self.current, m.my_team, now, m.slots))
        else:
            # re-attach to fresh Player objects (new fetch = new objects, possibly new injuries)
            by_id = {p.id: p for p in m.my_team}
            self.current = {k: [by_id[p.id] for p in v if p.id in by_id] for k, v in self.current.items()}
        cur_ok = all(len(self.current.get(s, [])) == n for s, n in m.slots.items())
        cur_win = next((w for w, lu in ranked if ids(lu) == ids(self.current)), None) if cur_ok else None
        dead = [p for p in flat(self.current) if not p.locked(now) and (p.unavailable or p.play_prob == 0)]

        if cur_win is None or dead:
            reason = f"{', '.join(p.name for p in dead)} can't play" if dead else "current lineup no longer legal"
            if cur_win is None and cur_ok:
                # what the old lineup is worth now, with the out player scoring 0
                cur_win = sim.win_prob(flat(self.current))["win_prob"]
            # change only what we must: the best lineup that keeps every healthy current starter,
            # unless a bigger reshuffle beats it by the full switch margin
            keep = ids(self.current) - {p.id for p in dead}
            minimal = next(((w, lu) for w, lu in ranked if keep <= ids(lu)), None)
            if minimal and best_win - minimal[0] < SWITCH_MARGIN:
                best_win, best = minimal
            lines, change = explain(best, self.current, best_win, cur_win or 0.0)
            self.switch(best, best_win, f"FORCED ({reason})", lines, change)
        elif ids(best) != ids(self.current):
            gain = best_win - cur_win
            lines, change = explain(best, self.current, best_win, cur_win)
            if gain >= SWITCH_MARGIN:
                self.switch(best, best_win, f"+{gain:.1%}", lines, change)
            else:
                self.stats["resisted"] += 1
                self.log("resist", f"resisted switch: {'; '.join(lines)} (only +{gain:.1%}, "
                                   f"needs +{SWITCH_MARGIN:.1%})", gain=gain)
                self.current_win = cur_win
        else:
            self.current_win = cur_win
            self.log("hold", f"keep lineup: win {cur_win:.1%} "
                             f"({'underdog' if cur_win < 0.5 else 'favored'})")

        self.maybe_escalate(sim, ranked)

    def switch(self, lineup, win, why, lines, change):
        self.current, self.current_win = lineup, win
        self.switched_this_loop = True
        self.stats["switches"] += 1
        self.log("switch", f"{why}: {'; '.join(lines)} ({change})")
        forced = why.startswith("FORCED")
        title = (f"🚨 {why[len('FORCED ('):-1]}: make this swap in ESPN" if forced
                 else f"🔁 Better lineup found ({why} win chance): make this swap in ESPN")
        steps = []
        for line in lines:   # "Start X over Y: reason" -> bold action + reason underneath
            action, _, reason = line.partition(": ")
            steps.append(f"**{action}**" + (f"\n› {reason}" if reason else ""))
        before, after = change.removeprefix("win ").split(" -> ")
        self.alert(f"{title}\n" + "\n".join(steps) + f"\n\nWin chance: {before} → **{after}**")

    def maybe_escalate(self, sim, ranked):
        """If the best two lineups are a coin flip, send ONE alert with our lean."""
        if len(ranked) < 2:
            return
        (w1, a), (w2, b) = ranked[0], ranked[1]
        if w1 - w2 > COINFLIP:
            return
        # ONE coin-flip alert per lineup we're recommending, and none on a loop that
        # already sent a swap alert: the user shouldn't get a stream of maybes
        if self.switched_this_loop or self.escalated_for == ids(self.current):
            return
        self.escalated_for = ids(self.current)
        (i, o), *_ = describe_swap(a, b) or [(None, None)]
        if i is None or o is None:
            return
        # lean = how often lineup A outscores lineup B in the same simulated games
        ta, tb = sim.total(flat(a)), sim.total(flat(b))
        lean = float((ta > tb).mean() + 0.5 * (ta == tb).mean())
        text = (f"Coin flip: {i.name} or {o.name}? Leaning {i.name} {lean:.0%}/{1 - lean:.0%} "
                f"(win {w1:.1%} vs {w2:.1%}). Swap in the ESPN app to override.")
        self.log("escalate", text)
        self.alert(f"🤔 Coin flip: {i.name} or {o.name}?\n"
                   f"Leaning **{i.name}**: outscores the other option in {lean:.0%} of simulations.\n"
                   f"Win chance {w1:.1%} vs {w2:.1%}, too close to be sure.\n"
                   f"Swap in the ESPN app if you disagree.")

    # -- main loop --
    def run(self):
        self.log("boot", f"Lockbox up. Switch margin {SWITCH_MARGIN:.1%}, coin flip <{COINFLIP:.0%}, "
                         f"budget ${self.budget:.2f}")
        deadline = None
        while True:
            now = self.clock()
            if deadline and now >= deadline:
                return self.finish("kickoff: lineup locked")
            if self.max_loops is not None and self.loops >= self.max_loops:
                return self.finish(f"stopped after {self.loops} loops")
            self.loops += 1
            spent = self.check_budget()
            m = self.get_fresh()
            if m is None:
                if self.failures >= MAX_FAILURES:
                    self.alert(f"🛑 Lockbox is stuck and stopping\n"
                               f"{self.failures} data failures in a row, so it won't act on bad data. "
                               f"Check your lineup manually.")
                    return self.finish("stuck on bad data")
            else:
                deadline = self.kickoff_deadline(m)
                self.step(m)
                spent = news.spend()
            self.log("status", f"loop {self.loops} | win {self.current_win or 0:.1%} | "
                               f"LLM ${spent:.4f}/${self.budget:.2f} ({news.USAGE['calls']} calls) | "
                               f"stop at {deadline:%a %H:%M}" if deadline else f"loop {self.loops}")
            wait = self.loop_seconds
            if deadline:
                wait = max(0, min(wait, (deadline - self.clock()).total_seconds()))
            if self.max_loops is None or self.loops < self.max_loops:
                self.sleep(wait)

    def finish(self, why):
        lineup = ", ".join(f"{s} {p.name}" for s, ps in (self.current or {}).items() for p in ps)
        s = self.stats
        report = (f"Lockbox final ({why}). Win chance {self.current_win or 0:.1%}. "
                  f"Lineup: {lineup}. {s['switches']} switches, {s['resisted']} resisted, "
                  f"{s['held_stale']} stale holds, LLM spend ${news.spend():.4f}.")
        self.log("stop", report)
        rows = [f"{_label(slot):5} {p.name}" for slot in _ordered(self.current or {})
                for p in (self.current or {})[slot]]
        title = "🔒 Lineup locked at kickoff" if why.startswith("kickoff") else f"🔒 Lockbox stopped ({why})"
        self.send(f"{title}\n"
                  f"Win chance: **{self.current_win or 0:.1%}**\n```\n" + "\n".join(rows) + "\n```\n"
                  f"{s['switches']} switch(es) · {s['resisted']} resisted · {s['held_stale']} stale holds · "
                  f"LLM ${news.spend():.4f}")
        return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loops", type=int, help="stop after N loops (default: run until kickoff)")
    ap.add_argument("--every", type=float, default=LOOP_SECONDS / 60, help="minutes between loops")
    ap.add_argument("--stop-at", help="override hard stop, e.g. '2026-10-25 10:00'")
    ap.add_argument("--budget", type=float, default=BUDGET_USD)
    ap.add_argument("--plan", action="store_true", help="print the recommended lineup once and exit")
    a = ap.parse_args()
    if a.plan:
        m = get_matchup()
        sim, now = Sim(m), datetime.now()
        lineups = legal_lineups(m.my_team, now, m.slots)
        ranked = rank_lineups(sim, lineups)
        print(f"\nLeague slots (from ESPN): {', '.join(f'{_label(s)} x{n}' for s, n in m.slots.items())}")
        if m.critical:
            print("\n!!! DATA INCOMPLETE - do not act on this lineup: " + "; ".join(m.critical) + "\n")
        print(lineup_report(sim, ranked, current_lineup(m.my_team), m.my_team, now, m.slots))
        return
    stop_at = datetime.fromisoformat(a.stop_at) if a.stop_at else None
    league = {}

    def fetch():
        # log in to ESPN once (~9 s) and reuse it; a failed fetch logs in again next loop
        try:
            league.setdefault("l", connect())
            return get_matchup(league["l"])
        except Exception:
            league.clear()
            raise

    Agent(fetch=fetch, stop_at=stop_at, loop_seconds=a.every * 60, budget=a.budget, max_loops=a.loops).run()


if __name__ == "__main__":
    main()
