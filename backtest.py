"""Replay past weeks: did Lockbox pick better players than the lineups you actually set?

For each week, the roster you had that week is rewound to kickoff (that week's
injury report and byes, stats only from earlier weeks, final scores hidden).
Then four ways of setting the lineup are scored with the REAL points:
  - You:              the lineup you actually started
  - ESPN projections: most ESPN-projected points
  - Lockbox medians:  most points by our own medians (same projections, points objective)
  - Lockbox top pick: highest win probability, even when it's a coin flip
  - Lockbox agent:    what the real agent does: starts from YOUR lineup and only switches
                      when the gain is >= its switch margin (or a starter can't play)
and compared with the best lineup possible in hindsight. We judge points and
player choices, not win/loss: the opponent's score is outside the lineup's control.
Weeks where Lockbox scored LESS than you are shown too.
"""
import json
import sys
from datetime import datetime

from agent import SWITCH_MARGIN, describe_swap, flat, legal_lineups, rank_lineups
from data import connect, get_matchup
from sim import Sim, current_lineup, starters

TOP, AGENT = "Lockbox top pick", "Lockbox agent"
STRATEGIES = ["You", "ESPN projections", "Lockbox medians", TOP, AGENT]


def actual(lineup):
    return sum(p.actual for p in flat(lineup))


def replay(league, week):
    m = get_matchup(league, week, historical=True)
    if m.critical:
        return {"week": week, "skipped": "; ".join(m.critical)}
    now = datetime.now()   # no game times in historical mode, so nothing is locked
    lineups = legal_lineups(m.my_team, now, m.slots)
    sim = Sim(m)
    ranked = rank_lineups(sim, lineups)
    win_of = {frozenset(p.id for p in flat(lu)): w for w, lu in ranked}

    yours = current_lineup(m.my_team)
    top_win, top = ranked[0]
    your_win = win_of.get(frozenset(p.id for p in flat(yours)))   # None if your lineup isn't legal
    # the agent's rule: keep your lineup unless the top pick wins by the margin, or yours can't be played
    must_switch = your_win is None or any(p.unavailable for p in flat(yours))
    agent_pick = top if must_switch or top_win - your_win >= SWITCH_MARGIN else yours
    picks = {
        "You": yours,
        "ESPN projections": max(lineups, key=lambda lu: sum(p.projected for p in flat(lu))),
        "Lockbox medians": max(lineups, key=lambda lu: sum(p.median * p.play_prob for p in flat(lu))),
        TOP: top,
        AGENT: agent_pick,
    }
    opp = sum(p.actual for p in starters(m.opponent))
    best_possible = max(actual(lu) for lu in lineups)   # hindsight: the most points this roster could score
    row = {"week": week, "opp": round(opp, 2), "best_possible": round(best_possible, 2), "results": {}}
    for name, lu in picks.items():
        key = frozenset(p.id for p in flat(lu))
        pts = actual(lu)
        row["results"][name] = {
            "points": round(pts, 2),
            "won": pts > opp,
            "pregame_win_prob": win_of.get(key, sim.win_prob(flat(lu))["win_prob"]),
        }
    # what each version changed vs. your lineup, with what each player actually scored
    def swaps(lu):
        return [{"start": i.name, "start_pts": i.actual, "bench": o.name if o else None,
                 "bench_pts": o.actual if o else None} for i, o in describe_swap(lu, yours)]
    row["swaps"] = swaps(agent_pick)
    row["blocked"] = [] if agent_pick is top else swaps(top)   # top-pick swaps the margin held back
    row["gain"] = None if your_win is None else top_win - your_win
    row["forced"] = must_switch
    return row


def pts(r, s):
    return r["results"][s]["points"]


def _change_lines(r, key):
    out = []
    for sw in r[key]:
        if sw["bench"] is None:
            continue
        diff = sw["start_pts"] - sw["bench_pts"]
        out.append((diff, f"wk {r['week']}: start {sw['start']} ({sw['start_pts']:.1f}) over "
                          f"{sw['bench']} ({sw['bench_pts']:.1f}) -> {diff:+.1f}"))
    return out


def report(rows):
    played = [r for r in rows if "skipped" not in r]
    print("\nPoints scored with each lineup (real points; 'best' = best possible lineup in hindsight)")
    print(f"{'week':>4} " + " ".join(f"{s:>17}" for s in STRATEGIES) + f" {'best':>7}  {'agent vs you':>13}")
    for r in rows:
        if "skipped" in r:
            print(f"{r['week']:>4}  skipped: {r['skipped']}")
            continue
        diff = pts(r, AGENT) - pts(r, "You")
        tag = "kept yours" if not r["swaps"] else f"{diff:+.1f}"
        print(f"{r['week']:>4} " + " ".join(f"{pts(r, s):17.1f}" for s in STRATEGIES)
              + f" {r['best_possible']:7.1f}  {tag:>13}")

    best = sum(r["best_possible"] for r in played)
    print(f"\nTotals over {len(played)} weeks (and share of the best possible {best:.1f} pts):")
    for s in STRATEGIES:
        total = sum(pts(r, s) for r in played)
        print(f"  {s:18} {total:7.1f} pts   {total / best:.1%} of best possible")

    you = sum(pts(r, "You") for r in played)
    for name in (AGENT, TOP):
        lb = sum(pts(r, name) for r in played)
        better = sum(pts(r, name) > pts(r, "You") for r in played)
        worse = [r["week"] for r in played if pts(r, name) < pts(r, "You")]
        same = len(played) - better - len(worse)
        print(f"\n{name} vs. your lineups: {lb - you:+.1f} points ({(lb - you) / len(played):+.1f} per week). "
              f"More in {better} week(s), same in {same}, less in {len(worse)}"
              + (f" (week {', '.join(map(str, worse))})" if worse else "") + ".")

    print(f"\nChanges the agent made (switch margin {SWITCH_MARGIN:.1%}):")
    made = [c for r in played for c in _change_lines(r, "swaps")]
    for diff, line in made:
        print(f"  {line} {'helped' if diff > 0 else 'HURT' if diff < 0 else 'even'}")
    print(f"  => {sum(d > 0 for d, _ in made)} of {len(made)} changes helped, {sum(d < 0 for d, _ in made)} hurt")

    print("\nSwaps the margin held back (too close to call), and how they would have turned out:")
    held = [c for r in played for c in _change_lines(r, "blocked")]
    for diff, line in held:
        print(f"  {line} {'(would have helped: missed)' if diff > 0 else '(would have hurt: avoided)' if diff < 0 else ''}")
    if not held:
        print("  none")
    else:
        print(f"  => guardrail avoided {-sum(d for d, _ in held if d < 0):.1f} pts of bad swaps, "
              f"missed {sum(d for d, _ in held if d > 0):.1f} pts of good ones")

    print(f"\nNot judged on win/loss: the opponent's score is outside the lineup's control. "
          f"{len(played)} weeks is a small sample.")


def main():
    weeks = [int(w) for w in sys.argv[1:]] or list(range(1, 7))
    league = connect()
    rows = []
    for w in weeks:
        print(f"[backtest] replaying week {w}...", flush=True)
        rows.append(replay(league, w))
    report(rows)
    with open("backtest_results.json", "w") as f:
        json.dump(rows, f, indent=1, default=float)


if __name__ == "__main__":
    main()
