"""Monte Carlo simulation of this week's matchup.

Each player's score is drawn from his floor (20th pct) / median / ceiling (85th pct),
connected by a piecewise-linear curve with tails beyond them. With probability
1 - play_prob he doesn't play and scores 0. Players from the same NFL team move
together a bit (a shootout lifts everyone on that offense), which matters when
both fantasy teams own pieces of the same game.

All players are simulated once up front, and every lineup is scored on the same
draws, so comparing two lineups isn't muddied by random noise.
"""
import numpy as np
from scipy.stats import norm

from data import FLOOR_Q, MEDIAN_Q, CEILING_Q, STARTER_SLOTS, get_matchup

N_SIMS = 20_000
TEAM_CORRELATION = 0.25   # how much teammates' outcomes move together


def _quantile_curve(p):
    """Points (percentile, score) that define a player's outcome distribution."""
    floor, median, ceiling = p.floor, p.median, p.ceiling
    low = max(0.0, floor - 0.6 * (median - floor)) if p.position != "D/ST" else floor - (median - floor)
    high = ceiling + 0.8 * (ceiling - median)   # fantasy scores have a long right tail
    return [0.0, FLOOR_Q, MEDIAN_Q, CEILING_Q, 1.0], [low, floor, median, ceiling, high]


class Sim:
    def __init__(self, matchup, n=N_SIMS, seed=7):
        self.matchup = matchup
        self.players = {p.id: p for p in matchup.my_team + matchup.opponent}
        rng = np.random.default_rng(seed)
        ids = list(self.players)

        # correlated percentiles: shared NFL-team shock + individual noise (Gaussian copula)
        teams = sorted({p.pro_team for p in self.players.values()})
        team_z = {t: rng.standard_normal(n) for t in teams}
        self.scores = {}
        for pid in ids:
            p = self.players[pid]
            z = (np.sqrt(TEAM_CORRELATION) * team_z[p.pro_team]
                 + np.sqrt(1 - TEAM_CORRELATION) * rng.standard_normal(n))
            xs, ys = _quantile_curve(p)
            pts = np.interp(norm.cdf(z), xs, ys)
            plays = rng.random(n) < (p.play_prob if not p.unavailable else 0.0)
            pts = np.where(plays, pts, 0.0)
            if p.game_progress >= 100:
                pts = np.full(n, p.points)          # game over: the score is known
            elif p.game_progress > 0:
                # in progress: points so far + a share of a fresh draw for the rest of the game
                pts = p.points + pts * (1 - p.game_progress / 100)
            self.scores[pid] = pts

    def total(self, lineup):
        """lineup: list of Player (or ids). Returns n simulated totals."""
        return sum(self.scores[getattr(p, "id", p)] for p in lineup)

    def win_prob(self, my_lineup, opp_lineup=None):
        opp_lineup = opp_lineup or starters(self.matchup.opponent)
        mine, theirs = self.total(my_lineup), self.total(opp_lineup)
        win = np.mean(mine > theirs) + 0.5 * np.mean(mine == theirs)
        return {
            "win_prob": float(win),
            "my_median": float(np.median(mine)),
            "opp_median": float(np.median(theirs)),
            "my_p10_p90": (float(np.percentile(mine, 10)), float(np.percentile(mine, 90))),
        }


def starters(players):
    return [p for p in players if p.slot not in ("BE", "IR")]


def check_lineup(lineup, slots=STARTER_SLOTS):
    """Raise if the lineup doesn't fill the league's slots legally. lineup: {slot: [players]}."""
    for slot, need in slots.items():
        got = lineup.get(slot, [])
        if len(got) != need:
            raise ValueError(f"{slot}: need {need}, got {len(got)}")
        for p in got:
            if slot not in p.eligible:
                raise ValueError(f"{p.name} can't play {slot}")
    flat = [p.id for ps in lineup.values() for p in ps]
    if len(flat) != len(set(flat)):
        raise ValueError("same player in two slots")
    return [p for ps in lineup.values() for p in ps]


def current_lineup(players):
    lineup = {}
    for p in starters(players):
        lineup.setdefault(p.slot, []).append(p)
    return lineup


if __name__ == "__main__":
    m = get_matchup()
    sim = Sim(m)

    print(f"\nWeek {m.week}: {N_SIMS:,} simulations\n")
    print(f"  {'':10}{'player':22}{'floor':>6}{'med':>6}{'ceil':>6}{'play%':>7}{'sim avg':>9}")
    for label, team in (("Team A", m.my_team), ("Opponent", m.opponent)):
        for p in starters(team):
            s = sim.scores[p.id]
            print(f"  {label:10}{p.name[:21]:22}{p.floor:6.1f}{p.median:6.1f}{p.ceiling:6.1f}"
                  f"{p.play_prob:7.0%}{s.mean():9.1f}")
        print()

    mine = current_lineup(m.my_team)
    r = sim.win_prob(check_lineup(mine))
    print(f"Current lineup: win {r['win_prob']:.1%}  "
          f"(median {r['my_median']:.1f} vs {r['opp_median']:.1f}; "
          f"your 10th-90th pct {r['my_p10_p90'][0]:.0f}-{r['my_p10_p90'][1]:.0f})")

    # one alternative to show the sim reacts: swap QBs
    by_name = {p.name: p for p in m.my_team}
    alt = {k: list(v) for k, v in mine.items()}
    alt["QB"] = [by_name["Bryce Young"]]
    r2 = sim.win_prob(check_lineup(alt))
    print(f"Bryce Young at QB: win {r2['win_prob']:.1%}  (median {r2['my_median']:.1f})")
