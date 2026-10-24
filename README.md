# Lockbox

**An unsupervised fantasy football agent that sets your lineup to maximize your chance of *winning this week*, and knows when it's unsure, overspending, or should stop.**

Built at c0mpiled-16 (UCSD) for the **Unsupervised** track.

Rankings tools like ESPN and FantasyPros give everyone the same projected points. Lockbox optimizes for **your specific matchup**:

- Every player is a **range of outcomes** (floor / median / ceiling), not one number.
- A **Monte Carlo simulation** (20,000 games) plays your lineup against your opponent's.
- It picks the lineup with the highest **win probability**. As the underdog, upside matters. As the favorite, safety matters.
- Every decision comes with a **one-line reason**, and alerts go to Discord.
- It watches your matchup **until kickoff**, with no supervision.

## Guardrails (the point of the Unsupervised track)

| Guardrail | What it does |
|---|---|
| Flip-flop control | Only switches lineups when the gain is at least 1.5 points of win probability. Every resisted switch is logged. |
| Coin-flip escalation | When the top two lineups are within 2%, it sends **one** alert with its lean ("Leaning Kelce 64/36") instead of guessing silently. |
| Forced switches | A starter ruled OUT is replaced immediately, changing only what's necessary. |
| Stale data | If a fetch fails, is over 30 minutes old, or is missing this season's stats, it **holds** its last decision. After 3 failures in a row, it alerts and stops. |
| Budget cap | Hard cap on LLM spend per run, shown every loop. At the cap the LLM turns off and news falls back to keyword rules. |
| Hard stop | Locks and stops at kickoff, then sends a final report. Players whose games have started are never moved. |
| Conflicting sources | When ESPN and the official NFL roster disagree, it reads the latest news (Claude Haiku) and cites the source, e.g. "Schefter: exemption removed, expected to play". |

## Backtest (weeks 1-6, real points, no lookahead)

Each past week is rewound to kickoff: that week's injury report, roster status and byes, stats only from earlier weeks, and final scores hidden from the simulation.

| Lineup | Points | Share of best possible |
|---|---|---|
| My actual lineups | 754.7 | 86.8% |
| ESPN projections | 745.5 | 85.7% |
| **Lockbox agent** | **784.1** | **90.1%** |

- **+29.4 points** over my real lineups across 6 weeks. **3 of its 4 changes helped**, and one cost 1.8 points (week 5).
- The flip-flop guardrail **blocked a coin-flip swap that would have cost 15.5 points** (week 1). It also held back one that would have gained 8.0, and the report shows that too.
- It's judged on points, not wins: the opponent's score is outside the lineup's control. Six weeks is a small sample.

## How it works

```
data.py      ESPN league + nflverse stats -> floor/median/ceiling per player
             (2024-26 history, defense ratings from real yards/YPC/TDs allowed,
              usage trends, injured/returning teammates, news check for doubtful players)
news.py      latest news + ESPN/Sleeper status -> Claude Haiku verdict (PLAYING/OUT/UNCERTAIN)
sim.py       Monte Carlo of the matchup (teammates correlated, finished games use real points)
agent.py     searches every legal lineup (bench + FLEX included), decides, runs the guardrail loop
backtest.py  replays past weeks vs. my lineups, ESPN projections and the best lineup in hindsight
demo.py      3-minute demo: real data, demo clock, simulated 11:31 AM inactive
alerts.py    Discord webhook (prints to the terminal if none is set)
```

## Run it

```bash
pip install espn_api nflreadpy anthropic python-dotenv requests numpy scipy polars certifi
```

`.env` (never committed):
```
ESPN_S2=...  SWID=...  LEAGUE_ID=...  TEAM_ID=...
ANTHROPIC_API_KEY=...
DISCORD_WEBHOOK_URL=...   # optional
```

```bash
python3 agent.py --plan            # recommended lineup, slot by slot, with reasons and the FLEX decision
python3 agent.py --every 15        # run unsupervised until kickoff
python3 backtest.py                # replay weeks 1-6
python3 demo.py                    # live demo (--offline uses a saved snapshot, --fast skips pauses)
python3 tests/test_guardrails.py   # 12 guardrail tests on made-up data
```

ESPN doesn't allow third-party apps to set lineups, so Lockbox **recommends and alerts**, and you make the swap.
