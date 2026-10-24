# Lockbox — Project Brief

## What this is
Lockbox is an autonomous fantasy football agent that makes start/sit decisions for an ESPN league. Built solo at the **c0mpiled-16 hackathon (UCSD, Sat Oct 24, 2026)**, theme "Invisible Interfaces," entered in the **Unsupervised** track: agents that run for hours without supervision and know when they are stuck, overspending, looping, or need to stop.

Building window: 2:00 PM to 7:30 PM (submit by 7:20). Judges are YC startup founders, several remote, so a clean demo and a backup video matter.

## Core idea (what makes it different from FantasyPros/ESPN start-sit)
Those tools give everyone the same rankings. Lockbox optimizes **for this user's specific matchup**:
- Each player is modeled as a **range of outcomes** (floor / median / ceiling), not one projected number.
- It runs a **Monte Carlo simulation** of the user's matchup vs. the opponent's lineup.
- It picks the lineup with the highest **win probability**, not the most projected points. Underdog → favor high-ceiling players; favorite → favor high-floor players.
- Every decision comes with a **one-line explanation**.

## Guardrails (the heart of the Unsupervised track; make these visible in the demo)
- **Flip-flop control:** only switch a player if the new choice beats the current one by a clear margin (hysteresis). Log every switch it resisted.
- **Confidence escalation:** if the top two lineups are within ~2% win probability, send ONE alert with its lean ("Coin flip: X or Y, leaning Y 55/45 — reply to override"). Otherwise stay quiet.
- **Stale data check:** if a data source fails or hasn't updated, don't act on old info.
- **Budget cap:** hard limit on LLM/API spend per run; show spend live.
- **Hard stop:** lock and stop at kickoff, then send a short report.

## Data (already set up and tested)
- **ESPN league:** `espn_api` library (`from espn_api.football import League`). Credentials are in `.env`: `LEAGUE_ID`, `TEAM_ID` (user is team 8), `ESPN_S2`, `SWID`. Season year 2026. See `test.py` (prints this week's matchups; works).
- **NFL stats:** `nflreadpy` (`import nflreadpy as nfl; nfl.load_player_stats([2026])`). See `stats_test.py` (works). Use for player outcome ranges and the backtest.
- **LLM:** `anthropic` SDK, key `ANTHROPIC_API_KEY` in `.env`. Use `claude-haiku-5-5` for frequent/cheap calls. See `key_test.py` (works).
- Environment: macOS, Python 3.14, run with `python3`. A pip warning about `espn-api` wanting older `idna` is known and harmless — ignore it.
- ESPN does not allow setting other people's lineups. Lockbox **recommends** and alerts; the user makes the swap. Mock any "auto-swap" for the demo.

## Build plan (one file at a time; get each working before the next)
| Time | File | Purpose | Done when |
|---|---|---|---|
| 2:00–3:00 | `data.py` | Pull user roster, opponent, injuries, recent NFL stats | Prints user's players with recent stats |
| 3:00–4:15 | `sim.py` | Floor/median/ceiling per player + matchup Monte Carlo | Prints win % for a given lineup |
| 4:15–5:15 | `agent.py` | Choose best lineup, explain it, run on a loop with all guardrails | Loop runs and logs decisions |
| 5:15–6:15 | `backtest.py` | Replay weeks 1–7 vs. baselines (top projected points; user's actual lineups) | Produces an honest points/wins number, including where it lost |
| 6:15–6:45 | `demo.py` | Simulate a starter ruled inactive at "11:31 AM" → agent swaps and alerts | Swap happens live on screen |
| 6:45–7:20 | — | Rehearse pitch, record backup video, submit | Submitted by 7:20 |

**Scope rules:** a simple working version beats a fancy half-done one. If something takes >20 minutes, skip it. At 6:15, stop adding features. Must-have: data, outcome ranges, simulation, lineup decision + reason, flip-flop control, escalation alert, backtest number. Nice-to-have: live news polling, polished dashboard. Skip: real ESPN lineup writes.

**Alerts:** a Discord webhook (URL in `.env` as `DISCORD_WEBHOOK_URL`) is the simplest option; printing to the terminal is an acceptable fallback.

## Demo script (3 minutes)
1. Hook: "Every Sunday I stare at my lineup and guess. So I built an agent that doesn't."
2. The edge: same players, different start because I'm the underdog this week.
3. Guardrails: it resists a flip-flop, then sends one coin-flip alert.
4. Drama: simulated inactive at 11:31, auto-swapped with an alert.
5. Proof: backtest number on screen.
6. Close: "Agents will soon make decisions for us. This one knows when it's unsure and when to stop."

## Rules for Claude Code
- Never print, log, or commit secrets from `.env`. Keep `.env` in `.gitignore`.
- In any demo output, show neutral team names ("Team A", "Opponent") instead of real league team names.
- Keep the backtest honest; report losses too.
- Test each file before moving on, and keep code simple and readable.
