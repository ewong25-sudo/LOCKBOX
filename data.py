"""Pull this week's matchup from ESPN and give every player a floor / median / ceiling.

Baselines come from each player's own 2024-2026 weekly history (recent seasons
weighted more). Then three adjustments:
  1. usage trend: is he getting more or fewer targets/carries lately than in his baseline?
  2. defense: the NFL defense he faces, rated from real stats it allows
     (yards, YPC, TDs, catch rate, sacks, points), not fantasy points
  3. teammates in and out: targets/carries left behind by teammates ruled Out/Doubtful
     are handed to the healthy players (same position first), and teammates returning
     from injury/suspension take volume back
"""
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import nflreadpy as nfl
import polars as pl
from dotenv import load_dotenv
from espn_api.football import League

import news

load_dotenv()

# Keep nflverse downloads on disk for 30 min: a cold fetch is 1-2 minutes on slow wifi,
# and old seasons never change. Live injury news still comes from ESPN + the news check.
nfl.config.update_config(cache_mode="filesystem", cache_duration=30 * 60)

YEAR = 2026
SEASONS = [2024, 2025, 2026]
# Share of the total weight each season gets (recent counts more). A season with
# fewer than 4 games gets a proportionally smaller share, so one game can't dominate.
SEASON_SHARE = {2026: 0.50, 2025: 0.33, 2024: 0.17}
FULL_SHARE_GAMES = 4
DEFENSE_SEASON_WEIGHT = {2026: 2.0, 2025: 1.0}         # defenses change a lot year to year
FLOOR_Q, MEDIAN_Q, CEILING_Q = 0.20, 0.50, 0.85
MIN_GAMES = 8        # fewer than this and we blend in a replacement-level baseline
DEF_SCALE = 0.10     # each standard deviation of defense strength moves projections 10%
DEF_CAP = 0.25       # ...up to +/-25%
SKILL = ("QB", "RB", "WR", "TE")
# Fallback only: get_matchup() reads the league's real slots from ESPN settings
STARTER_SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "RB/WR/TE": 1, "D/ST": 1, "K": 1}
NON_STARTING = ("BE", "IR")
RECENT_GAMES = 4     # "lately" = last 4 games this season
TREND_RANGE = (0.85, 1.20)   # usage trend can move a range by at most this much
SAME_POS_PULL = 2.0  # a WR's vacated targets go to other WRs at 2x the rate of RBs/TEs
INJURY_BOOST_CAP = 1.40      # injured teammates can raise a median by at most 40%...
RETURN_DRAG_CAP = 0.60       # ...and returning ones can cut it to no less than 60%
RETURN_RAMP = 0.8    # players back from injury usually get ~80% of their old workload at first
MIN_RETURN_VOLUME = 4        # a returner must have averaged 4+ targets+carries to matter
CHECK_NEWS_FOR = ("QUESTIONABLE", "DOUBTFUL", "DAY_TO_DAY")   # ESPN tags worth a news check
NOT_ACTIVE = {"RES": "reserve/IR", "EXE": "exempt list", "INA": "inactive", "CUT": "released",
              "DEV": "practice squad", "RET": "retired", "TRT": "left team"}
OUT_STATUSES = ("Out", "Doubtful")
ESPN_TO_NFLVERSE = {"LAR": "LA", "WSH": "WAS"}
# "Replacement level" = what a fringe starter at the position scores (by 2025 points per game rank)
REPLACEMENT_RANKS = {"QB": (13, 24), "RB": (25, 48), "WR": (37, 60), "TE": (13, 24)}

WARNINGS = []   # fallbacks hit during the current get_matchup(), for the agent's stale-data check
CRITICAL = []   # fallbacks bad enough that the agent must not act on this data


def log(msg, warn=False, critical=False):
    print(f"[data] {msg}")
    if warn or critical:
        WARNINGS.append(msg)
    if critical:
        CRITICAL.append(msg)


@dataclass
class Player:
    id: int
    name: str
    position: str
    pro_team: str           # current NFL team (nflverse code)
    opp_team: str           # NFL team he faces this week ("" if bye)
    slot: str               # current ESPN lineup slot ("BE" = bench)
    eligible: list          # lineup slots this player can fill
    injury: str             # ACTIVE, QUESTIONABLE, OUT, ...
    on_bye: bool
    projected: float        # ESPN projection, kept only as a cross-check
    espn_history: list = field(default_factory=list)  # 2026 league-scored points (used for K and D/ST)
    games: int = 0          # games in the baseline sample
    floor: float = 0.0
    median: float = 0.0
    ceiling: float = 0.0
    def_factor: float = 1.0
    def_note: str = ""
    usage: str = ""         # e.g. "8.5 tgt, 2.0 car, 84% snaps /g (last 4)"
    trend: float = 1.0      # recent usage vs. his baseline sample
    team_shift: float = 0.0     # median points gained (+) from injured teammates or lost (-) to returning ones
    game_time: object = None    # kickoff (naive local time, from ESPN); None on bye
    game_progress: int = 0      # 0 = not started, 100 = final
    points: float = 0.0         # actual fantasy points so far this week
    actual: float = 0.0         # backtest only: final points, hidden from the sim
    play_prob: float = 1.0  # chance he plays at all (from the news check when status is in doubt)
    news: dict = field(default_factory=dict)  # news verdict, reason, and source time
    flags: list = field(default_factory=list)  # every fallback or caveat, in plain words

    @property
    def unavailable(self):
        return self.on_bye or self.injury in ("OUT", "INJURY_RESERVE", "SUSPENSION")

    @property
    def starting(self):
        return self.slot not in NON_STARTING

    def locked(self, now):
        """True once his game has kicked off: ESPN won't let him move in or out."""
        return self.game_progress > 0 or (self.game_time is not None and now >= self.game_time)


@dataclass
class Matchup:
    week: int
    my_team: list
    opponent: list
    fetched_at: float       # unix time, for the stale-data check
    warnings: list = field(default_factory=list)  # sources that failed this fetch
    critical: list = field(default_factory=list)  # failures that make this data unsafe to act on
    slots: dict = field(default_factory=lambda: dict(STARTER_SLOTS))  # league's starting slots


# ---------- ESPN ----------

LEAGUE_CACHE = Path(__file__).parent / ".cache" / "espn_league.pkl"
LEAGUE_CACHE_S = 6 * 3600   # settings/teams barely change; box scores + player stats are always fetched fresh


def connect(fresh=False):
    """The ESPN league. Building it downloads the whole season (~30 s on slow wifi), but we
    only use its settings and IDs from it, so it's cached on disk. Credentials are never
    written to the cache; they're re-attached from .env on load."""
    cookies = {"espn_s2": os.getenv("ESPN_S2"), "SWID": os.getenv("SWID")}
    if not fresh:
        try:
            if time.time() - LEAGUE_CACHE.stat().st_mtime < LEAGUE_CACHE_S:
                with open(LEAGUE_CACHE, "rb") as f:
                    league = pickle.load(f)
                league.espn_request.cookies = cookies
                return league
        except Exception:
            pass   # missing / old / unreadable cache: build it again
    league = League(league_id=int(os.getenv("LEAGUE_ID")), year=YEAR,
                    espn_s2=cookies["espn_s2"], swid=cookies["SWID"])
    try:
        league.espn_request.cookies = None
        LEAGUE_CACHE.parent.mkdir(exist_ok=True)
        with open(LEAGUE_CACHE, "wb") as f:
            pickle.dump(league, f)
    except Exception:
        pass   # caching is an optimization; never fail the fetch over it
    finally:
        league.espn_request.cookies = cookies
    return league


def _nfl_team(code):
    if not code or code == "None":
        return ""
    return ESPN_TO_NFLVERSE.get(code, code)


def league_slots(league):
    """The league's starting slots, e.g. {"QB": 1, "RB": 2, ..., "RB/WR/TE": 1}, from ESPN settings."""
    counts = getattr(league.settings, "position_slot_counts", None) or {}
    slots = {s: n for s, n in counts.items() if n > 0 and s not in NON_STARTING}
    if not slots:
        log("league slot settings unavailable; using standard QB/2RB/2WR/TE/FLEX/DST/K", warn=True)
        return dict(STARTER_SLOTS)
    return slots


def _players(league, lineup, week, slots=STARTER_SLOTS, infos=None):
    """infos: {playerId: ESPN player info}; pass it in to fetch both teams in one request."""
    if infos is None:
        infos = {p.playerId: p for p in league.player_info(playerId=[p.playerId for p in lineup])}
    players = []
    for p in lineup:
        if p.slot_position == "IR":
            continue
        info = infos.get(p.playerId)
        history = [s["points"] for w, s in sorted(info.stats.items())
                   if 0 < w < week and s.get("breakdown")] if info else []
        players.append(Player(
            id=p.playerId,
            name=p.name,
            position=p.position,
            pro_team=_nfl_team(p.proTeam),
            opp_team=_nfl_team(p.pro_opponent),
            slot=p.slot_position,
            eligible=[s for s in p.eligibleSlots if s in slots],   # ESPN's own eligibility list
            injury=p.injuryStatus or "ACTIVE",
            on_bye=bool(p.on_bye_week),
            projected=float(p.projected_points or 0),
            espn_history=history,
            game_time=getattr(p, "game_date", None),
            game_progress=int(getattr(p, "game_played", 0) or 0) if not p.on_bye_week else 0,
            points=float(p.points or 0),
        ))
    return players


# ---------- NFL stats ----------

def load_stats(week):
    """Regular-season weekly stats for 2024-2026, only weeks already played."""
    stats = None
    for attempt in (1, 2):   # downloads occasionally fail; one retry before degrading
        try:
            stats = nfl.load_player_stats(SEASONS)
            break
        except Exception as e:
            log(f"stats download failed ({type(e).__name__}), attempt {attempt}/2")
    if stats is None:
        log(f"multi-season load failed; using {SEASONS[0]}-{YEAR - 1} only", warn=True, critical=True)
        stats = nfl.load_player_stats([s for s in SEASONS if s != YEAR])
    stats = stats.filter(
        (pl.col("season_type") == "REG")
        & ((pl.col("season") < YEAR) | (pl.col("week") < week))
    )
    games_now = stats.filter(pl.col("season") == YEAR)["game_id"].n_unique()
    expected = 14 * (week - 1)   # ~14+ games a week even with byes
    if games_now < expected * 0.8:
        # missing this season's games would silently change every projection: flag it loudly
        log(f"only {games_now} {YEAR} games found (expected ~{expected}); leaning on "
            f"{SEASONS[0]}-{YEAR - 1}", warn=True, critical=week > 3)
    else:
        log(f"found {games_now} {YEAR} games (weeks 1-{week - 1}) + {SEASONS[0]}-{YEAR - 1}")
    return stats


def _wquantile(values, weights, q):
    pairs = sorted(zip(values, weights))
    target, acc = q * sum(weights), 0.0
    for v, w in pairs:
        acc += w
        if acc >= target:
            return v
    return pairs[-1][0]


def _quantiles(values, weights=None):
    weights = weights or [1.0] * len(values)
    return tuple(_wquantile(values, weights, q) for q in (FLOOR_Q, MEDIAN_Q, CEILING_Q))


def _replacement(stats):
    """Floor/median/ceiling of a fringe starter at each position, from 2025 games."""
    last = stats.filter(pl.col("season") == YEAR - 1)
    out = {}
    for pos, (lo, hi) in REPLACEMENT_RANKS.items():
        ranked = (
            last.filter(pl.col("position") == pos)
            .group_by("player_id").agg(pl.col("fantasy_points_ppr").mean().alias("ppg"), pl.len().alias("g"))
            .filter(pl.col("g") >= 6).sort("ppg", descending=True)
            .slice(lo - 1, hi - lo + 1)
        )
        pool = last.filter(pl.col("player_id").is_in(ranked["player_id"].implode()))
        out[pos] = _quantiles(pool["fantasy_points_ppr"].to_list())
    return out


def _id_maps():
    """ESPN id -> nflverse gsis id, and PFR id -> gsis id (snap counts use PFR ids)."""
    ids = nfl.load_ff_playerids().select("espn_id", "pfr_id", "gsis_id").filter(pl.col("gsis_id").is_not_null())
    espn, pfr = {}, {}
    for espn_id, pfr_id, gsis_id in ids.iter_rows():
        if pfr_id:
            pfr[pfr_id] = gsis_id
        try:
            espn[int(float(espn_id))] = gsis_id
        except (TypeError, ValueError):
            pass
    return espn, pfr


# ---------- usage and teammates in/out ----------

def build_usage(stats, week, pfr_to_gsis):
    """Each player's recent per-game volume (targets, carries, snap %), keyed by gsis id."""
    recent = (stats.filter(pl.col("season") == YEAR)
              .sort("week").group_by("player_id").tail(RECENT_GAMES))
    usage = {}
    for r in (recent.group_by("player_id").agg(
            pl.col("team").last(), pl.col("position").last(), pl.len().alias("games"),
            pl.col("week").max().alias("last_week"), pl.col("week").alias("weeks"),
            pl.col("targets").mean().alias("tgt"), pl.col("carries").mean().alias("car"),
            pl.col("fantasy_points_ppr").sum().alias("pts"),
            (pl.col("targets") + pl.col("carries")).sum().alias("opps"),
    ).iter_rows(named=True)):
        r["pts_per_opp"] = r["pts"] / r["opps"] if r["opps"] else 0.0
        r["snap"] = None
        r["tgt_tg"] = r["car_tg"] = 0.0
        usage[r["player_id"]] = r

    # Per TEAM game over the team's last 4 games (a missed game counts as 0), so a team's
    # players add up to its real volume (~35 targets/g) and ex-contributors drop out of the pool
    season = stats.filter(pl.col("season") == YEAR)
    team_weeks = {t: sorted(ws)[-RECENT_GAMES:] for t, ws in
                  season.group_by("team").agg(pl.col("week").unique()).iter_rows()}
    window = season.filter(pl.struct("team", "week").map_elements(
        lambda s: s["week"] in team_weeks.get(s["team"], []), return_dtype=pl.Boolean))
    for g, team, tgt, car in (window.group_by("player_id", "team")
                              .agg(pl.col("targets").sum(), pl.col("carries").sum()).iter_rows()):
        if g in usage and usage[g]["team"] == team:
            n = len(team_weeks[team])
            usage[g]["tgt_tg"], usage[g]["car_tg"] = tgt / n, car / n
    try:
        snaps = nfl.load_snap_counts([YEAR]).filter(pl.col("week") < week)
        snaps = snaps.sort("week").group_by("pfr_player_id").tail(RECENT_GAMES)
        for pfr_id, pct in snaps.group_by("pfr_player_id").agg(pl.col("offense_pct").mean()).iter_rows():
            g = pfr_to_gsis.get(pfr_id)
            if g in usage:
                usage[g]["snap"] = pct
    except Exception as e:
        log(f"snap counts unavailable ({type(e).__name__}); usage uses targets/carries only", warn=True)
    return usage


def load_out_players(week, espn_players, espn_to_gsis):
    """gsis ids of everyone ruled Out/Doubtful this week (official NFL report + ESPN status)."""
    out = set()
    try:
        inj = nfl.load_injuries([YEAR]).filter(
            (pl.col("week") == week) & pl.col("report_status").is_in(OUT_STATUSES))
        out |= set(inj["gsis_id"].drop_nulls().to_list())
        log(f"injury report: {len(out)} players Out/Doubtful in week {week}")
    except Exception as e:
        log(f"injury report unavailable ({type(e).__name__}); using ESPN statuses only", warn=True)
    for p in espn_players:
        if p.unavailable and not p.on_bye and p.id in espn_to_gsis:
            out.add(espn_to_gsis[p.id])
    return out


def load_roster_status(week):
    """gsis id -> (NFL team, roster status) for this week, from official weekly rosters."""
    try:
        r = nfl.load_rosters_weekly([YEAR]).filter(pl.col("week") == week)
        return {g: (t, s) for g, t, s in r.select("gsis_id", "team", "status").drop_nulls().iter_rows()}
    except Exception as e:
        log(f"weekly rosters unavailable ({type(e).__name__}); can't spot returning teammates", warn=True)
        return {}


def check_news(players, espn_to_gsis, roster):
    """For anyone whose status is in doubt (an injury tag, or ESPN and the NFL roster
    disagree), read the latest news and let it decide. Updates players and roster in place."""
    cases, targets = [], []
    for p in players:
        if p.on_bye or p.position == "D/ST":
            continue
        g = espn_to_gsis.get(p.id)
        team, status = roster.get(g, (None, None))
        conflict = status in NOT_ACTIVE and not p.unavailable
        if conflict or p.injury in CHECK_NEWS_FOR:
            cases.append(dict(name=p.name, team=p.pro_team, espn_id=p.id, espn_injury=p.injury,
                              nfl_roster=NOT_ACTIVE.get(status, status)))
            targets.append((p, g, team))
    if not cases:
        return
    log(f"news check: {len(cases)} players with a doubtful or conflicting status")
    for (p, g, team), r in zip(targets, news.resolve_many(cases)):
        p.news = r
        p.play_prob = r["play_prob"]
        if r["verdict"] == "OUT":
            p.injury = "OUT"
        elif r["verdict"] == "PLAYING" and g and team:
            roster[g] = (team, "ACT")  # the news is newer than the roster file


def find_returners(stats, usage, out_ids, roster):
    """Skill players active this week who did NOT play their team's latest game but
    carried real volume before: they're back from injury/suspension and will take
    targets/carries from teammates. Volume = their last 4 games (any season) x ramp-up."""
    team_last = {}
    for u in usage.values():
        team_last[u["team"]] = max(team_last.get(u["team"], 0), u["last_week"])
    candidates = [g for g, (team, status) in roster.items()
                  if status == "ACT" and g not in out_ids and team in team_last
                  and (g not in usage or usage[g]["last_week"] < team_last[team])]
    rows = (stats.filter(pl.col("player_id").is_in(candidates) & pl.col("position").is_in(["RB", "WR", "TE"]))
            .sort("season", "week").group_by("player_id").tail(RECENT_GAMES))
    returners = {}
    for r in rows.group_by("player_id").agg(
            pl.col("position").last(), pl.col("targets").mean().alias("tgt"),
            pl.col("carries").mean().alias("car"), pl.col("season").max().alias("season"),
    ).iter_rows(named=True):
        if r["season"] < YEAR - 1 or r["tgt"] + r["car"] < MIN_RETURN_VOLUME:
            continue  # hasn't played since 2024, or never had a real role
        g = r["player_id"]
        returners[g] = dict(
            player_id=g, team=roster[g][0], position=r["position"],
            tgt=r["tgt"] * RETURN_RAMP, car=r["car"] * RETURN_RAMP,
            weeks=usage[g]["weeks"] if g in usage else [],
        )
    log(f"returning teammates: {len(returners)} players back with real volume")
    return returners


def _team_shift(p, gsis_id, usage, out_ids, returners, names):
    """Expected points gained from injured teammates' volume, minus points lost to
    teammates coming back.

    Injured: only teammates who played the team's latest game count (if they
    already sat out, everyone's recent usage already reflects it). Their targets
    go to healthy teammates in proportion to each one's targets, same position
    pulling 2x; their RB carries go to the healthy RBs.

    Returning: the same split in reverse. A returner only takes back volume from
    the recent games he missed, so if he sat 1 of my last 4 games, 1/4 of his
    volume comes out of my numbers.
    """
    me = usage.get(gsis_id)
    if not me or p.pro_team != me["team"]:
        return 0.0, []
    team = [u for u in usage.values() if u["team"] == me["team"] and u["position"] in SKILL]
    team_last = max(u["last_week"] for u in team)
    gone = [u for u in team if u["player_id"] in out_ids and u["player_id"] != gsis_id
            and u["last_week"] == team_last]
    healthy = [u for u in team if u["player_id"] not in out_ids]
    back = [r for r in returners.values() if r["team"] == me["team"] and r["player_id"] != gsis_id]

    def share(o, sign):
        # o's per-game workload is split by each healthy teammate's share of team volume
        pull = lambda u: SAME_POS_PULL if u["position"] == o["position"] else 1.0
        tgt = car = 0.0
        tgt_pool = sum(u["tgt_tg"] * pull(u) for u in healthy)
        if o["tgt"] and tgt_pool:
            tgt = o["tgt"] * me["tgt_tg"] * pull(me) / tgt_pool
        car_pool = sum(u["car_tg"] for u in healthy if u["position"] == "RB")
        if o["position"] == "RB" and p.position == "RB" and o["car"] and car_pool:
            car = o["car"] * me["car_tg"] / car_pool
        return sign * tgt, sign * car

    notes, total_tgt, total_car = [], 0.0, 0.0
    for o in gone:
        t, c = share(o, +1)
        total_tgt, total_car = total_tgt + t, total_car + c
        if t + c >= 0.3:
            notes.append(f"{names.get(o['player_id'], '?')} out ({o['tgt']:.1f} tgt, {o['car']:.1f} car/g) -> +{t + c:.1f} opp/g")
    for r in back:
        missed = sum(1 for w in me["weeks"] if w not in r["weeks"]) / len(me["weeks"])
        t, c = share(r, -missed)
        total_tgt, total_car = total_tgt + t, total_car + c
        if abs(t + c) >= 0.3:
            notes.append(f"{names.get(r['player_id'], '?')} returning (~{r['tgt']:.1f} tgt, {r['car']:.1f} car/g, "
                         f"missed {missed:.0%} of my recent games) -> {t + c:.1f} opp/g")
    if not notes:
        return 0.0, []
    pts = (total_tgt + total_car) * me["pts_per_opp"]
    return pts, notes + [f"net teammate effect: {pts:+.1f} pts"]


# ---------- defense ratings ----------

# For each position group: (stat label, how to compute it, +1 if a higher number means a weaker defense)
def _defense_metrics(d):
    g = d["games"]
    return {
        "run": [
            ("rush yds/g", d["rb_rush_yds"] / g, +1),
            ("yds/carry", d["rb_rush_yds"] / max(d["rb_carries"], 1), +1),
            ("rush TD/g", d["rb_rush_tds"] / g, +1),
        ],
        "WR": [
            ("WR yds/g", d["wr_yds"] / g, +1),
            ("yds/target", d["wr_yds"] / max(d["wr_tgt"], 1), +1),
            ("catch %", d["wr_rec"] / max(d["wr_tgt"], 1), +1),
            ("WR TD/g", d["wr_tds"] / g, +1),
        ],
        "TE": [
            ("TE yds/g", d["te_yds"] / g, +1),
            ("yds/target", d["te_yds"] / max(d["te_tgt"], 1), +1),
            ("catch %", d["te_rec"] / max(d["te_tgt"], 1), +1),
            ("TE TD/g", d["te_tds"] / g, +1),
        ],
        "QB": [
            ("pass yds/g", d["pass_yds"] / g, +1),
            ("pass TD/g", d["pass_tds"] / g, +1),
            ("INT/g", d["ints"] / g, -1),
            ("sacks/g", d["sacks"] / g, -1),
            ("pts allowed/g", d["pts_allowed"] / max(d["sched_games"], 1), +1),
        ],
    }


def build_defense_ratings(stats, week):
    """Rate every NFL defense per position group from real stats it allowed (2025 + 2026)."""
    rows = stats.filter(pl.col("season").is_in(list(DEFENSE_SEASON_WEIGHT))).with_columns(
        pl.col("season").replace_strict(DEFENSE_SEASON_WEIGHT, return_dtype=pl.Float64).alias("w")
    )

    def wsum(expr, name):
        return (expr * pl.col("w")).sum().alias(name)

    pos = pl.col("position")
    agg = rows.group_by("opponent_team").agg(
        wsum(pl.when(pos == "RB").then(pl.col("rushing_yards")).otherwise(0), "rb_rush_yds"),
        wsum(pl.when(pos == "RB").then(pl.col("carries")).otherwise(0), "rb_carries"),
        wsum(pl.when(pos == "RB").then(pl.col("rushing_tds")).otherwise(0), "rb_rush_tds"),
        *[wsum(pl.when(pos == p).then(pl.col(c)).otherwise(0), f"{p.lower()}_{n}")
          for p in ("WR", "TE")
          for c, n in (("receiving_yards", "yds"), ("targets", "tgt"), ("receptions", "rec"), ("receiving_tds", "tds"))],
        wsum(pl.when(pos == "QB").then(pl.col("passing_yards")).otherwise(0), "pass_yds"),
        wsum(pl.when(pos == "QB").then(pl.col("passing_tds")).otherwise(0), "pass_tds"),
        wsum(pl.when(pos == "QB").then(pl.col("passing_interceptions")).otherwise(0), "ints"),
        wsum(pl.when(pos == "QB").then(pl.col("sacks_suffered")).otherwise(0), "sacks"),
    )
    # games per defense (weighted), counted once per game rather than once per player row
    games = (rows.select("opponent_team", "game_id", "w").unique()
             .group_by("opponent_team").agg(pl.col("w").sum().alias("games")))
    agg = agg.join(games, on="opponent_team")

    # points allowed come from the schedule, not player stats
    pts = {}
    try:
        sched = nfl.load_schedules(list(DEFENSE_SEASON_WEIGHT)).filter(
            (pl.col("game_type") == "REG") & pl.col("home_score").is_not_null()
            & ((pl.col("season") < YEAR) | (pl.col("week") < week))
        )
        for r in sched.iter_rows(named=True):
            w = DEFENSE_SEASON_WEIGHT[r["season"]]
            for team, allowed in ((r["home_team"], r["away_score"]), (r["away_team"], r["home_score"])):
                p, g = pts.get(team, (0.0, 0.0))
                pts[team] = (p + w * allowed, g + w)
    except Exception as e:
        log(f"schedules unavailable ({type(e).__name__}); QB rating skips points allowed", warn=True)

    teams = {}
    for r in agg.iter_rows(named=True):
        r["pts_allowed"], r["sched_games"] = pts.get(r["opponent_team"], (0.0, 0.0))
        teams[r["opponent_team"]] = _defense_metrics(r)

    # z-score every stat across the league, flip so + always means "easier for the offense"
    ratings = {}
    for group in ("run", "WR", "TE", "QB"):
        names = [m[0] for m in next(iter(teams.values()))[group]]
        z_by_team = {t: [] for t in teams}
        for i, name in enumerate(names):
            vals = {t: teams[t][group][i][1] for t in teams}
            if name == "pts allowed/g" and not pts:
                continue
            mean = sum(vals.values()) / len(vals)
            sd = (sum((v - mean) ** 2 for v in vals.values()) / len(vals)) ** 0.5 or 1.0
            for t, v in vals.items():
                z_by_team[t].append(teams[t][group][i][2] * (v - mean) / sd)
        order = sorted(teams, key=lambda t: sum(z_by_team[t]) / len(z_by_team[t]))
        for rank, t in enumerate(order, 1):
            z = sum(z_by_team[t]) / len(z_by_team[t])
            factor = 1 + max(-DEF_CAP, min(DEF_CAP, DEF_SCALE * z))
            detail = ", ".join(f"{n} {v:.2f}" if v < 1 else f"{n} {v:.1f}" for n, v, _ in teams[t][group])
            ratings[(t, group)] = (round(factor, 3), rank, len(order), detail)
    _cross_check(rows, ratings)
    return ratings


def _cross_check(rows, ratings):
    """Sanity check only: do our ratings agree with plain fantasy points allowed?"""
    for group, positions in (("run", ["RB"]), ("WR", ["WR"]), ("TE", ["TE"]), ("QB", ["QB"])):
        fp = (rows.filter(pl.col("position").is_in(positions))
              .group_by("opponent_team", "game_id").agg(pl.col("fantasy_points_ppr").sum().alias("p"))
              .group_by("opponent_team").agg(pl.col("p").mean()))
        pairs = [(ratings[(t, group)][0], p) for t, p in fp.iter_rows() if (t, group) in ratings]
        if len(pairs) < 3:
            continue
        xs, ys = zip(*pairs)
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in pairs)
        corr = cov / ((sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5 or 1)
        log(f"cross-check: {group} rating vs fantasy pts allowed, correlation {corr:.2f}")


# ---------- per-player ranges ----------

@dataclass
class Context:
    stats: pl.DataFrame
    gsis: dict              # ESPN id -> gsis id
    replacement: dict
    ratings: dict
    usage: dict
    out_ids: set
    returners: dict
    roster: dict            # gsis id -> (team, official roster status)
    names: dict             # gsis id -> name, for explanations


def project(p, ctx):
    stats, gsis, replacement, ratings = ctx.stats, ctx.gsis, ctx.replacement, ctx.ratings
    if p.news:
        p.flags.append(f"news -> {p.news['verdict']} ({p.news['play_prob']:.0%} to play, "
                       f"via {p.news['via']}): {p.news['reason']}")
    if p.unavailable:
        p.flags.append("on bye" if p.on_bye else f"{p.injury.lower()}: projected 0")
        return

    if p.position in ("K", "D/ST"):
        # nflreadpy has no kicker/defense fantasy scoring, so use ESPN's 2026 league-scored history
        if len(p.espn_history) >= 3:
            p.floor, p.median, p.ceiling = _quantiles(p.espn_history)
            p.games = len(p.espn_history)
        else:
            p.floor, p.median, p.ceiling = p.projected * 0.5, p.projected, p.projected * 1.6
            p.flags.append("few games: range built around ESPN projection")
        p.flags.append("no defense adjustment for K/DST")
        return

    gsis_id = gsis.get(p.id)
    rows = stats.filter(pl.col("player_id") == gsis_id) if gsis_id else stats.clear()
    if gsis_id is None:
        p.flags.append("no ID match in nflverse")
    status = ctx.roster.get(gsis_id, (None, None))[1]
    if status in NOT_ACTIVE and not p.news:
        # ESPN thinks he can play, the official roster says otherwise, and no news settled it
        p.flags.append(f"CONFLICT: ESPN says {p.injury}, NFL roster says {NOT_ACTIVE[status]} - check before kickoff")
    values = rows["fantasy_points_ppr"].to_list()
    seasons = rows["season"].to_list()
    counts = {s: seasons.count(s) for s in set(seasons)}
    # per-game weight = that season's share / its games, so a new role this year isn't drowned out by old games
    weights = [SEASON_SHARE[s] * min(1, counts[s] / FULL_SHARE_GAMES) / counts[s] for s in seasons]
    p.games = len(values)
    repl = replacement.get(p.position, (0.0, 0.0, 0.0))

    if not values:
        own = repl
        p.flags.append(f"no history: using replacement-level {p.position}")
    else:
        own = _quantiles(values, weights)
        if len(values) < MIN_GAMES:
            k = len(values) / MIN_GAMES
            own = tuple(k * o + (1 - k) * r for o, r in zip(own, repl))
            p.flags.append(f"only {len(values)} games: blended with replacement-level {p.position}")

    # Returning from injury/suspension: little 2026 data, so the range leans on 2024-25
    games_2026 = rows.filter(pl.col("season") == YEAR).height
    weeks_played = stats.filter(pl.col("season") == YEAR)["week"].n_unique()
    if values and weeks_played >= 3 and games_2026 <= 1 and len(values) >= MIN_GAMES:
        p.flags.append(f"returning: {games_2026} game(s) in {YEAR}, range leans on {SEASONS[0]}-{YEAR - 1}")

    # Team context: are his numbers from the team he plays for now?
    if values and p.pro_team:
        on_team = rows.filter(pl.col("team") == p.pro_team).height
        if on_team < 3:
            old = rows["team"].mode().to_list()
            p.flags.append(f"history mostly with {old[0] if old else '?'}, now {p.pro_team}: may be stale")

    group = "run" if p.position == "RB" else p.position
    rating = ratings.get((p.opp_team, group))
    if rating:
        p.def_factor, rank, n, detail = rating
        side = f"#{rank} toughest" if rank <= n / 2 else f"#{n - rank + 1} weakest"
        label = {"run": "run D", "WR": "D vs WRs", "TE": "D vs TEs", "QB": "pass/scoring D"}[group]
        p.def_note = f"{p.opp_team} {side} {label} ({detail})"
    elif p.opp_team:
        p.flags.append(f"no defense rating for {p.opp_team}")

    # Usage: recent targets/carries vs. what he averaged in the games his baseline came from
    u = ctx.usage.get(gsis_id)
    if u:
        snap = f", {u['snap'] * 100:.0f}% snaps" if u["snap"] is not None else ""
        p.usage = f"{u['tgt']:.1f} tgt, {u['car']:.1f} car{snap} /g"
        opps = (rows["targets"] + rows["carries"]).to_list()
        base_opp = sum(o * w for o, w in zip(opps, weights)) / sum(weights) if weights else 0
        recent_opp = u["tgt"] + u["car"]
        if p.position != "QB" and u["games"] >= 3 and base_opp >= 1:
            # square root: the baseline already leans on recent games, so only half-trust the gap
            p.trend = round(min(TREND_RANGE[1], max(TREND_RANGE[0], (recent_opp / base_opp) ** 0.5)), 3)
            if abs(p.trend - 1) >= 0.05:
                p.flags.append(f"usage {'up' if p.trend > 1 else 'down'}: {recent_opp:.1f} opp/g lately "
                               f"vs {base_opp:.1f} in baseline -> x{p.trend:.2f}")
    elif values:
        p.flags.append("no recent usage (hasn't played lately)")

    floor, median, ceiling = (v * p.def_factor * p.trend for v in own)

    # Teammates in and out: more volume raises the median and the ceiling most
    # ("greater potential"); a returning teammate does the reverse
    if u:
        pts, notes = _team_shift(p, gsis_id, ctx.usage, ctx.out_ids, ctx.returners, ctx.names)
        if pts:
            if median:
                pts = max(-median * (1 - RETURN_DRAG_CAP), min(pts, median * (INJURY_BOOST_CAP - 1)))
            p.team_shift = round(pts, 1)
            floor = max(0.0, floor + 0.5 * pts)
            median, ceiling = median + pts, ceiling + 1.5 * pts
        p.flags.extend(notes)
    p.floor, p.median, p.ceiling = round(floor, 1), round(median, 1), round(ceiling, 1)


def as_of_kickoff(players, week, espn_to_gsis, roster):
    """Rewind a past week to what was knowable before kickoff (for the backtest).

    ESPN's past box scores carry TODAY's injury tags and the final scores, so:
    statuses come from that week's official injury report and roster status, byes
    from that week's schedule, and actual points move to p.actual where the sim can't see them.
    """
    inj = nfl.load_injuries([YEAR]).filter(pl.col("week") == week)
    out = set(inj.filter(pl.col("report_status").is_in(OUT_STATUSES))["gsis_id"].drop_nulls().to_list())
    sched = nfl.load_schedules([YEAR]).filter((pl.col("week") == week) & (pl.col("game_type") == "REG"))
    playing = set(sched["home_team"].to_list()) | set(sched["away_team"].to_list())
    for p in players:
        p.actual = p.points
        p.points, p.game_progress, p.game_time = 0.0, 0, None
        p.on_bye = bool(p.pro_team) and p.pro_team not in playing
        g = espn_to_gsis.get(p.id)
        # IR / exempt-list players aren't on the injury report, so check that week's roster status too
        inactive = roster.get(g, (None, None))[1] in NOT_ACTIVE
        p.injury = "OUT" if g in out or inactive else "ACTIVE"


def get_matchup(league=None, week=None, historical=False):
    """historical=True replays a past week as of kickoff (see as_of_kickoff)."""
    WARNINGS.clear()
    CRITICAL.clear()
    league = league or connect()
    week = week or league.current_week
    team_id = int(os.getenv("TEAM_ID"))
    for box in league.box_scores(week):
        if box.home_team.team_id == team_id:
            mine, theirs = box.home_lineup, box.away_lineup
        elif box.away_team.team_id == team_id:
            mine, theirs = box.away_lineup, box.home_lineup
        else:
            continue
        slots = league_slots(league)
        # both rosters' season stats in ONE request
        infos = {p.playerId: p for p in league.player_info(playerId=[p.playerId for p in mine + theirs])}
        m = Matchup(week=week, my_team=_players(league, mine, week, slots, infos),
                    opponent=_players(league, theirs, week, slots, infos), fetched_at=time.time(), slots=slots)
        everyone = m.my_team + m.opponent
        stats = load_stats(week)
        espn_to_gsis, pfr_to_gsis = _id_maps()
        names = dict(stats.select("player_id", "player_display_name").unique("player_id").iter_rows())
        usage = build_usage(stats, week, pfr_to_gsis)
        roster = load_roster_status(week)
        if historical:
            as_of_kickoff(everyone, week, espn_to_gsis, roster)   # no peeking at results or today's news
        else:
            check_news(everyone, espn_to_gsis, roster)   # before anything uses statuses
        out_ids = load_out_players(week, everyone, espn_to_gsis)
        ctx = Context(
            stats=stats,
            gsis=espn_to_gsis,
            replacement=_replacement(stats),
            ratings=build_defense_ratings(stats, week),
            usage=usage,
            out_ids=out_ids,
            returners=find_returners(stats, usage, out_ids, roster),
            roster=roster,
            names=names,
        )
        for p in everyone:
            project(p, ctx)
        m.warnings, m.critical = list(WARNINGS), list(CRITICAL)
        return m
    raise RuntimeError(f"No matchup found for team {team_id} in week {week}")


def _print_team(label, players):
    print(f"\n{label}")
    print(f"  {'slot':9}{'pos':5}{'player':22}{'team':5}{'opp':5}{'def':>6}{'trend':>6}{'tm+/-':>6}"
          f"{'floor':>7}{'median':>7}{'ceil':>7}{'ESPN':>7}  usage (last {RECENT_GAMES})")
    for p in sorted(players, key=lambda p: (not p.starting, p.position)):
        print(f"  {p.slot:9}{p.position:5}{p.name[:21]:22}{p.pro_team:5}{p.opp_team or 'BYE':5}"
              f"{p.def_factor:6.2f}{p.trend:6.2f}{p.team_shift:+6.1f}"
              f"{p.floor:7.1f}{p.median:7.1f}{p.ceiling:7.1f}{p.projected:7.1f}  {p.usage}")
    for p in players:
        notes = ([p.def_note] if p.def_note and abs(p.def_factor - 1) >= 0.05 else []) + p.flags
        notes = [n for n in notes if n not in ("on bye",) and "projected 0" not in n and "K/DST" not in n]
        for n in notes:
            print(f"    {p.name}: {n}")


if __name__ == "__main__":
    m = get_matchup()
    print(f"\nWeek {m.week}   (def = defense multiplier; >1 soft matchup, <1 tough)")
    _print_team("Team A (you)", m.my_team)
    _print_team("Opponent", m.opponent)
    if news.USAGE["calls"]:
        print(f"\n[data] news LLM: {news.USAGE['calls']} calls, "
              f"{news.USAGE['input_tokens']} in / {news.USAGE['output_tokens']} out tokens")
