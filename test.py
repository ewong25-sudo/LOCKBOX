import os
from dotenv import load_dotenv
from espn_api.football import League

load_dotenv()
league = League(
    league_id=int(os.getenv("LEAGUE_ID")),
    year=2026,
    espn_s2=os.getenv("ESPN_S2"),
    swid=os.getenv("SWID"),
)

for m in league.box_scores():
    print(m.home_team.team_name, "vs", m.away_team.team_name)
