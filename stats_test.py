import nflreadpy as nfl

stats = nfl.load_player_stats([2026])
print(stats.head())
