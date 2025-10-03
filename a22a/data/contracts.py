"""
Schema & join-key assertions (stubs for Phases 1–2).
Implement concrete checks once data frames are available.
"""
from dataclasses import dataclass

@dataclass
class TableContract:
    name: str
    required_cols: list
    primary_key: list
    foreign_keys: dict | None = None

PBP_CONTRACT = TableContract(
    name="pbp",
    required_cols=["game_id","play_id","season","week","posteam","defteam"],
    primary_key=["game_id","play_id"],
    foreign_keys={"game_id":"games"}
)

GAMES_CONTRACT = TableContract(
    name="games",
    required_cols=["game_id","season","week","kickoff_datetime","home_team","away_team"],
    primary_key=["game_id"]
)
