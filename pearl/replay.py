import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd


@dataclass
class ParsedReplay:
    metadata: dict
    analyzer: dict
    game_df: pd.DataFrame
    ball_df: pd.DataFrame
    player_dfs: Dict[str, pd.DataFrame]

    @staticmethod
    def load(replay_dir) -> "ParsedReplay":
        if isinstance(replay_dir, str):
            replay_dir = Path(replay_dir)
        with (replay_dir / "metadata.json").open("r", encoding="utf8") as f:
            metadata = json.load(f)
        with (replay_dir / "analyzer.json").open("r", encoding="utf8") as f:
            analyzer = json.load(f)
        ball_df = load_parquet(replay_dir / "__ball.parquet")
        game_df = load_parquet(replay_dir / "__game.parquet")

        player_dfs = {}
        for player_file in replay_dir.glob("player_*.parquet"):
            player_id = player_file.name.split("_")[1].split(".")[0]
            player_dfs[player_id] = load_parquet(player_file)

        return ParsedReplay(metadata, analyzer, game_df, ball_df, player_dfs)


def load_parquet(*args, **kwargs):
    return pd.read_parquet(*args, engine="pyarrow", **kwargs)
