import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd


# TODO remove file

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


# class ParsedReplay:  # New version with on-demand loading via properties
#     def __init__(self, replay_dir=None, metadata=None, analyzer=None, game_df=None, ball_df=None, player_dfs=None):
#         if isinstance(replay_dir, str):
#             replay_dir = Path(replay_dir)
#         self._replay_dir = replay_dir
#         self._metadata = metadata
#         self._analyzer = analyzer
#         self._game_df = game_df
#         self._ball_df = ball_df
#         self._player_dfs = player_dfs
#
#     @staticmethod
#     def load(replay_dir) -> "ParsedReplay":
#         return ParsedReplay(replay_dir)
#
#     @property
#     def metadata(self):
#         if self._metadata is None:
#             with (self._replay_dir / "metadata.json").open("r", encoding="utf8") as f:
#                 self._metadata = json.load(f)
#         return self._metadata
#
#     @property
#     def analyzer(self):
#         if self._analyzer is None:
#             with (self._replay_dir / "analyzer.json").open("r", encoding="utf8") as f:
#                 self._analyzer = json.load(f)
#         return self._analyzer
#
#     @property
#     def game_df(self):
#         if self._game_df is None:
#             self._game_df = load_parquet(self._replay_dir / "__game.parquet")
#         return self._game_df
#
#     @property
#     def ball_df(self):
#         if self._ball_df is None:
#             self._ball_df = load_parquet(self._replay_dir / "__ball.parquet")
#         return self._ball_df
#
#     @property
#     def player_dfs(self):
#         if self._player_dfs is None:
#             self._player_dfs = {}
#             for player_file in self._replay_dir.glob("player_*.parquet"):
#                 player_id = player_file.name.split("_")[1].split(".")[0]
#                 self._player_dfs[player_id] = load_parquet(player_file)
#         return self._player_dfs


def load_parquet(*args, **kwargs):
    return pd.read_parquet(*args, engine="pyarrow", **kwargs)
