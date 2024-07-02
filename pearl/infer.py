import argparse
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pearl.data import PlayerData
from pearl.model import NextGoalPredictor, CarballTransformer
from pearl.replay import ParsedReplay
from pearl.replay_to_data import replay_to_data
from pearl.shapley import shapley_value


def powerset(iterable):
    return itertools.chain.from_iterable(
        itertools.combinations(iterable, r)
        for r in range(len(iterable) + 1)
    )


def main(args):
    replay_dir = Path(args.replay_folder)
    replay_paths = [p.parent for p in replay_dir.glob("**/__game.parquet")]
    replay_paths += [p for p in replay_dir.glob("**/*.replay")]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    include_ties = args.include_ties

    model = NextGoalPredictor(
        CarballTransformer(
            dim=256,
            num_layers=4,
            num_heads=4,
            ff_dim=1024,
            include_game_info=False
        ),
        include_ties=include_ties,
    )
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    batch_size = args.batch_size

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    model.eval()
    with torch.no_grad(), tqdm(replay_paths) as pbar:
        for replay_path in pbar:
            out_path = os.path.join(args.save_path, replay_path.name)
            if os.path.exists(out_path):
                continue
            try:
                parsed_replay = ParsedReplay.load(replay_path)
            except FileNotFoundError:
                continue
            players = [p for p in parsed_replay.metadata["players"]
                       if p["unique_id"] in parsed_replay.player_dfs]
            player_ids = [p["online_id_kind"] + ":" + p["online_id"] for p in players]
            uid_to_pid = {int(p["unique_id"]): p["online_id_kind"] + ":" + p["online_id"] for p in players}

            goal_cols = ["logit_blue", "logit_orange"] + ["logit_tie"] * include_ties

            player_combinations = ["|".join(sorted(s)) for s in powerset(player_ids)]
            results = pd.DataFrame(index=parsed_replay.game_df.index,
                                   columns=(goal_cols
                                            + [f"{pid}:team" for pid in player_ids]
                                            + [f"{pid}:age" for pid in player_ids]))
            results[:] = np.nan

            starts_ends = []
            gameplay_periods = parsed_replay.analyzer["gameplay_periods"]
            for gameplay_period in gameplay_periods:
                start_frame = gameplay_period["start_frame"]
                goal_frame = gameplay_period["goal_frame"]

                if goal_frame is None:
                    end_frame = gameplay_period["end_frame"]
                else:
                    end_frame = goal_frame

                starts_ends.append((start_frame, end_frame))

            episodes = replay_to_data(parsed_replay, ignore_unfinished=False)
            predictions = []
            for i in range(0, len(episodes), batch_size):
                batch = episodes[i:i + batch_size]
                x, y = batch.to_torch(device=device)
                y_hat = model(*x)
                predictions.append(y_hat)
            predictions = torch.cat(predictions).cpu().numpy()
            episodes.save(out_path + "_episodes.npz")
            np.save(out_path + "_predictions.npy", predictions)


def calculate_shapley_values(results):
    def get_result_fn(players):
        players = "|".join(sorted(players))
        return results[players]

    all_ids = max(results.columns, key=lambda x: len(x))
    shapley_values = shapley_value(get_result_fn, all_ids.split("|"))
    return shapley_values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay_folder", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--use_ignore", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--include_ties", action="store_true")
    args = parser.parse_args()
    main(args)
