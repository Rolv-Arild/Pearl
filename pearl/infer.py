import argparse
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pearl.data import replay_to_data
from pearl.replay import ParsedReplay
from pearl.shapley import shapley_value


def powerset(iterable):
    return itertools.chain.from_iterable(
        itertools.combinations(iterable, r)
        for r in range(len(iterable) + 1)
    )


def main(args):
    replay_dir = Path(args.parsed_replay_folder)
    replay_paths = [p.parent for p in replay_dir.glob("**/__game.parquet")]

    model = torch.jit.load(args.model_path)
    batch_size = args.batch_size

    with torch.no_grad(), tqdm(replay_paths) as pbar:
        for replay_path in pbar:
            parsed_replay = ParsedReplay.load(replay_path)
            data = replay_to_data(parsed_replay, ignore_unfinished=False)

            players = [p for p in parsed_replay.metadata["players"]
                       if p["unique_id"] in parsed_replay.player_dfs]
            player_ids = [p["online_id_kind"] + ":" + p["online_id"] for p in players]

            player_combinations = ["|".join(sorted(s)) for s in powerset(player_ids)]
            results = pd.DataFrame(index=parsed_replay.game_df.index, columns=player_combinations)
            results[:] = np.nan

            starts_ends = []
            gameplay_periods = parsed_replay.analyzer["gameplay_periods"]
            for gameplay_period in gameplay_periods:
                start_frame = gameplay_period["start_frame"]
                goal_frame = gameplay_period["goal_frame"]

                if goal_frame is None:
                    end_frame = gameplay_period["end_frame"]

                starts_ends.append((start_frame, end_frame))

            for (start, end), episode in zip(starts_ends, data):
                for comb, masked in episode.mask_combinations(False, True, False):
                    masked_players = set(comb[0])
                    predictions = []
                    for i in range(len(masked)):
                        batch = masked[i:i + batch_size]
                        x, y = batch.to_torch(device=args.device)
                        y_hat = model(x)
                        predictions.append(y_hat)
                    predictions = torch.cat(predictions).cpu().numpy()
                    players_present = [pid for i, pid in enumerate(player_ids) if i not in masked_players]
                    players_present = "|".join(sorted(players_present))
                    results.loc[start:end, players_present] = predictions

            results.to_parquet(os.path.join(args.save_path, replay_path.name + ".parquet"))


def calculate_shapley_values(results):
    def get_result_fn(players):
        players = "|".join(sorted(players))
        return results[players]

    all_ids = max(results.columns, key=lambda x: len(x))
    shapley_values = shapley_value(get_result_fn, all_ids.split("|"))
    return shapley_values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed_replay_folder", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--device", type=str, required=True)
    args = parser.parse_args()
    main(args)
