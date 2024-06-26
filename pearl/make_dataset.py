import argparse
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

from pearl.data import EpisodeData
from pearl.replay_to_data import replay_to_data
from rlgym_tools.replays.parsed_replay import ParsedReplay


def process_replay(replay_path):
    try:
        parsed_replay = ParsedReplay.load(replay_path)
        episodes = replay_to_data(parsed_replay)
        return episodes
    except Exception as err:
        print(f"Failed to load {replay_path}: {err}")
        return None


def main(replay_dir: str, output_dir: str, shard_size: int):
    replay_dir = Path(replay_dir)
    output_dir = Path(output_dir)

    shard = EpisodeData.new_empty(shard_size, normalized=True)
    n = 0
    replay_paths = [p.parent for p in replay_dir.glob("**/__game.parquet")]  # Either parsed replay
    replay_paths += [p for p in replay_dir.glob("**/*.replay")]  # Or unparsed replay

    random.shuffle(replay_paths)
    pbar = tqdm(replay_paths, "Processing replays")
    i = 0
    e = 0
    with ProcessPoolExecutor() as ex:
        for replay_path, episodes in zip(pbar, ex.map(process_replay, replay_paths)):
            if episodes is None:
                continue
            next_i = i + len(episodes)
            if next_i > shard_size:
                shard = shard[:i]
                shard.save(output_dir / f"shard_{n}.npz")
                shard = EpisodeData.new_empty(shard_size, normalized=True)
                n += 1
                i = 0
                next_i = len(episodes)
            episodes.episode_id[:] += e
            e = episodes.episode_id[-1] + 1
            shard[i:next_i] = episodes
            i = next_i
            pbar.set_postfix_str(f"replay={replay_path.name}, n={n}, frames={i:_}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--shard_size", type=int, default=30 * 60 * 60 * 24)
    args = parser.parse_args()
    main(
        replay_dir=args.replay_dir,
        output_dir=args.output_dir,
        shard_size=args.shard_size
    )
