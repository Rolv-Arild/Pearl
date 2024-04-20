import argparse
import random
from pathlib import Path

from tqdm import tqdm

from pearl.data import replay_to_data, EpisodeData
from pearl.replay import ParsedReplay


def main(replay_dir: str, output_dir: str, shard_size: int):
    replay_dir = Path(replay_dir)
    output_dir = Path(output_dir)

    shard = EpisodeData.new_empty(shard_size, normalized=True)
    n = 0
    replay_paths = [p for p in replay_dir.glob("**/__game.parquet")]
    random.shuffle(replay_paths)
    pbar = tqdm(replay_paths, "Processing replays")
    i = 0
    e = 0
    for replay_path in pbar:
        replay_path = replay_path.parent
        try:
            parsed_replay = ParsedReplay.load(replay_path)
            for episode in replay_to_data(parsed_replay):
                next_i = i + len(episode)
                if next_i > shard_size:
                    shard = shard[:i]
                    shard.save(output_dir / f"shard_{n}.npz")
                    shard = EpisodeData.new_empty(shard_size, normalized=True)
                    n += 1
                    i = 0
                    next_i = len(episode)
                episode.episode_id[:] = e
                e += 1
                shard[i:next_i] = episode
                i = next_i
            pbar.set_postfix_str(f"replay={replay_path.name}, n={n}, frames={i:_}")
        except Exception as err:
            print(f"Failed to load {replay_path}: {err}")
            continue


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
