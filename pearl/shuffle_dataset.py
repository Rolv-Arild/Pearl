import math
import os
import shutil
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

from pearl.data import EpisodeData, GameInfo


def load_episode_data(file, target_size=None):
    if os.path.isfile(file):
        return EpisodeData.load(file)
    dummy = EpisodeData.new_empty(target_size, normalized=True)
    dummy.game_info[:, GameInfo.IGNORE] = np.nan
    return dummy


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    shard_size = args.shard_size

    while True:
        files = os.listdir(input_dir)
        files = [os.path.join(input_dir, file) for file in files if file.endswith(".npz")]
        if len(files) >= 240:
            break
    train_files = files[:-1]
    val_files = files[-1:]  # Keep the last file as validation

    # Now, we need to make new shards with globally shuffled data.
    # We use a butterfly shuffle, so the complexity is O(n * log(n))

    # Initialize the outputs as the original files
    print("Copying files")
    for n, file in enumerate(train_files):
        shutil.copy(file, os.path.join(output_dir, f"tmp_shard_{n}.npz"))

    num_shards = len(train_files)
    w = 1
    lim = math.ceil(np.log2(num_shards))
    swapped = [(n,) for n in range(2 ** lim)]
    pbar = tqdm(list(range(lim)), "Shuffling shards")
    for p in pbar:
        # Note that we use ceil instead of floor, so we don't lose any data.
        # This means we will encounter shards that don't exist, which we will fill with dummy data.
        # The dummy data is removed later on
        with ProcessPoolExecutor(16) as ex:
            futures = []
            t = 0
            while t < num_shards:
                s = 0
                while s < w:
                    a = s + t
                    b = a + w
                    pbar.set_postfix_str(f"Swapping {a} and {b} (t={t}, s={s})")
                    path_a = os.path.join(output_dir, f"tmp_shard_{a}.npz")
                    path_b = os.path.join(output_dir, f"tmp_shard_{b}.npz")
                    # mix_shards(path_a, path_b, shard_size)
                    future = ex.submit(mix_shards, path_a, path_b, shard_size)
                    futures.append(future)

                    res: tuple = swapped[a] + swapped[b]
                    assert len(set(res)) == len(res), f"Duplicate shards in swap: {res}"
                    swapped[a] = res
                    swapped[b] = res

                    s += 1
                t += 2 * w
            for f, future in enumerate(futures):
                pbar.set_postfix_str(f"Future progress: {f / len(futures):.1%}")
                future.result()
        w *= 2

    # Remove dummy data
    n = 0
    i = 0
    out_shard = EpisodeData.new_empty(shard_size, normalized=True)
    files = [os.path.join(output_dir, shard_file) for shard_file in os.listdir(output_dir)
             if shard_file.startswith("tmp_shard_")]
    with ProcessPoolExecutor(16) as ex:
        pbar = tqdm(ex.map(load_episode_data, files), "Removing dummy data")
        for shard, fpath in zip(pbar, files):
            shard = shard[~np.isnan(shard.game_info[:, GameInfo.IGNORE])]
            if i + len(shard) > shard_size:
                out_shard[i:] = shard[:shard_size - i]
                shard = shard[shard_size - i:]
                out_shard.save(os.path.join(output_dir, f"training_shard_{n}.npz"))
                n += 1
                out_shard = EpisodeData.new_empty(shard_size, normalized=True)
                i = 0
            out_shard[i:i + len(shard)] = shard
            i += len(shard)
            pbar.set_postfix_str(f"Shard {n}, frames {i:_}")
            os.remove(fpath)
        if i > 0:
            out_shard[:i].save(os.path.join(output_dir, f"training_shard_{n}.npz"))

    # Clean up any remaining intermediate files
    print("Cleaning up")
    files = os.listdir(output_dir)
    assert len(files) == len(train_files)
    for shard_file in files:
        if shard_file.startswith("tmp_shard_"):
            os.remove(os.path.join(output_dir, shard_file))

    # Copy validation files
    print("Copying validation files")
    for n, file in enumerate(val_files):
        shutil.copy(file, os.path.join(output_dir, f"validation_shard_{n}.npz"))


def mix_shards(path_a, path_b, shard_size):
    shard_a = load_episode_data(path_a, shard_size)
    shard_b = load_episode_data(path_b, shard_size)
    total_shard = shard_a + shard_b
    total_shard.shuffle()
    mid = len(total_shard) // 2
    shard_a = total_shard[:mid]
    shard_b = total_shard[mid:]
    shard_a.save(path_a)
    shard_b.save(path_b)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--shard_size", type=int, default=30 * 60 * 60 * 24)
    args = parser.parse_args()
    main(args)
