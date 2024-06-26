import math
import os
import shutil
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from pearl.data import EpisodeData


def load_episode_data(file):
    return EpisodeData.load(file)


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    files = os.listdir(input_dir)  # Keep the last file as validation
    files = [os.path.join(input_dir, file) for file in files if file.endswith(".npz")]
    train_files = files[:-1]
    val_files = files[-1:]
    # Get the dataset size first
    sizes = []
    pbar = tqdm(train_files, desc="Getting dataset size")
    for file in pbar:
        try:
            shard = EpisodeData.load(file)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            raise e
        sizes.append(len(shard))
        pbar.set_postfix_str(f"{file}, size={len(shard)}")
    total_samples = sum(sizes)
    print(f"Total samples: {total_samples}")

    # Make a global list of shuffled indices
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    # Now, we need to make new shards with globally shuffled data.
    # This is O(n^2) but it only requires two shards in memory at a time.
    target_size = math.ceil(total_samples / len(train_files))  # Make sure we don't have any nearly empty shards
    current_shard = 0
    current_index = 0

    print("Shuffling dataset")
    while current_index < total_samples:
        new_shard = EpisodeData.new_empty(target_size, normalized=True)
        new_shard.ball_data[:] = np.nan  # So we can check if we missed any samples at the end
        new_to_old = indices[current_index:current_index + target_size]
        # E.g. new_to_old[0] = 5 means that the first sample in the new shard is the 5th sample in all the old shards

        # Remove empty samples from the end so the indices match
        new_shard = new_shard[:len(new_to_old)]

        start_index = 0
        # Parallelize
        with ThreadPoolExecutor(1) as executor:
            pbar = tqdm(train_files, desc=f"Shard {current_shard}")
            exmap = executor.map(load_episode_data, train_files)
            for n, (file, old_shard) in enumerate(zip(pbar, exmap)):
                end_index = start_index + len(old_shard)

                # Find the samples that are in the new shard
                mask = np.logical_and(new_to_old >= start_index, new_to_old < end_index)
                new_indices = new_to_old[mask] - start_index
                new_shard[mask] = old_shard[new_indices]

                start_index = end_index

            # Check if we missed any samples
            if np.isnan(new_shard.ball_data).any():
                debug = True
                raise ValueError("Missed samples")

            print(f"Saving shard {current_shard} with {len(new_shard)} samples")
            new_shard.save(os.path.join(output_dir, f"train_shard_{current_shard}.npz"))
            current_shard += 1
            current_index += target_size

    # Copy validation files
    for n, file in enumerate(val_files):
        shutil.copy(file, os.path.join(output_dir, f"validation_shard_{n}.npz"))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    main(args)
