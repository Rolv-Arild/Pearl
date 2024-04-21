import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW

from pearl.data import EpisodeData
from pearl.model import NextGoalPredictor, CarballTransformer
from pearl.train import SIZES
import argparse


def lr_plot(size, dataset_folder, batch_size=64, gradient_accumulation_steps=1, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dim, num_layers, num_heads, ff_dim = SIZES[size]

    model = NextGoalPredictor(
        CarballTransformer(
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
        ),
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-10)
    loss_fn = BCEWithLogitsLoss()
    files = os.listdir(dataset_folder)
    losses = []
    lrs = []
    for file in files:
        shard = EpisodeData.load(os.path.join(dataset_folder, file))
        shard.player_data[np.isnan(shard.player_data)] = 0.

        macro_batch_size = batch_size * gradient_accumulation_steps
        for i in range(0, len(shard) - macro_batch_size, macro_batch_size):
            tot_loss = 0
            for _ in range(gradient_accumulation_steps):
                batch = shard[i:i + batch_size]
                x, y = batch.to_torch(device)
                optimizer.zero_grad()
                y_pred = model(*x)
                loss = loss_fn(y_pred, y) / gradient_accumulation_steps
                loss.backward()
                tot_loss += loss.item()
                i += batch_size
            optimizer.step()
            losses.append(tot_loss)
            lrs.append(optimizer.param_groups[0]['lr'])
            optimizer.param_groups[0]['lr'] *= 1.1
            if lrs[-1] > 1 or losses[-1] > 10:
                break
        else:
            continue
        break

    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.show()
    debug = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=str, help='Size of the model to use')
    parser.add_argument('--dataset_folder', type=str, help='Folder containing the dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size to use')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate before performing a backward pass")
    parser.add_argument('--device', type=str, default=None, help='Device to use')
    args = parser.parse_args()
    lr_plot(args.size, args.dataset_folder, args.batch_size, args.gradient_accumulation_steps, args.device)
