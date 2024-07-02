# Finds the best batch size using gradient noise scale
import argparse
import os

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from pearl.data import EpisodeData, NO_TEAM
from pearl.model import NextGoalPredictor, CarballTransformer
from pearl.train import SIZES


class GradientNoiseScale:
    """Calculates the gradient noise scale (1 / SNR), or critical batch size,
    from _An Empirical Model of Large-Batch Training_,
    https://arxiv.org/abs/1812.06162).

    Args:
        beta (float): The decay factor for the exponential moving averages used to
            calculate the gradient noise scale.
            Default: 0.9998
        eps (float): Added for numerical stability.
            Default: 1e-8
    """

    def __init__(self, beta=0.9998, eps=1e-8):
        self.beta = beta
        self.eps = eps
        self.ema_sq_norm = 0.
        self.ema_var = 0.
        self.beta_cumprod = 1.
        self.gradient_noise_scale = float('nan')

    def state_dict(self):
        """Returns the state of the object as a :class:`dict`."""
        return dict(self.__dict__.items())

    def load_state_dict(self, state_dict):
        """Loads the object's state.
        Args:
            state_dict (dict): object state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def update(self, sq_norm_small_batch, sq_norm_large_batch, n_small_batch, n_large_batch):
        """Updates the state with a new batch's gradient statistics, and returns the
        current gradient noise scale.

        Args:
            sq_norm_small_batch (float): The mean of the squared 2-norms of microbatch or
                per sample gradients.
            sq_norm_large_batch (float): The squared 2-norm of the mean of the microbatch or
                per sample gradients.
            n_small_batch (int): The batch size of the individual microbatch or per sample
                gradients (1 if per sample).
            n_large_batch (int): The total batch size of the mean of the microbatch or
                per sample gradients.
        """
        est_sq_norm = (n_large_batch * sq_norm_large_batch - n_small_batch * sq_norm_small_batch) / (
                n_large_batch - n_small_batch)
        est_var = (sq_norm_small_batch - sq_norm_large_batch) / (1 / n_small_batch - 1 / n_large_batch)
        est_sq_norm = abs(est_sq_norm)
        est_var = abs(est_var)
        self.ema_sq_norm = self.beta * self.ema_sq_norm + (1 - self.beta) * est_sq_norm
        self.ema_var = self.beta * self.ema_var + (1 - self.beta) * est_var
        self.beta_cumprod *= self.beta
        self.gradient_noise_scale = max(self.ema_var, self.eps) / max(self.ema_sq_norm, self.eps)
        return self.gradient_noise_scale

    def get_gns(self):
        """Returns the current gradient noise scale."""
        return self.gradient_noise_scale

    def get_stats(self):
        """Returns the current (debiased) estimates of the squared mean gradient
        and gradient variance."""
        return self.ema_sq_norm / (1 - self.beta_cumprod), self.ema_var / (1 - self.beta_cumprod)


def gradient_noise_scale(sq_norm_small_batch, sq_norm_large_batch, n_small_batch, n_large_batch):
    g2 = ((n_large_batch * sq_norm_large_batch - n_small_batch * sq_norm_small_batch)
          / (n_large_batch - n_small_batch))
    s = ((sq_norm_small_batch - sq_norm_large_batch)
         / (1 / n_small_batch - 1 / n_large_batch))
    g2 = abs(g2)
    s = abs(s)
    return s / g2


def main(size, dataset_folder, device=None,
         include_game_info=False, include_ties=False, predict_game_win=False):
    if device is None:
        device = 'cpu'  # 'cuda' if torch.cuda.is_available() else 'cpu'

    dim, num_layers, num_heads, ff_dim = SIZES[size]

    model = NextGoalPredictor(
        CarballTransformer(
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            include_game_info=include_game_info,
        ),
        include_ties=include_ties
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-10)
    loss_fn = CrossEntropyLoss()
    files = os.listdir(dataset_folder)

    gns_estimator = GradientNoiseScale()

    gradient_accumulation_steps = 16
    batch_size = 512
    for file in files:
        shard = EpisodeData.load(os.path.join(dataset_folder, file))
        shard.shuffle()

        i = 0
        while i < len(shard) - batch_size:
            grads = []
            for _ in range(gradient_accumulation_steps):
                optimizer.zero_grad()
                batch = shard[i:i + batch_size]
                x, y = batch.to_torch(device)
                y = y[int(predict_game_win)]
                if not include_ties:
                    y[y == NO_TEAM] = -100
                optimizer.zero_grad()
                y_pred = model(*x)
                loss = loss_fn(y_pred, y)
                if torch.isnan(loss).any():
                    raise ValueError('Loss is NaN')
                loss.backward()
                i += batch_size

                # Get the gradient
                grad = []
                for param in model.parameters():
                    grad.append(param.grad.flatten())
                grad = torch.cat(grad)
                grads.append(grad)

            # Calculate the gradient noise scale
            grads = torch.stack(grads)
            mean_grad = grads.mean(dim=0)
            sq_norm_large_batch = torch.norm(mean_grad) ** 2
            sq_norm_small_batch = (torch.norm(grads, dim=1) ** 2).mean()
            gns1 = gns_estimator.update(sq_norm_small_batch, sq_norm_large_batch,
                                        batch_size, batch_size * gradient_accumulation_steps)
            gns2 = gradient_noise_scale(sq_norm_small_batch, sq_norm_large_batch,
                                        batch_size, batch_size * gradient_accumulation_steps)
            print(f'Gradient noise scale: {gns2:.4f} (EMA: {gns1:.4f}, BS: {batch_size})')
            batch_size = round(float(gns1 / gradient_accumulation_steps))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=str, help='Size of the model to use')
    parser.add_argument('--dataset_folder', type=str, help='Folder containing the dataset')
    parser.add_argument('--device', type=str, default=None, help='Device to use')
    parser.add_argument('--include_game_info', action='store_true', help='Include game info in the model')
    parser.add_argument('--include_ties', action='store_true', help='Include ties in the loss')
    parser.add_argument('--predict_game_win', action='store_true', help='Predict game win instead of goal')
    args = parser.parse_args()
    main(
        size=args.size,
        dataset_folder=args.dataset_folder,
        device=args.device,
        include_game_info=args.include_game_info,
        include_ties=args.include_ties,
        predict_game_win=args.predict_game_win
    )
