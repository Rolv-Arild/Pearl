import argparse
import os

import numpy as np
import torch
import wandb
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm

from pearl.data import EpisodeData, NO_TEAM, GameInfo
from pearl.metrics import Accuracy, AccuracyAtNSec, EpisodeUniqueness, NoMaskMetric, \
    CalibrationScore, PredictionVariance
from pearl.model import NextGoalPredictor, CarballTransformer

SIZES = {
    "tiny": (128, 2, 2, 512),
    "mini": (256, 4, 4, 1024),
    "small": (512, 4, 8, 2048),
    "medium": (512, 8, 8, 2048),
    "base": (768, 12, 12, 3072),
    "large": (1024, 24, 16, 4096),
}


class NGPTrainer:
    def __init__(self, name, dataset_dir: str, save_path: str, batch_size: int, learning_rate: float, size: str,
                 include_scoreboard: bool, include_ties: bool, predict_win: bool,
                 gradient_accumulation_steps: int = 1,
                 augment=True, mask=None, device=None, validate_every=1000,
                 seed=123):
        self.dataset_dir = dataset_dir
        self.save_path = save_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if size in SIZES:
            dim, num_layers, num_heads, ff_dim = SIZES[size]
        else:
            dim, num_layers, num_heads, ff_dim = (int(v) for v in size.split(","))

        assert not predict_win or include_scoreboard, "Can't predict win without scoreboard"
        assert not include_ties or include_scoreboard, "Can't include ties without scoreboard"

        self.include_scoreboard = include_scoreboard
        self.include_ties = include_ties
        self.predict_win = predict_win

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.augment = augment
        self.mask = mask

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.validate_every = validate_every

        self.model = NextGoalPredictor(
            CarballTransformer(
                dim=dim,
                num_layers=num_layers,
                num_heads=num_heads,
                ff_dim=ff_dim,
                include_game_info=include_scoreboard
            ),
            include_ties=include_ties,
        ).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.loss_fn = CrossEntropyLoss()
        self.metrics = {
            k: [
                   Accuracy(),
                   AccuracyAtNSec(1),  # Should get to 100%, otherwise something is wrong (impossible to cross field)
                   AccuracyAtNSec(5),
                   AccuracyAtNSec(10),
                   AccuracyAtNSec(30),
                   # NormalizedBrierScore(),
                   EpisodeUniqueness(),
                   CalibrationScore(),
                   PredictionVariance(),
               ] + ([
                        NoMaskMetric(Accuracy()),
                        NoMaskMetric(AccuracyAtNSec(1)),
                        NoMaskMetric(AccuracyAtNSec(5)),
                        NoMaskMetric(AccuracyAtNSec(10)),
                        NoMaskMetric(AccuracyAtNSec(30)),
                        # NoMaskMetric(NormalizedBrierScore()),
                        NoMaskMetric(CalibrationScore()),
                    ] if mask is not None else [])
            for k in ["train", "val"]
        }
        if name is None:
            n = 1
            prefix = "nwp" if predict_win else "ngp"
            if include_ties:
                prefix += "t"
            while True:
                name = f"{prefix}-{size}-{n}"
                if name not in os.listdir(save_path):
                    break
                n += 1
        os.makedirs(os.path.join(self.save_path, name))
        self.logger = wandb.init(
            name=name,
            project="next-goal-predictor",
            config={
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "dim": dim,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "ff_dim": ff_dim,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "augment": augment,
                "mask": mask,
                "seed": seed,
                "include_scoreboard": include_scoreboard,
                "include_ties": include_ties,
                "predict_win": predict_win,
            })

        files = [f for f in os.listdir(dataset_dir) if "shard" in f]
        self.train_files = [f for f in files if "train" in f]
        self.val_files = [f for f in files if "validation" in f]
        if len(self.train_files) == 0 or len(self.val_files) == 0:
            self.train_files = files[:-1]
            self.val_files = files[-1:]

        self.epoch = 0
        self.n_updates = 0
        self.min_loss = np.inf

        self.seed_everything(seed)

    @staticmethod
    def seed_everything(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def validate(self):
        # For validation, we calculate loss and metric across the whole split
        pbar = tqdm(self.val_files, desc="Validating")
        val_metrics = self.metrics["val"]
        for metric in val_metrics:
            metric.reset()
        total_loss = 0
        n = 0
        self.model.eval()
        with torch.no_grad():
            for file in pbar:
                shard: EpisodeData = EpisodeData.load(os.path.join(self.dataset_dir, file))
                # TODO: Fix this in the data generation
                shard = shard[~np.isnan(shard.game_info[:, GameInfo.TIME_REMAINING])]

                if self.augment:
                    shard.normalize_ball_quadrant()
                    # shard.swap_teams("random")
                    # shard.mirror_x("random")
                if self.mask is not None:
                    shard.mask_randomly(self.mask, rng=np.random.default_rng(123))

                for i in range(0, len(shard), self.batch_size):
                    batch = shard[i:i + self.batch_size]
                    x, y = batch.to_torch(self.device)
                    y = y[int(self.predict_win)]
                    if not self.include_ties:
                        y[y == NO_TEAM] = -100
                    y_pred = self.model(*x)
                    loss = self.loss_fn(y_pred, y)
                    total_loss += loss.item() * len(batch)
                    n += len(batch)
                    for metric in val_metrics:
                        metric.submit(y, y_pred, batch)

        avg_loss = total_loss / n
        if avg_loss < self.min_loss:
            self.min_loss = avg_loss
            torch.save(self.model.state_dict(), os.path.join(self.save_path, self.logger.name, "best.pth"))

        # Save the entire training state
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "n_updates": self.n_updates,
            "min_loss": self.min_loss,
        }, os.path.join(self.save_path, self.logger.name, f"checkpoint-{self.n_updates}.pth"))

        metrics = {str(metric): metric.calculate() for metric in val_metrics}

        self.logger.log({"val/loss": avg_loss}, commit=False)
        self.logger.log({f"val/{str(metric)}": value for metric, value in metrics.items()}, commit=False)
        print(f"{self.n_updates}\n\tValidation loss: {avg_loss:.4f}\n\t{metrics}")

    def train(self):
        # For training, we calculate loss and metric per batch
        pbar = tqdm(desc="Training", total=self.validate_every)
        train_metrics = self.metrics["train"]
        while True:
            self.model.train()
            for file in self.train_files:
                shard: EpisodeData = EpisodeData.load(os.path.join(self.dataset_dir, file))
                shard.shuffle()
                if self.augment:
                    shard.normalize_ball_quadrant()
                    # shard.swap_teams("random")
                    # shard.mirror_x("random")
                if self.mask is not None:
                    shard.mask_randomly(self.mask)
                # TODO: Fix this in the data generation
                shard = shard[~np.isnan(shard.game_info[:, GameInfo.TIME_REMAINING])]

                macro_batch_size = self.batch_size * self.gradient_accumulation_steps
                for i in range(0, len(shard) - macro_batch_size, macro_batch_size):
                    tot_loss = 0
                    metrics = {}
                    for metric in train_metrics:
                        metric.reset()

                    for _ in range(self.gradient_accumulation_steps):
                        batch = shard[i:i + self.batch_size]
                        x, y = batch.to_torch(self.device)
                        y = y[int(self.predict_win)]
                        if not self.include_ties:
                            y[y == NO_TEAM] = -100
                        self.optimizer.zero_grad()
                        y_pred = self.model(*x)
                        loss = self.loss_fn(y_pred, y) / self.gradient_accumulation_steps
                        if torch.isnan(loss).any():
                            raise ValueError("Loss is NaN")

                        for metric in train_metrics:
                            metric.submit(y, y_pred, batch)

                        tot_loss += loss.item()
                        loss.backward()
                        i += self.batch_size

                    self.optimizer.step()
                    self.n_updates += 1

                    for metric in train_metrics:
                        metrics["train/" + str(metric)] = metric.calculate()

                    pbar.update(1)
                    pbar.set_postfix_str(f"Loss: {tot_loss:.4f}, File: {file}, Epoch: {self.epoch}")

                    if self.n_updates % self.validate_every == 0:
                        self.validate()
                        self.model.train()
                        pbar.reset()

                    self.logger.log({
                        "train/loss": tot_loss,
                        "train/epoch": self.epoch,
                        "train/samples": self.n_updates * macro_batch_size,
                        **metrics
                    })
            self.epoch += 1


def main(args):
    trainer = NGPTrainer(
        name=None,
        dataset_dir=args.dataset_dir,
        save_path=args.save_path,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        size=args.size,
        include_scoreboard=args.include_scoreboard,
        include_ties=args.include_ties,
        predict_win=args.predict_win,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        augment=not args.no_augment,
        mask=args.mask,
        device=args.device,
        validate_every=args.validate_every,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--size", type=str, required=True)
    parser.add_argument("--include_scoreboard", action="store_true")
    parser.add_argument("--include_ties", action="store_true")
    parser.add_argument("--predict_win", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--validate_every", type=int, default=1000)
    args = parser.parse_args()
    main(args)
