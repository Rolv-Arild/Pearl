import numpy as np
import torch

from pearl.data import EpisodeData, BallData, PlayerData, BoostData, GameInfo


def _divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return float("nan")


class BaseMetric:
    def __init__(self, name: str):
        self.name = name

    def reset(self):
        raise NotImplementedError

    def submit(self, y_true: torch.Tensor, y_pred: torch.Tensor, episode_data: EpisodeData):
        raise NotImplementedError

    def calculate(self):
        raise NotImplementedError

    def __str__(self):
        return self.name


class Accuracy(BaseMetric):
    def __init__(self):
        super().__init__("accuracy")
        self.correct = 0
        self.total = 0

    def reset(self):
        self.correct = 0
        self.total = 0

    def submit(self, y_true: torch.Tensor, y_pred: torch.Tensor, episode_data: EpisodeData):
        mask = y_true != -100
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        self.correct += torch.sum(y_pred.argmax(dim=-1) == y_true).item()
        self.total += len(y_true)

    def calculate(self):
        return _divide(self.correct, self.total)


class AccuracyAtNSec(BaseMetric):
    def __init__(self, n_sec: int):
        super().__init__("accuracy@{}s".format(n_sec))
        self.n_sec = n_sec
        self.correct = 0
        self.total = 0

    def reset(self):
        self.correct = 0
        self.total = 0

    def submit(self, y_true: torch.Tensor, y_pred: torch.Tensor, episode_data: EpisodeData):
        mask = abs(episode_data.time_until_end - self.n_sec) < 0.5  # This one is numpy
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        mask = ((y_true == 0) | (y_true == 1))  # While these are torch
        y_true = y_true[mask]
        y_pred = y_pred[mask][:, [0, 1]]
        self.correct += torch.sum(y_pred.argmax(dim=-1) == y_true).item()
        self.total += len(y_true)

    def calculate(self):
        return _divide(self.correct, self.total)


class EpisodeUniqueness(BaseMetric):
    def __init__(self):
        super().__init__("episode_uniqueness")
        self.episode_ids = np.array([])
        self.total = 0

    def reset(self):
        self.episode_ids = np.array([])
        self.total = 0

    def submit(self, y_true: torch.Tensor, y_pred: torch.Tensor, episode_data: EpisodeData):
        diff = np.setdiff1d(episode_data.episode_id, self.episode_ids)
        self.episode_ids = np.concatenate([self.episode_ids, diff])
        self.total += len(episode_data.episode_id)

    def calculate(self):
        return _divide(len(self.episode_ids), self.total)


class NormalizedBrierScore(BaseMetric):
    # The normalized Brier score adjusts the Brier score by also figuring out
    # the maximum and minimum possible Brier scores for a set of predictions.
    def __init__(self):
        super().__init__("normalized_brier_score")
        self.brier_score = 0
        self.total = 0
        self.max_brier_score = 0
        self.min_brier_score = 0

    def reset(self):
        self.brier_score = 0
        self.total = 0
        self.max_brier_score = 0
        self.min_brier_score = 0

    def submit(self, y_true: torch.Tensor, y_pred: torch.Tensor, episode_data: EpisodeData):
        mask = (y_true == 0) | (y_true == 1)
        y_true = y_true[mask]
        y_pred = y_pred[mask][:, [0, 1]]
        y_pred = torch.softmax(y_pred, dim=1)[:, 1]
        self.brier_score += torch.sum((y_pred - y_true) ** 2).item()
        self.total += len(y_true)

        # Maximum Brier score is always predicting the class that maximizes the error
        max_brier = torch.sum(torch.max(y_pred, 1 - y_pred) ** 2).item()
        self.max_brier_score += max_brier
        # And the minimum is when for every sample, the probability of the true label is the model prediction
        # p*(1-p)^2 + (1-p)*p^2 = (1-p)*p
        min_brier = torch.sum((1 - y_pred) * y_pred).item()
        self.min_brier_score += min_brier

    def calculate(self):
        return (self.brier_score - self.min_brier_score) / (self.max_brier_score - self.min_brier_score)


class CalibrationScore(BaseMetric):
    def __init__(self, bin_count=10):
        super().__init__(f"calibration_score_{bin_count}_bins")
        self.bin_count = bin_count
        self.total_preds = np.zeros(self.bin_count)
        self.total_counts = np.zeros((self.bin_count, 2))

    def reset(self):
        self.total_preds = np.zeros(self.bin_count)
        self.total_counts = np.zeros((self.bin_count, 2))

    def submit(self, y_true: torch.Tensor, y_pred: torch.Tensor, episode_data: EpisodeData):
        has_goal = (y_true == 0) | (y_true == 1)
        y_pred = torch.softmax(y_pred[:, [0, 1]], dim=1)[:, 1]
        bin_edges = np.linspace(0, 1, self.bin_count + 1)
        i = 0
        for bin_start, bin_end in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (y_pred >= bin_start) & (y_pred <= bin_end) & has_goal
            preds = y_pred[mask]
            labels = y_true[mask]
            if len(labels) > 0:
                self.total_preds[i] += preds.sum().item()
                self.total_counts[i] += [torch.sum(labels < 0.5).item(), torch.sum(labels > 0.5).item()]
            i += 1

    def calculate(self):
        error = 0
        for i in range(self.bin_count):
            count = self.total_counts[i].sum()
            if count > 0:
                expected = self.total_preds[i] / count  # Average prediction in this bin
                actual = self.total_counts[i][1] / count  # Fraction of positive labels
                error += count * (expected - actual) ** 2
        return _divide(error, self.total_counts.sum())


class PredictionVariance(BaseMetric):
    def __init__(self):
        super().__init__("prediction_variance")
        self.n = 0
        self.sum = 0
        self.sum_sq = 0

    def reset(self):
        self.n = 0
        self.sum = 0
        self.sum_sq = 0

    def submit(self, y_true: torch.Tensor, y_pred: torch.Tensor, episode_data: EpisodeData):
        y_pred = y_pred[:, 1] - y_pred[:, 0]
        self.n += len(y_pred)
        self.sum += y_pred.sum().item()
        self.sum_sq += (y_pred ** 2).sum().item()

    def calculate(self):
        var = (self.sum_sq - (self.sum * self.sum) / self.n) / (self.n - 1)
        return var


class NoMaskMetric(BaseMetric):
    def __init__(self, metric: BaseMetric):
        super().__init__("no_mask/" + metric.name)
        self.metric = metric

    def reset(self):
        self.metric.reset()

    def submit(self, y_true: torch.Tensor, y_pred: torch.Tensor, episode_data: EpisodeData):
        masked = np.concatenate([
            episode_data.game_info[:, GameInfo.MASK].reshape(-1, 1),
            episode_data.ball_data[:, :, BallData.MASK],
            episode_data.player_data[:, :, PlayerData.MASK],
            episode_data.boost_data[:, :, BoostData.MASK],
        ], axis=1).any(axis=1)
        y_true = y_true[~masked]
        y_pred = y_pred[~masked]
        self.metric.submit(y_true, y_pred, episode_data[~masked])

    def calculate(self):
        return self.metric.calculate()
