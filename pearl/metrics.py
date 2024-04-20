import numpy as np
import torch

from pearl.data import EpisodeData, BallData, PlayerData, BoostData


class BaseMetric:
    def __init__(self, name: str):
        self.name = name

    def reset(self):
        raise NotImplementedError

    def submit(self, y_true, y_pred, episode_data: EpisodeData):
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

    def submit(self, y_true, y_pred, episode_data: EpisodeData):
        self.correct += torch.sum((y_pred > 0) == y_true).item()
        self.total += len(y_true)

    def calculate(self):
        return self.correct / self.total


class AccuracyAtNSec(BaseMetric):
    def __init__(self, n_sec: int):
        super().__init__("accuracy@{}s".format(n_sec))
        self.n_sec = n_sec
        self.correct = 0
        self.total = 0

    def reset(self):
        self.correct = 0
        self.total = 0

    def submit(self, y_true, y_pred, episode_data: EpisodeData):
        mask = episode_data.time_until_end <= self.n_sec
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        self.correct += torch.sum((y_pred > 0) == y_true).item()
        self.total += len(y_true)

    def calculate(self):
        return self.correct / self.total


class EpisodeUniqueness(BaseMetric):
    def __init__(self):
        super().__init__("episode_uniqueness")
        self.episode_ids = np.array([])
        self.total = 0

    def reset(self):
        self.episode_ids = np.array([])
        self.total = 0

    def submit(self, y_true, y_pred, episode_data: EpisodeData):
        diff = np.setdiff1d(episode_data.episode_id, self.episode_ids)
        self.episode_ids = np.concatenate([self.episode_ids, diff])
        self.total += len(episode_data.episode_id)

    def calculate(self):
        return len(self.episode_ids) / self.total


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

    def submit(self, y_true, y_pred, episode_data: EpisodeData):
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
        self.bin_correct = np.zeros(self.bin_count)
        self.bin_total = np.zeros(self.bin_count)

    def reset(self):
        self.bin_correct = np.zeros(self.bin_count)
        self.bin_total = np.zeros(self.bin_count)

    def submit(self, y_true, y_pred, episode_data: EpisodeData):
        bin_edges = torch.linspace(0, 1, self.bin_count + 1).to(y_pred.device)
        bin_indices = torch.bucketize(y_pred, bin_edges) - 1
        for i in range(self.bin_count):
            mask = bin_indices == i
            self.bin_correct[i] += torch.sum((y_pred[mask] > 0) == y_true[mask]).item()
            self.bin_total[i] += len(y_true[mask])

    def calculate(self):
        bin_accuracy = self.bin_correct / self.bin_total
        return np.mean(np.abs(bin_accuracy - np.linspace(0, 1, self.bin_count)))


class Correlation(BaseMetric):
    def __init__(self):
        super().__init__("correlation")
        self.y_true = np.array([])
        self.y_pred = np.array([])

    def reset(self):
        self.y_true = np.array([])
        self.y_pred = np.array([])

    def submit(self, y_true, y_pred, episode_data: EpisodeData):
        self.y_true = np.concatenate([self.y_true, y_true.cpu().numpy()])
        self.y_pred = np.concatenate([self.y_pred, y_pred.cpu().numpy()])

    def calculate(self):
        return np.corrcoef(self.y_true, self.y_pred)[0, 1]


class NoMaskMetric(BaseMetric):
    def __init__(self, metric: BaseMetric):
        super().__init__("no_mask/" + metric.name)
        self.metric = metric

    def reset(self):
        self.metric.reset()

    def submit(self, y_true, y_pred, episode_data: EpisodeData):
        masked = np.concatenate([
            episode_data.ball_data[:, :, BallData.MASK.value],
            episode_data.player_data[:, :, PlayerData.MASK.value],
            episode_data.boost_data[:, :, BoostData.MASK.value],
        ], axis=1).any(axis=1)
        y_true = y_true[~masked]
        y_pred = y_pred[~masked]
        self.metric.submit(y_true, y_pred, episode_data[~masked])

    def calculate(self):
        return self.metric.calculate()
