from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Iterator, NamedTuple

import numpy as np
import pandas as pd
import torch

from pearl.replay import ParsedReplay

BOOST_LOCATIONS = np.array([
    (0.0, -4240.0, 70.0),
    (-1792.0, -4184.0, 70.0),
    (1792.0, -4184.0, 70.0),
    (-3072.0, -4096.0, 73.0),
    (3072.0, -4096.0, 73.0),
    (- 940.0, -3308.0, 70.0),
    (940.0, -3308.0, 70.0),
    (0.0, -2816.0, 70.0),
    (-3584.0, -2484.0, 70.0),
    (3584.0, -2484.0, 70.0),
    (-1788.0, -2300.0, 70.0),
    (1788.0, -2300.0, 70.0),
    (-2048.0, -1036.0, 70.0),
    (0.0, -1024.0, 70.0),
    (2048.0, -1036.0, 70.0),
    (-3584.0, 0.0, 73.0),
    (-1024.0, 0.0, 70.0),
    (1024.0, 0.0, 70.0),
    (3584.0, 0.0, 73.0),
    (-2048.0, 1036.0, 70.0),
    (0.0, 1024.0, 70.0),
    (2048.0, 1036.0, 70.0),
    (-1788.0, 2300.0, 70.0),
    (1788.0, 2300.0, 70.0),
    (-3584.0, 2484.0, 70.0),
    (3584.0, 2484.0, 70.0),
    (0.0, 2816.0, 70.0),
    (- 940.0, 3308.0, 70.0),
    (940.0, 3308.0, 70.0),
    (-3072.0, 4096.0, 73.0),
    (3072.0, 4096.0, 73.0),
    (-1792.0, 4184.0, 70.0),
    (1792.0, 4184.0, 70.0),
    (0.0, 4240.0, 70.0),
])

DEMO_RESPAWN_TIME = 3
SMALL_BOOST_RESPAWN_TIME = 4
BIG_BOOST_RESPAWN_TIME = 10


class PlayerData(Enum):
    IGNORE = 0
    MASK = 1
    TEAM = 2
    POS_X = 3
    POS_Y = 4
    POS_Z = 5
    VEL_X = 6
    VEL_Y = 7
    VEL_Z = 8
    FW_X = 9
    FW_Y = 10
    FW_Z = 11
    UP_X = 12
    UP_Y = 13
    UP_Z = 14
    ANG_VEL_X = 15
    ANG_VEL_Y = 16
    ANG_VEL_Z = 17
    BOOST_AMOUNT = 18
    IS_DEMOED = 19
    RESPAWN_TIMER = 20
    # TODO consider adding jump/dodge/handbrake info


class BallData(Enum):
    IGNORE = 0
    MASK = 1
    POS_X = 2
    POS_Y = 3
    POS_Z = 4
    VEL_X = 5
    VEL_Y = 6
    VEL_Z = 7
    ANG_VEL_X = 8
    ANG_VEL_Y = 9
    ANG_VEL_Z = 10


class BoostData(Enum):
    IGNORE = 0
    MASK = 1
    TIMER_PAD_0 = 2
    TIMER_PAD_1 = 3
    TIMER_PAD_2 = 4
    TIMER_PAD_3 = 5
    TIMER_PAD_4 = 6
    TIMER_PAD_5 = 7
    TIMER_PAD_6 = 8
    TIMER_PAD_7 = 9
    TIMER_PAD_8 = 10
    TIMER_PAD_9 = 11
    TIMER_PAD_10 = 12
    TIMER_PAD_11 = 13
    TIMER_PAD_12 = 14
    TIMER_PAD_13 = 15
    TIMER_PAD_14 = 16
    TIMER_PAD_15 = 17
    TIMER_PAD_16 = 18
    TIMER_PAD_17 = 19
    TIMER_PAD_18 = 20
    TIMER_PAD_19 = 21
    TIMER_PAD_20 = 22
    TIMER_PAD_21 = 23
    TIMER_PAD_22 = 24
    TIMER_PAD_23 = 25
    TIMER_PAD_24 = 26
    TIMER_PAD_25 = 27
    TIMER_PAD_26 = 28
    TIMER_PAD_27 = 29
    TIMER_PAD_28 = 30
    TIMER_PAD_29 = 31
    TIMER_PAD_30 = 32
    TIMER_PAD_31 = 33
    TIMER_PAD_32 = 34
    TIMER_PAD_33 = 35
    TIMER_PAD_34 = 36


# We're using a fixed number of boost timers for now, if we wanted to make this more general we could use:
# BoostColumn = Enum(
#     "BoostColumn",
#     """MASK IGNORE
#     POS_X POS_Y POS_Z
#     BOOST_AMOUNT
#     RESPAWN_TIMER"""
# )


# TODO consider adding columns for game info (e.g. time, score)


def quat_to_rot_mtx(quat: np.ndarray) -> np.ndarray:
    w = -quat[0]
    x = -quat[1]
    y = -quat[2]
    z = -quat[3]

    theta = np.zeros((3, 3))

    norm = np.dot(quat, quat)
    if norm != 0:
        s = 1.0 / norm

        # front direction
        theta[0, 0] = 1.0 - 2.0 * s * (y * y + z * z)
        theta[1, 0] = 2.0 * s * (x * y + z * w)
        theta[2, 0] = 2.0 * s * (x * z - y * w)

        # left direction
        theta[0, 1] = 2.0 * s * (x * y - z * w)
        theta[1, 1] = 1.0 - 2.0 * s * (x * x + z * z)
        theta[2, 1] = 2.0 * s * (y * z + x * w)

        # up direction
        theta[0, 2] = 2.0 * s * (x * z + y * w)
        theta[1, 2] = 2.0 * s * (y * z - x * w)
        theta[2, 2] = 1.0 - 2.0 * s * (x * x + y * y)

    return theta


quat_to_rot_mtx = np.vectorize(quat_to_rot_mtx, signature="(4)->(3,3)")


@dataclass
class EpisodeData:
    # game_info: np.ndarray
    ball_data: np.ndarray
    player_data: np.ndarray
    boost_data: np.ndarray
    next_goal_side: np.ndarray
    time_until_end: np.ndarray
    episode_id: np.ndarray
    is_normalized: bool = False

    @staticmethod
    def new_empty(num_rows: int, num_balls=1, num_players=6, num_boosts=1, normalized=False):
        return EpisodeData(
            # game_info=np.zeros((num_rows, len(GameInfo))),
            ball_data=np.zeros((num_rows, num_balls, len(BallData))),
            player_data=np.zeros((num_rows, num_players, len(PlayerData))),
            boost_data=np.zeros((num_rows, num_boosts, len(BoostData))),
            next_goal_side=np.zeros((num_rows,)),
            time_until_end=np.zeros((num_rows,)),
            episode_id=np.zeros((num_rows,)),
            is_normalized=normalized
        )

    def __len__(self):
        return len(self.time_until_end)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        return EpisodeData(
            # game_info=self.game_info[idx],
            ball_data=self.ball_data[idx],
            player_data=self.player_data[idx],
            boost_data=self.boost_data[idx],
            next_goal_side=self.next_goal_side[idx],
            time_until_end=self.time_until_end[idx],
            episode_id=self.episode_id[idx],
            is_normalized=self.is_normalized
        )

    def __setslice__(self, i, j, sequence):
        if sequence.is_normalized != self.is_normalized:
            raise ValueError("Normalization mismatch")
        self.ball_data[i:j] = sequence.ball_data
        self.player_data[i:j] = sequence.player_data
        self.boost_data[i:j] = sequence.boost_data
        self.next_goal_side[i:j] = sequence.next_goal_side
        self.time_until_end[i:j] = sequence.time_until_end
        self.episode_id[i:j] = sequence.episode_id

    def __setitem__(self, idx, value):
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        if value.is_normalized != self.is_normalized:
            raise ValueError("Normalization mismatch")
        self.ball_data[idx] = value.ball_data
        self.player_data[idx] = value.player_data
        self.boost_data[idx] = value.boost_data
        self.next_goal_side[idx] = value.next_goal_side
        self.time_until_end[idx] = value.time_until_end
        self.episode_id[idx] = value.episode_id

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __add__(self, other):
        if self.is_normalized and other.is_normalized:
            is_normalized = True
        elif self.is_normalized or other.is_normalized:
            raise ValueError("Cannot add normalized and unnormalized data")
        else:
            is_normalized = False
        return EpisodeData(
            # game_info=np.concatenate((self.game_info, other.game_info)),
            ball_data=np.concatenate((self.ball_data, other.ball_data)),
            player_data=np.concatenate((self.player_data, other.player_data)),
            boost_data=np.concatenate((self.boost_data, other.boost_data)),
            next_goal_side=np.concatenate((self.next_goal_side, other.next_goal_side)),
            time_until_end=np.concatenate((self.time_until_end, other.time_until_end)),
            episode_id=np.concatenate((self.episode_id, other.episode_id + self.episode_id[-1] + 1)),
            is_normalized=is_normalized
        )

    def normalize(self):
        if not self.is_normalized:
            self.ball_data[:, :, BallData.POS_X.value] /= 4096.0
            self.ball_data[:, :, BallData.POS_Y.value] /= 5120.0
            self.ball_data[:, :, BallData.POS_Z.value] /= 2044.0
            self.ball_data[:, :, BallData.VEL_X.value] /= 6000.0
            self.ball_data[:, :, BallData.VEL_Y.value] /= 6000.0
            self.ball_data[:, :, BallData.VEL_Z.value] /= 6000.0
            self.ball_data[:, :, BallData.ANG_VEL_X.value] /= 6.0
            self.ball_data[:, :, BallData.ANG_VEL_Y.value] /= 6.0
            self.ball_data[:, :, BallData.ANG_VEL_Z.value] /= 6.0

            self.player_data[:, :, PlayerData.POS_X.value] /= 4096.0
            self.player_data[:, :, PlayerData.POS_Y.value] /= 5120.0
            self.player_data[:, :, PlayerData.POS_Z.value] /= 2044.0
            self.player_data[:, :, PlayerData.VEL_X.value] /= 2300.0
            self.player_data[:, :, PlayerData.VEL_Y.value] /= 2300.0
            self.player_data[:, :, PlayerData.VEL_Z.value] /= 2300.0
            self.player_data[:, :, PlayerData.ANG_VEL_X.value] /= 5.5
            self.player_data[:, :, PlayerData.ANG_VEL_Y.value] /= 5.5
            self.player_data[:, :, PlayerData.ANG_VEL_Z.value] /= 5.5
            # Forward and up are pre-normalized
            self.player_data[:, :, PlayerData.BOOST_AMOUNT.value] /= 100.0
            self.player_data[:, :, PlayerData.RESPAWN_TIMER.value] /= DEMO_RESPAWN_TIME

            self.boost_data[:, BoostData.TIMER_PAD_0.value:BoostData.TIMER_PAD_34.value + 1] /= BIG_BOOST_RESPAWN_TIME
            self.is_normalized = True

    def swap_teams(self, idx=None, rng=None):
        if idx is None:
            idx = slice(None)
        elif idx == "random":
            if rng is None:
                rng = np.random
            idx = rng.random(len(self)) < 0.5
        self.ball_data[idx, :, BallData.POS_X.value] *= -1
        self.ball_data[idx, :, BallData.POS_Y.value] *= -1
        self.ball_data[idx, :, BallData.VEL_X.value] *= -1
        self.ball_data[idx, :, BallData.VEL_Y.value] *= -1
        self.ball_data[idx, :, BallData.ANG_VEL_X.value] *= -1
        self.ball_data[idx, :, BallData.ANG_VEL_Y.value] *= -1

        self.player_data[idx, :, PlayerData.TEAM.value] *= -1
        self.player_data[idx, :, PlayerData.POS_X.value] *= -1
        self.player_data[idx, :, PlayerData.POS_Y.value] *= -1
        self.player_data[idx, :, PlayerData.VEL_X.value] *= -1
        self.player_data[idx, :, PlayerData.VEL_Y.value] *= -1
        self.player_data[idx, :, PlayerData.ANG_VEL_X.value] *= -1
        self.player_data[idx, :, PlayerData.ANG_VEL_Y.value] *= -1
        self.player_data[idx, :, PlayerData.FW_X.value] *= -1
        self.player_data[idx, :, PlayerData.FW_Y.value] *= -1
        self.player_data[idx, :, PlayerData.UP_X.value] *= -1
        self.player_data[idx, :, PlayerData.UP_Y.value] *= -1

        self.boost_data[idx, :, 2:] = self.boost_data[idx, :, :1:-1]

        self.next_goal_side[idx] *= -1

    def mirror_x(self, idx=None, rng=None):
        if idx is None:
            idx = slice(None)
        elif idx == "random":
            if rng is None:
                rng = np.random
            idx = rng.random(len(self)) < 0.5
        self.ball_data[idx, :, BallData.POS_X.value] *= -1
        self.ball_data[idx, :, BallData.VEL_X.value] *= -1
        self.ball_data[idx, :, BallData.ANG_VEL_X.value] *= -1

        self.player_data[idx, :, PlayerData.POS_X.value] *= -1
        self.player_data[idx, :, PlayerData.VEL_X.value] *= -1
        self.player_data[idx, :, PlayerData.ANG_VEL_X.value] *= -1
        self.player_data[idx, :, PlayerData.FW_X.value] *= -1
        self.player_data[idx, :, PlayerData.UP_X.value] *= -1

        for i, loc in enumerate(BOOST_LOCATIONS):
            inv_loc = loc * np.array([-1, 1, 1])
            ii = np.argmin(np.linalg.norm(BOOST_LOCATIONS - inv_loc, axis=1))
            if i >= ii:
                continue  # So we don't swap twice or with itself
            # Skip ignore and mask columns
            i += 2
            ii += 2
            # Swap the boost pads
            col1 = self.boost_data[idx, :, i].copy()
            self.boost_data[idx, :, i] = self.boost_data[idx, :, ii]
            self.boost_data[idx, :, ii] = col1

    def shuffle(self, rng=None):
        if rng is None:
            rng = np.random
        idx = rng.permutation(len(self))
        self.ball_data = self.ball_data[idx]
        self.player_data = self.player_data[idx]
        self.boost_data = self.boost_data[idx]
        self.next_goal_side = self.next_goal_side[idx]
        self.time_until_end = self.time_until_end[idx]
        self.episode_id = self.episode_id[idx]

    def mask_randomly(self, mode="uniform", remove_team_info=True, rng=None):
        if rng is None:
            rng = np.random
        entities = self.ball_data.shape[1] + self.player_data.shape[1] + 1
        if mode == "uniform":
            mask = rng.random(size=(len(self), entities)) < rng.random(size=(len(self), 1))
        elif mode == "binomial":
            mask = rng.random(size=(len(self), entities)) < 0.5
        elif mode == "triangular":
            mask = rng.random(size=(len(self), entities)) < rng.triangular(0, 0, 1, size=(len(self), 1))
        else:
            raise ValueError(f"Unknown mode {mode}")
        for i in range(entities):
            m = mask[:, i]
            if i < self.ball_data.shape[1]:
                self.ball_data[m, i, :] = 0
                self.ball_data[m, i, BallData.MASK.value] = 1
            elif i < self.ball_data.shape[1] + self.player_data.shape[1]:
                i -= self.ball_data.shape[1]
                if remove_team_info:
                    self.player_data[m, i, :] = 0
                else:
                    # Remove everything except team info
                    self.player_data[m, i, :PlayerData.TEAM.value] = 0
                    self.player_data[m, i, PlayerData.MASK.value + 1:] = 0
                self.player_data[m, i, PlayerData.MASK.value] = 1
            else:
                self.boost_data[m, :] = 0
                self.boost_data[m, :, BoostData.MASK.value] = 1

    def to_torch(self, device=None):
        x = (
            torch.from_numpy(self.ball_data).float().to(device),
            torch.from_numpy(self.player_data).float().to(device),
            torch.from_numpy(self.boost_data).float().to(device),
        )
        y = torch.from_numpy((self.next_goal_side + 1) / 2).float().to(device)
        return x, y

    def save(self, path):
        np.savez_compressed(path, ball_data=self.ball_data, player_data=self.player_data, boost_data=self.boost_data,
                            next_goal_side=self.next_goal_side, time_until_end=self.time_until_end,
                            episode_id=self.episode_id, is_normalized=self.is_normalized)

    @staticmethod
    def load(path):
        data = np.load(path)
        return EpisodeData(data["ball_data"], data["player_data"], data["boost_data"],
                           data["next_goal_side"], data["time_until_end"],
                           data["episode_id"], data["is_normalized"])


def replay_to_data(replay: ParsedReplay, normalize: bool = True, ignore_unfinished: bool = True) \
        -> Iterator[EpisodeData]:
    """
    Convert a replay to a sequence of training data.

    :param replay: the replay to convert.
    :param normalize: whether to normalize the data.
    :param ignore_unfinished: whether to ignore gameplay periods with no goal scored.
    :return: a sequence of training data.
             Each element is a tuple of (x, y) where x is a tuple of
             (balls, players, boosts) and y is a tuple of (next_goal_side, time_until_end).
    """
    players = [p for p in replay.metadata["players"]
               if p["unique_id"] in replay.player_dfs]

    times = replay.game_df["time"]

    gameplay_periods = replay.analyzer["gameplay_periods"]
    for gameplay_period in gameplay_periods:
        start_frame = gameplay_period["start_frame"]
        goal_frame = gameplay_period["goal_frame"]

        if goal_frame is None:
            end_frame = gameplay_period["end_frame"]
            if ignore_unfinished:
                continue
        else:
            end_frame = goal_frame

        size = end_frame - start_frame + 1  # loc is inclusive
        ep = EpisodeData.new_empty(size)

        # Ball data
        df = replay.ball_df.loc[start_frame:end_frame, :].fillna(0.)
        ep.ball_data[:, 0, BallData.POS_X.value] = df["pos_x"].values
        ep.ball_data[:, 0, BallData.POS_Y.value] = df["pos_y"].values
        ep.ball_data[:, 0, BallData.POS_Z.value] = df["pos_z"].values
        ep.ball_data[:, 0, BallData.VEL_X.value] = df["vel_x"].values
        ep.ball_data[:, 0, BallData.VEL_Y.value] = df["vel_y"].values
        ep.ball_data[:, 0, BallData.VEL_Z.value] = df["vel_z"].values
        ep.ball_data[:, 0, BallData.ANG_VEL_X.value] = df["ang_vel_x"].values
        ep.ball_data[:, 0, BallData.ANG_VEL_Y.value] = df["ang_vel_y"].values
        ep.ball_data[:, 0, BallData.ANG_VEL_Z.value] = df["ang_vel_z"].values

        # Player data
        ep.player_data[:, :, PlayerData.IGNORE.value] = 1
        uid_to_idx = {}
        for i, player in enumerate(players):
            uid = player["unique_id"]
            uid_to_idx[uid] = i

            df = replay.player_dfs[uid].loc[start_frame:end_frame, :].copy()
            for c in df.columns:
                if "jump" in c or "dodge" in c or "flip" in c:
                    df[c] = df[c].fillna(0.)
                elif c.startswith("match_"):  # Goals, assists, saves
                    df[c] = df[c].fillna(0)
                elif c in ("boost_pickup", "boost_amount"):
                    df[c] = df[c].fillna(0)
                elif c in ("handbrake",):
                    df[c] = df[c].astype(bool)  # Replace None values with False
            df = df.astype(float).ffill().fillna(0.)  # Demos create nan values, so we fill them with last known value

            ep.player_data[:, :, PlayerData.IGNORE.value] = 0
            ep.player_data[:, i, PlayerData.TEAM.value] = 2 * player["is_orange"] - 1  # -1 for blue, 1 for orange
            ep.player_data[:, i, PlayerData.POS_X.value] = df["pos_x"].values
            ep.player_data[:, i, PlayerData.POS_Y.value] = df["pos_y"].values
            ep.player_data[:, i, PlayerData.POS_Z.value] = df["pos_z"].values
            ep.player_data[:, i, PlayerData.VEL_X.value] = df["vel_x"].values
            ep.player_data[:, i, PlayerData.VEL_Y.value] = df["vel_y"].values
            ep.player_data[:, i, PlayerData.VEL_Z.value] = df["vel_z"].values
            quat = df[["quat_w", "quat_x", "quat_y", "quat_z"]].values
            rot_mtx = quat_to_rot_mtx(quat)
            ep.player_data[:, i, PlayerData.FW_X.value] = rot_mtx[:, 0, 0]
            ep.player_data[:, i, PlayerData.FW_Y.value] = rot_mtx[:, 0, 1]
            ep.player_data[:, i, PlayerData.FW_Z.value] = rot_mtx[:, 0, 2]
            ep.player_data[:, i, PlayerData.UP_X.value] = rot_mtx[:, 2, 0]
            ep.player_data[:, i, PlayerData.UP_Y.value] = rot_mtx[:, 2, 1]
            ep.player_data[:, i, PlayerData.UP_Z.value] = rot_mtx[:, 2, 2]
            ep.player_data[:, i, PlayerData.ANG_VEL_X.value] = df["ang_vel_x"].values
            ep.player_data[:, i, PlayerData.ANG_VEL_Y.value] = df["ang_vel_y"].values
            ep.player_data[:, i, PlayerData.ANG_VEL_Z.value] = df["ang_vel_z"].values
            ep.player_data[:, i, PlayerData.BOOST_AMOUNT.value] = df["boost_amount"].values

            # Boost data
            boost_pickups = df["boost_pickup"]
            for pickup_frame in boost_pickups[boost_pickups > 0].index:
                pos = df.loc[pickup_frame, ["pos_x", "pos_y", "pos_z"]].values
                closest_boost = np.linalg.norm((BOOST_LOCATIONS - pos).astype(float), axis=1).argmin()
                pickup_time = times.loc[pickup_frame]
                if BOOST_LOCATIONS[closest_boost][2] > 71.5:
                    respawn_time = pickup_time + BIG_BOOST_RESPAWN_TIME
                else:
                    respawn_time = pickup_time + SMALL_BOOST_RESPAWN_TIME

                respawn_frame = next(iter(times[times > respawn_time].index), float("inf"))
                respawn_frame = min(respawn_frame - 1, end_frame)

                start_idx = pickup_frame - start_frame
                end_idx = respawn_frame - start_frame + 1
                assert start_idx < end_idx

                timer = respawn_time - times.loc[pickup_frame:respawn_frame]
                assert (timer >= 0).all()
                ep.boost_data[start_idx:end_idx, 0, BoostData.TIMER_PAD_0.value + closest_boost] = timer

        # Demos
        for demo in replay.metadata["demos"]:
            demo_frame = demo["frame_number"]
            if demo_frame < start_frame or demo_frame > end_frame:
                continue  # Not in this gameplay segment
            victim_uid = demo["victim_unique_id"]
            idx = uid_to_idx[victim_uid]
            demo_time = times.loc[demo_frame]
            respawn_time = demo_time + DEMO_RESPAWN_TIME

            respawn_frame = next(iter(times[times > respawn_time].index), float("inf"))
            respawn_frame = min(respawn_frame - 1, end_frame)

            start_idx = demo_frame - start_frame
            end_idx = respawn_frame - start_frame + 1
            assert start_idx < end_idx

            timer = respawn_time - times.loc[demo_frame:respawn_frame].values
            assert (timer >= 0).all()
            ep.player_data[start_idx:end_idx, idx, PlayerData.IS_DEMOED.value] = 1
            ep.player_data[start_idx:end_idx, idx, PlayerData.RESPAWN_TIMER.value] = timer

        # Goal
        ball_y = replay.ball_df.loc[end_frame, "pos_y"]
        ep.next_goal_side[:] = np.sign(ball_y) * (goal_frame is not None)
        assert np.all(ep.next_goal_side != 0)
        ep.time_until_end[:] = times.loc[end_frame] - times.loc[start_frame:end_frame]

        if normalize:
            ep.normalize()

        yield ep


if __name__ == '__main__':
    replay = ParsedReplay.load("./test_replays/00029e4d-242d-49ed-971d-1218daa2eefa")
    for ep in replay_to_data(replay):
        debug = True
