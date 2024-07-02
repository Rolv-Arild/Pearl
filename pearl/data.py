import enum
import itertools
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Iterator

import numpy as np
import torch
from rlgym.rocket_league.common_values import BOOST_LOCATIONS, BIG_PAD_RECHARGE_SECONDS

BOOST_LOCATIONS = np.array(BOOST_LOCATIONS)

DEMO_RESPAWN_TIME = 3
NO_TEAM = 2


@enum.unique
class PlayerData(IntEnum):
    IGNORE = 0  # Whether to mask out this player with attention
    MASK = auto()  # Whether the data is masked out (e.g. everything else set to 0)
    AGE = auto()  # The time since this player was last updated
    TEAM = auto()  # -1 for blue, +1 for orange
    POS_X = auto()
    POS_Y = auto()
    POS_Z = auto()
    VEL_X = auto()
    VEL_Y = auto()
    VEL_Z = auto()
    FW_X = auto()
    FW_Y = auto()
    FW_Z = auto()
    UP_X = auto()
    UP_Y = auto()
    UP_Z = auto()
    ANG_VEL_X = auto()
    ANG_VEL_Y = auto()
    ANG_VEL_Z = auto()
    BOOST_AMOUNT = auto()
    IS_DEMOED = auto()
    RESPAWN_TIMER = auto()
    # Now for a bunch of specific internal values, mostly handled by RocketSim
    ON_GROUND = auto()
    SUPERSONIC_TIME = auto()
    BOOST_ACTIVE_TIME = auto()
    HANDBRAKE = auto()
    HAS_JUMPED = auto()
    IS_HOLDING_JUMP = auto()
    IS_JUMPING = auto()
    JUMP_TIME = auto()
    HAS_FLIPPED = auto()
    HAS_DOUBLE_JUMPED = auto()
    AIR_TIME_SINCE_JUMP = auto()
    FLIP_TIME = auto()
    FLIP_TORQUE_X = auto()
    FLIP_TORQUE_Y = auto()
    FLIP_TORQUE_Z = auto()
    IS_AUTOFLIPPING = auto()
    AUTOFLIP_TIMER = auto()
    AUTOFLIP_DIRECTION = auto()


@enum.unique
class BallData(IntEnum):
    IGNORE = 0
    MASK = auto()
    POS_X = auto()
    POS_Y = auto()
    POS_Z = auto()
    VEL_X = auto()
    VEL_Y = auto()
    VEL_Z = auto()
    ANG_VEL_X = auto()
    ANG_VEL_Y = auto()
    ANG_VEL_Z = auto()


@enum.unique
class BoostData(IntEnum):
    IGNORE = 0
    MASK = auto()
    TIMER_PAD_0 = auto()
    TIMER_PAD_1 = auto()
    TIMER_PAD_2 = auto()
    TIMER_PAD_3 = auto()
    TIMER_PAD_4 = auto()
    TIMER_PAD_5 = auto()
    TIMER_PAD_6 = auto()
    TIMER_PAD_7 = auto()
    TIMER_PAD_8 = auto()
    TIMER_PAD_9 = auto()
    TIMER_PAD_10 = auto()
    TIMER_PAD_11 = auto()
    TIMER_PAD_12 = auto()
    TIMER_PAD_13 = auto()
    TIMER_PAD_14 = auto()
    TIMER_PAD_15 = auto()
    TIMER_PAD_16 = auto()
    TIMER_PAD_17 = auto()
    TIMER_PAD_18 = auto()
    TIMER_PAD_19 = auto()
    TIMER_PAD_20 = auto()
    TIMER_PAD_21 = auto()
    TIMER_PAD_22 = auto()
    TIMER_PAD_23 = auto()
    TIMER_PAD_24 = auto()
    TIMER_PAD_25 = auto()
    TIMER_PAD_26 = auto()
    TIMER_PAD_27 = auto()
    TIMER_PAD_28 = auto()
    TIMER_PAD_29 = auto()
    TIMER_PAD_30 = auto()
    TIMER_PAD_31 = auto()
    TIMER_PAD_32 = auto()
    TIMER_PAD_33 = auto()


# We're using a fixed number of boost timers for now, if we wanted to make this more general we could use:
# BoostColumn = Enum(
#     "BoostColumn",
#     """MASK IGNORE
#     POS_X POS_Y POS_Z
#     BOOST_AMOUNT
#     RESPAWN_TIMER"""
# )

@enum.unique
class GameInfo(IntEnum):
    IGNORE = 0
    MASK = auto()
    TIME_REMAINING = auto()
    IS_OVERTIME = auto()
    KICKOFF_TIMER = auto()
    BLUE_SCORE = auto()
    ORANGE_SCORE = auto()


@dataclass(slots=True)
class EpisodeData:
    game_info: np.ndarray  # Info about general stuff about the game (e.g. scoreboard)
    ball_data: np.ndarray  # Physics data for the ball
    player_data: np.ndarray  # Physics data for all the player
    boost_data: np.ndarray  # Boost timers
    next_goal_side: np.ndarray  # Which team scores next in this episode (0 blue, 1 orange, 2 none)
    match_win_side: np.ndarray  # Which team ultimately wins the match that this episode is in
    time_until_end: np.ndarray  # Time until end of episode (e.g. goal scored or ball hitting ground on 0s)
    episode_id: np.ndarray  # Distinct id for each episode
    is_xy_flipped: np.ndarray  # Whether x/y is flipped (shape (n,2)) compared to the original data
    is_normalized: bool = False  # Whether the data is normalized

    @staticmethod
    def new_empty(num_rows: int, num_balls=1, num_players=6, num_boosts=1, normalized=False):
        return EpisodeData(
            game_info=np.zeros((num_rows, len(GameInfo)), dtype=np.float32),
            ball_data=np.zeros((num_rows, num_balls, len(BallData)), dtype=np.float32),
            player_data=np.zeros((num_rows, num_players, len(PlayerData)), dtype=np.float32),
            boost_data=np.zeros((num_rows, num_boosts, len(BoostData)), dtype=np.float32),
            next_goal_side=np.zeros((num_rows,), dtype=np.int8),
            match_win_side=np.zeros((num_rows,), dtype=np.int8),
            time_until_end=np.zeros((num_rows,), dtype=np.float32),
            episode_id=np.zeros((num_rows,), dtype=np.int32),
            is_xy_flipped=np.zeros((num_rows, 2), dtype=bool),
            is_normalized=normalized,
        )

    def __len__(self):
        return len(self.time_until_end)

    def _get_attributes(self):
        # Since lots of different functions perform the same operations on all the attributes
        # I use getattr and setattr frequently, so I don't need to rewrite so much
        return self.__slots__  # noqa

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)

        kwargs = {}
        for attr in self._get_attributes():
            self_val = getattr(self, attr)
            if isinstance(self_val, np.ndarray) and self_val.ndim > 0:
                kwargs[attr] = self_val[idx]  # Note that this will produce a view
            else:
                kwargs[attr] = self_val

        return EpisodeData(**kwargs)

    def __setslice__(self, i, j, sequence: "EpisodeData"):
        self.__setitem__(slice(i, j), sequence)

    def __setitem__(self, idx, value: "EpisodeData"):
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        if value.is_normalized != self.is_normalized:
            raise ValueError("Normalization mismatch")
        for attr in self._get_attributes():
            self_val = getattr(self, attr)
            if isinstance(self_val, np.ndarray) and self_val.ndim > 0:
                other_val = getattr(value, attr)
                self_val[idx] = other_val

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __add__(self, other: "EpisodeData"):
        if self.is_normalized != other.is_normalized:
            raise ValueError("Cannot add normalized and unnormalized data")
        if len(other) == 0:
            return self
        if len(self) == 0:
            return other

        kwargs = {}
        for attr in self._get_attributes():
            self_val = getattr(self, attr)
            if isinstance(self_val, np.ndarray) and self_val.ndim > 0:
                other_val = getattr(other, attr)
                kwargs[attr] = np.concatenate((self_val, other_val))
            else:
                kwargs[attr] = self_val

        return EpisodeData(**kwargs)

    def normalize(self):
        if not self.is_normalized:
            self.game_info[:, GameInfo.TIME_REMAINING] = \
                np.clip(self.game_info[:, GameInfo.TIME_REMAINING], 0, 5 * 60) / (5 * 60)
            self.game_info[:, GameInfo.KICKOFF_TIMER] /= 5.0
            self.game_info[:, GameInfo.BLUE_SCORE] /= 10.0  # Soft upper bound
            self.game_info[:, GameInfo.ORANGE_SCORE] /= 10.0
            # absmax = abs(self.game_info).max()
            # if absmax > 1:
            #     print(f"Non-normalized value in player data? "
            #           f"Found value {absmax} at {np.where(abs(self.game_info) == absmax)}")

            self.ball_data[:, :, BallData.POS_X] /= 4096.0
            self.ball_data[:, :, BallData.POS_Y] /= 6000.0
            self.ball_data[:, :, BallData.POS_Z] /= 2044.0
            self.ball_data[:, :, BallData.VEL_X] /= 6000.0
            self.ball_data[:, :, BallData.VEL_Y] /= 6000.0
            self.ball_data[:, :, BallData.VEL_Z] /= 6000.0
            self.ball_data[:, :, BallData.ANG_VEL_X] /= 6.0
            self.ball_data[:, :, BallData.ANG_VEL_Y] /= 6.0
            self.ball_data[:, :, BallData.ANG_VEL_Z] /= 6.0
            # absmax = abs(self.ball_data).max()
            # if absmax > 1:
            #     print(f"Non-normalized value in player data? "
            #           f"Found value {absmax} at {np.where(abs(self.ball_data) == absmax)}")

            self.player_data[:, :, PlayerData.POS_X] /= 4096.0
            self.player_data[:, :, PlayerData.POS_Y] /= 6000.0
            self.player_data[:, :, PlayerData.POS_Z] /= 2044.0
            self.player_data[:, :, PlayerData.VEL_X] /= 2300.0
            self.player_data[:, :, PlayerData.VEL_Y] /= 2300.0
            self.player_data[:, :, PlayerData.VEL_Z] /= 2300.0
            self.player_data[:, :, PlayerData.ANG_VEL_X] /= 5.5
            self.player_data[:, :, PlayerData.ANG_VEL_Y] /= 5.5
            self.player_data[:, :, PlayerData.ANG_VEL_Z] /= 5.5
            # self.player_data[:, :, PlayerData.BOOST_AMOUNT] /= 1.0  # Boost amount is pre-normalized
            self.player_data[:, :, PlayerData.RESPAWN_TIMER] /= DEMO_RESPAWN_TIME

            self.player_data[:, :, PlayerData.SUPERSONIC_TIME] \
                = np.clip(self.player_data[:, :, PlayerData.SUPERSONIC_TIME], 0, 1.0)
            self.player_data[:, :, PlayerData.BOOST_ACTIVE_TIME] \
                = np.clip(self.player_data[:, :, PlayerData.BOOST_ACTIVE_TIME], 0, 0.1) / 0.1
            self.player_data[:, :, PlayerData.JUMP_TIME] /= 0.2
            self.player_data[:, :, PlayerData.AIR_TIME_SINCE_JUMP] = \
                np.clip(self.player_data[:, :, PlayerData.AIR_TIME_SINCE_JUMP], 0, 1.25) / 1.25
            self.player_data[:, :, PlayerData.FLIP_TIME] \
                = np.clip(self.player_data[:, :, PlayerData.FLIP_TIME], 0, 0.95) / 0.95
            self.player_data[:, :, PlayerData.AUTOFLIP_TIMER] \
                = np.clip(self.player_data[:, :, PlayerData.AUTOFLIP_TIMER], 0, 0.4) / 0.4
            # absmax = abs(self.player_data).max()
            # if absmax > 1:
            #     print(f"Non-normalized value in player data? "
            #           f"Found value {absmax} at {np.where(abs(self.player_data) == absmax)}")

            self.boost_data[:, :, BoostData.TIMER_PAD_0:BoostData.TIMER_PAD_33 + 1] /= BIG_PAD_RECHARGE_SECONDS
            # absmax = abs(self.boost_data).max()
            # if absmax > 1:
            #     print(f"Non-normalized value in player data? "
            #           f"Found value {absmax} at {np.where(abs(self.boost_data) == absmax)}")
            self.is_normalized = True

    def swap_teams(self, idx=None, rng=None):
        if idx is None:
            idx = slice(None)
        elif isinstance(idx, str) and idx == "random":
            if rng is None:
                rng = np.random
            idx = rng.random(len(self)) < 0.5
        blue_score = self.game_info[idx, GameInfo.BLUE_SCORE].copy()
        self.game_info[idx, GameInfo.BLUE_SCORE] = self.game_info[idx, GameInfo.ORANGE_SCORE]
        self.game_info[idx, GameInfo.ORANGE_SCORE] = blue_score

        self.ball_data[idx, :, BallData.POS_X] *= -1
        self.ball_data[idx, :, BallData.POS_Y] *= -1
        self.ball_data[idx, :, BallData.VEL_X] *= -1
        self.ball_data[idx, :, BallData.VEL_Y] *= -1
        self.ball_data[idx, :, BallData.ANG_VEL_X] *= -1
        self.ball_data[idx, :, BallData.ANG_VEL_Y] *= -1

        self.player_data[idx, :, PlayerData.TEAM] *= -1
        self.player_data[idx, :, PlayerData.POS_X] *= -1
        self.player_data[idx, :, PlayerData.POS_Y] *= -1
        self.player_data[idx, :, PlayerData.VEL_X] *= -1
        self.player_data[idx, :, PlayerData.VEL_Y] *= -1
        self.player_data[idx, :, PlayerData.ANG_VEL_X] *= -1
        self.player_data[idx, :, PlayerData.ANG_VEL_Y] *= -1
        self.player_data[idx, :, PlayerData.FW_X] *= -1
        self.player_data[idx, :, PlayerData.FW_Y] *= -1
        self.player_data[idx, :, PlayerData.UP_X] *= -1
        self.player_data[idx, :, PlayerData.UP_Y] *= -1

        self.boost_data[idx, :, 2:] = self.boost_data[idx, :, :1:-1]

        self.next_goal_side[idx] = (1 - self.next_goal_side[idx]) % 3  # 0->1, 1->0, 2->2
        self.is_xy_flipped[idx] = 1 - self.is_xy_flipped[idx]

    def mirror_x(self, idx=None, rng=None):
        if idx is None:
            idx = slice(None)
        elif isinstance(idx, str) and idx == "random":
            if rng is None:
                rng = np.random
            idx = rng.random(len(self)) < 0.5
        self.ball_data[idx, :, BallData.POS_X] *= -1
        self.ball_data[idx, :, BallData.VEL_X] *= -1
        self.ball_data[idx, :, BallData.ANG_VEL_Y] *= -1
        self.ball_data[idx, :, BallData.ANG_VEL_Z] *= -1

        self.player_data[idx, :, PlayerData.POS_X] *= -1
        self.player_data[idx, :, PlayerData.VEL_X] *= -1
        self.player_data[idx, :, PlayerData.ANG_VEL_Z] *= -1
        self.player_data[idx, :, PlayerData.ANG_VEL_Y] *= -1
        self.player_data[idx, :, PlayerData.FW_X] *= -1
        self.player_data[idx, :, PlayerData.UP_X] *= -1
        self.player_data[idx, :, PlayerData.FLIP_TORQUE_Y] *= -1  # Car-relative so y is left/right

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

        self.is_xy_flipped[idx, 0] = 1 - self.is_xy_flipped[idx, 0]

    def mirror_y(self, idx=None, rng=None):
        self.swap_teams(idx, rng)
        self.mirror_x(idx, rng)

    def normalize_ball_quadrant(self):
        # Normalize ball position to be in the positive x/y quadrant
        # Swap teams first, since it will also change the x sign
        # For balls on center lines, randomly swap teams
        assert self.ball_data.shape[1] == 1, "Can only normalize ball quadrant when there is one ball"
        neg_y = self.ball_data[:, 0, BallData.POS_Y] < 0
        zero_y = self.ball_data[:, 0, BallData.POS_Y] == 0
        rand = np.random.random(len(self)) < 0.5
        idx = neg_y | (zero_y & rand)
        self.swap_teams(idx)

        neg_x = self.ball_data[:, 0, BallData.POS_X] < 0
        zero_x = self.ball_data[:, 0, BallData.POS_X] == 0
        rand = np.random.random(len(self)) < 0.5
        idx = neg_x | (zero_x & rand)
        self.mirror_x(idx)

    def shuffle(self, rng=None):
        if rng is None:
            rng = np.random
        idx = rng.permutation(len(self))

        for attr in self._get_attributes():
            val = getattr(self, attr)
            if isinstance(val, np.ndarray) and val.ndim > 0:
                setattr(self, attr, val[idx])

    def mask_all_rows(self, game_info_mask: bool, ball_mask: list, player_mask: list, boost_mask: list,
                      use_ignore=False):
        if game_info_mask:
            self.game_info[:] = 0
            if use_ignore:
                self.game_info[:, GameInfo.IGNORE] = 1
            else:
                self.game_info[:, GameInfo.MASK] = 1
        # Each mask is a list of entities to mask
        for i in ball_mask:
            self.ball_data[:, i, :] = 0
            if use_ignore:
                self.ball_data[:, i, BallData.IGNORE] = 1
            else:
                self.ball_data[:, i, BallData.MASK] = 1
        for i in player_mask:
            self.player_data[:, i, :] = 0
            if use_ignore:
                self.player_data[:, i, PlayerData.IGNORE] = 1
            else:
                self.player_data[:, i, PlayerData.MASK] = 1
        for i in boost_mask:
            self.boost_data[:, i, :] = 0
            if use_ignore:
                self.boost_data[:, i, BoostData.IGNORE] = 1
            else:
                self.boost_data[:, i, BoostData.MASK] = 1

    def mask_combinations(self, mask_game_info=False, mask_ball=False, mask_players=True, mask_boost=False,
                          use_ignore=False) -> Iterator["EpisodeData"]:
        entities = (1 * mask_game_info
                    + self.ball_data.shape[1] * mask_ball
                    + self.player_data.shape[1] * mask_players
                    + self.boost_data.shape[1] * mask_boost)
        for comb in itertools.product([False, True], repeat=entities):
            game_info_mask = False
            ball_mask = []
            player_mask = []
            boost_mask = []
            masks = []
            i = 0
            if mask_game_info:
                if comb[i]:
                    game_info_mask = True
                i += 1
            if mask_ball:
                for j in range(self.ball_data.shape[1]):
                    if comb[i]:
                        ball_mask.append(j)
                    i += 1
                masks.append(ball_mask)
            if mask_players:
                for j in range(self.player_data.shape[1]):
                    if comb[i]:
                        player_mask.append(j)
                    i += 1
                masks.append(player_mask)
            if mask_boost:
                for j in range(self.boost_data.shape[1]):
                    if comb[i]:
                        boost_mask.append(j)
                    i += 1
                masks.append(boost_mask)
            ep = self.clone()
            ep.mask_all_rows(game_info_mask, ball_mask, player_mask, boost_mask, use_ignore=use_ignore)
            yield masks, ep

    def mask_randomly(self, mode="uniform", remove_team_info=True, rng=None):
        if rng is None:
            rng = np.random
        entities = self.ball_data.shape[1] + self.player_data.shape[1] + self.boost_data.shape[1]
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
                self.ball_data[m, i, BallData.MASK] = 1
            elif i < self.ball_data.shape[1] + self.player_data.shape[1]:
                i -= self.ball_data.shape[1]
                if remove_team_info:
                    self.player_data[m, i, :] = 0
                else:
                    # Remove everything except team info
                    self.player_data[m, i, :PlayerData.TEAM] = 0
                    self.player_data[m, i, PlayerData.MASK:] = 0
                self.player_data[m, i, PlayerData.MASK] = 1
            else:
                self.boost_data[m, :] = 0
                self.boost_data[m, :, BoostData.MASK] = 1

    def to_torch(self, device=None):
        x = (
            torch.from_numpy(self.game_info).float().to(device),
            torch.from_numpy(self.ball_data).float().to(device),
            torch.from_numpy(self.player_data).float().to(device),
            torch.from_numpy(self.boost_data).float().to(device),
        )
        y = (
            torch.from_numpy(self.next_goal_side).long().to(device),
            torch.from_numpy(self.match_win_side).long().to(device),
        )
        return x, y

    def save(self, path):
        kwargs = {
            attr: getattr(self, attr)
            for attr in self._get_attributes()
        }
        np.savez_compressed(path, **kwargs)

    @staticmethod
    def load(path):
        data = np.load(path)
        return EpisodeData(**data)

    def clone(self):
        kwargs = {}
        for attr in self._get_attributes():
            self_val = getattr(self, attr)
            if isinstance(self_val, np.ndarray):
                kwargs[attr] = self_val.copy()
            elif isinstance(self_val, (str, int, bool, float, tuple)):
                kwargs[attr] = self_val
            else:
                raise ValueError(f"Not sure how to copy type {type(self_val)}")

        return EpisodeData(**kwargs)
