from typing import Iterator

import numpy as np

from pearl.data import EpisodeData, BallData, PlayerData, BOOST_LOCATIONS, BoostData, DEMO_RESPAWN_TIME
from pearl.replay import ParsedReplay

BIG_BOOST_RESPAWN_TIME = 10
SMALL_BOOST_RESPAWN_TIME = 4


# TODO remove file


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
        ep.ball_data[:, 0, BallData.POS_X] = df["pos_x"].values
        ep.ball_data[:, 0, BallData.POS_Y] = df["pos_y"].values
        ep.ball_data[:, 0, BallData.POS_Z] = df["pos_z"].values
        ep.ball_data[:, 0, BallData.VEL_X] = df["vel_x"].values
        ep.ball_data[:, 0, BallData.VEL_Y] = df["vel_y"].values
        ep.ball_data[:, 0, BallData.VEL_Z] = df["vel_z"].values
        ep.ball_data[:, 0, BallData.ANG_VEL_X] = df["ang_vel_x"].values
        ep.ball_data[:, 0, BallData.ANG_VEL_Y] = df["ang_vel_y"].values
        ep.ball_data[:, 0, BallData.ANG_VEL_Z] = df["ang_vel_z"].values

        # Player data
        ep.player_data[:, :, PlayerData.IGNORE] = 1
        uid_to_idx = {}
        for i, player in enumerate(players):
            uid = player["unique_id"]
            uid_to_idx[uid] = i

            df = replay.player_dfs[uid].loc[start_frame:end_frame, :].copy()
            updated = ((df[["pos_x", "pos_y", "pos_z"]].diff() != 0).any(axis=1)
                       | (abs(df[["vel_x", "vel_y", "vel_z"]]) < 1).all(axis=1))
            df["time"] = times.loc[df.index]
            df.loc[~updated, :] = np.nan  # Set all columns to nan if not updated
            df.loc[np.random.random(df.shape[0]) < 0.01, :] = np.nan  # Randomly set 1% of rows to nan

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

            # ep.player_data[:, :, PlayerData.IGNORE] = 0
            # ep.player_data[:, :, PlayerData.MASK] = 0
            ep.player_data[:, i, PlayerData.AGE] = times.loc[start_frame:end_frame] - df["time"]
            ep.player_data[:, i, PlayerData.TEAM] = 2 * player["is_orange"] - 1  # -1 for blue, 1 for orange
            ep.player_data[:, i, PlayerData.POS_X] = df["pos_x"].values
            ep.player_data[:, i, PlayerData.POS_Y] = df["pos_y"].values
            ep.player_data[:, i, PlayerData.POS_Z] = df["pos_z"].values
            ep.player_data[:, i, PlayerData.VEL_X] = df["vel_x"].values
            ep.player_data[:, i, PlayerData.VEL_Y] = df["vel_y"].values
            ep.player_data[:, i, PlayerData.VEL_Z] = df["vel_z"].values
            quat = df[["quat_w", "quat_x", "quat_y", "quat_z"]].values
            rot_mtx = quat_to_rot_mtx(quat)
            ep.player_data[:, i, PlayerData.FW_X] = rot_mtx[:, 0, 0]
            ep.player_data[:, i, PlayerData.FW_Y] = rot_mtx[:, 0, 1]
            ep.player_data[:, i, PlayerData.FW_Z] = rot_mtx[:, 0, 2]
            ep.player_data[:, i, PlayerData.UP_X] = rot_mtx[:, 2, 0]
            ep.player_data[:, i, PlayerData.UP_Y] = rot_mtx[:, 2, 1]
            ep.player_data[:, i, PlayerData.UP_Z] = rot_mtx[:, 2, 2]
            ep.player_data[:, i, PlayerData.ANG_VEL_X] = df["ang_vel_x"].values
            ep.player_data[:, i, PlayerData.ANG_VEL_Y] = df["ang_vel_y"].values
            ep.player_data[:, i, PlayerData.ANG_VEL_Z] = df["ang_vel_z"].values
            ep.player_data[:, i, PlayerData.BOOST_AMOUNT] = df["boost_amount"].values

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
            ep.player_data[start_idx:end_idx, idx, PlayerData.IS_DEMOED] = 1
            ep.player_data[start_idx:end_idx, idx, PlayerData.RESPAWN_TIMER] = timer

        # Goal
        ball_y = replay.ball_df.loc[end_frame, "pos_y"]
        ep.next_goal_side[:] = np.sign(ball_y) * (goal_frame is not None)
        ep.time_until_end[:] = times.loc[end_frame] - times.loc[start_frame:end_frame]

        if normalize:
            ep.normalize()

        yield ep


if __name__ == '__main__':
    replay = ParsedReplay.load("./test_replays/00029e4d-242d-49ed-971d-1218daa2eefa")
    for ep in replay_to_data(replay):
        debug = True
