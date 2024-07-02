from rlgym.rocket_league.api import GameState

from pearl.data import EpisodeData, BallData, PlayerData, BoostData, GameInfo, NO_TEAM
from rlgym_tools.replays.convert import replay_to_rlgym
from rlgym_tools.replays.parsed_replay import ParsedReplay

# Selectors for vector values
BALL_POS_COLS = (BallData.POS_X, BallData.POS_Y, BallData.POS_Z)
BALL_VEL_COLS = (BallData.VEL_X, BallData.VEL_Y, BallData.VEL_Z)
BALL_ANG_VEL_COLS = (BallData.ANG_VEL_X, BallData.ANG_VEL_Y, BallData.ANG_VEL_Z)

PLAYER_POS_COLS = (PlayerData.POS_X, PlayerData.POS_Y, PlayerData.POS_Z)
PLAYER_VEL_COLS = (PlayerData.VEL_X, PlayerData.VEL_Y, PlayerData.VEL_Z)
PLAYER_FW_COLS = (PlayerData.FW_X, PlayerData.FW_Y, PlayerData.FW_Z)
PLAYER_UP_COLS = (PlayerData.UP_X, PlayerData.UP_Y, PlayerData.UP_Z)
PLAYER_ANG_VEL_COLS = (PlayerData.ANG_VEL_X, PlayerData.ANG_VEL_Y, PlayerData.ANG_VEL_Z)
PLAYER_FLIP_TORQUE_COLS = (PlayerData.FLIP_TORQUE_X, PlayerData.FLIP_TORQUE_Y, PlayerData.FLIP_TORQUE_Z)


def replay_to_data(replay: ParsedReplay, normalize: bool = True, ignore_unfinished: bool = False) -> EpisodeData:
    num_frames = len(replay.game_df)  # Upper bound on number of frames we keep
    episode_data = EpisodeData.new_empty(num_frames)

    i = 0
    episode_start_index = 0
    e = 0
    for replay_frame in replay_to_rlgym(replay, calculate_error=False):
        episode_data.episode_id[i] = e

        populate_index(episode_data, i, replay_frame)

        i += 1
        if replay_frame.scoreboard.go_to_kickoff or replay_frame.scoreboard.is_over:
            if ignore_unfinished and not replay_frame.state.goal_scored:
                episode_data = (episode_data[:episode_start_index] +
                                EpisodeData.new_empty(num_frames - episode_start_index))
                i = episode_start_index
            else:
                e += 1
                episode_start_index = i

    episode_data = episode_data[:i].clone()

    if normalize:
        episode_data.normalize()

    return episode_data


def populate_index(episode_data, i, replay_frame, include_interpolated_data=False):
    state: GameState = replay_frame.state
    scoreboard = replay_frame.scoreboard

    episode_data.game_info[i, GameInfo.TIME_REMAINING] = scoreboard.game_timer_seconds
    episode_data.game_info[i, GameInfo.IS_OVERTIME] = scoreboard.is_overtime
    episode_data.game_info[i, GameInfo.KICKOFF_TIMER] = scoreboard.kickoff_timer_seconds
    episode_data.game_info[i, GameInfo.BLUE_SCORE] = scoreboard.blue_score
    episode_data.game_info[i, GameInfo.ORANGE_SCORE] = scoreboard.orange_score

    episode_data.ball_data[i, 0, BALL_POS_COLS] = state.ball.position
    episode_data.ball_data[i, 0, BALL_VEL_COLS] = state.ball.linear_velocity
    episode_data.ball_data[i, 0, BALL_ANG_VEL_COLS] = state.ball.angular_velocity

    for j, (car_id, car) in enumerate(sorted(state.cars.items())):
        age = replay_frame.update_age[car_id]
        episode_data.player_data[i, j, PlayerData.AGE] = age
        if age == 0 or include_interpolated_data:  # or i == episode_start_index:
            episode_data.player_data[i, j, PlayerData.TEAM] = -1 if car.is_blue else 1  # car.team_num
            episode_data.player_data[i, j, PLAYER_POS_COLS] = car.physics.position
            episode_data.player_data[i, j, PLAYER_VEL_COLS] = car.physics.linear_velocity
            episode_data.player_data[i, j, PLAYER_FW_COLS] = car.physics.forward
            episode_data.player_data[i, j, PLAYER_UP_COLS] = car.physics.up
            episode_data.player_data[i, j, PLAYER_ANG_VEL_COLS] = car.physics.angular_velocity
            episode_data.player_data[i, j, PlayerData.BOOST_AMOUNT] = car.boost_amount
            episode_data.player_data[i, j, PlayerData.IS_DEMOED] = car.is_demoed
            episode_data.player_data[i, j, PlayerData.RESPAWN_TIMER] = car.demo_respawn_timer
            episode_data.player_data[i, j, PlayerData.ON_GROUND] = car.on_ground
            episode_data.player_data[i, j, PlayerData.SUPERSONIC_TIME] = car.supersonic_time
            episode_data.player_data[i, j, PlayerData.BOOST_ACTIVE_TIME] = car.boost_active_time
            episode_data.player_data[i, j, PlayerData.HANDBRAKE] = car.handbrake
            episode_data.player_data[i, j, PlayerData.HAS_JUMPED] = car.has_jumped
            episode_data.player_data[i, j, PlayerData.IS_HOLDING_JUMP] = car.is_holding_jump
            episode_data.player_data[i, j, PlayerData.IS_JUMPING] = car.is_jumping
            episode_data.player_data[i, j, PlayerData.JUMP_TIME] = car.jump_time
            episode_data.player_data[i, j, PlayerData.HAS_FLIPPED] = car.has_flipped
            episode_data.player_data[i, j, PlayerData.HAS_DOUBLE_JUMPED] = car.has_double_jumped
            episode_data.player_data[i, j, PlayerData.AIR_TIME_SINCE_JUMP] = car.air_time_since_jump
            episode_data.player_data[i, j, PlayerData.FLIP_TIME] = car.flip_time
            episode_data.player_data[i, j, PLAYER_FLIP_TORQUE_COLS] = car.flip_torque
            episode_data.player_data[i, j, PlayerData.IS_AUTOFLIPPING] = car.is_autoflipping
            episode_data.player_data[i, j, PlayerData.AUTOFLIP_TIMER] = car.autoflip_timer
            episode_data.player_data[i, j, PlayerData.AUTOFLIP_DIRECTION] = car.autoflip_direction
            # episode_data.player_data[i, j, PlayerData.IS_BOOSTING] = car.is_boosting
            # episode_data.player_data[i, j, PlayerData.IS_SUPERSONIC] = car.is_supersonic
            # episode_data.player_data[i, j, PlayerData.CAN_FLIP] = car.can_flip
            # episode_data.player_data[i, j, PlayerData.IS_FLIPPING] = car.is_flipping
            # episode_data.player_data[i, j, PlayerData.HAD_CAR_CONTACT] = car.had_car_contact
        else:
            episode_data.player_data[i, j, PlayerData.TEAM:] = episode_data.player_data[i - 1, j, PlayerData.TEAM:]
    episode_data.player_data[i, len(state.cars):, PlayerData.IGNORE] = 1

    episode_data.boost_data[i, 0, BoostData.TIMER_PAD_0:BoostData.TIMER_PAD_33 + 1] = state.boost_pad_timers
    episode_data.time_until_end[i] = replay_frame.episode_seconds_remaining

    next_goal_side = replay_frame.next_scoring_team if replay_frame.next_scoring_team is not None else NO_TEAM
    episode_data.next_goal_side[i] = next_goal_side

    winning_team = replay_frame.winning_team if replay_frame.winning_team is not None else NO_TEAM
    episode_data.match_win_side[i] = winning_team
