import numpy as np
import math
import itertools


def shapley_value(get_result_fn, player_pool):
    """
    Calculate the Shapley value of a player in a game.
    :param get_result_fn: A function that takes a list of players and returns the result of the game.
    :param player_pool: A list of players.
    :return: A dictionary of player to their Shapley value.
    """
    player_pool = sorted(player_pool)
    n = len(player_pool)
    shapley_values = {player: 0 for player in player_pool}

    # Compute the result of all possible permutations of players so we can reuse them.
    results = {}
    for r in range(0, n + 1):
        for comb in itertools.combinations(player_pool, r=r):
            comb = tuple(sorted(comb))
            if comb not in results:
                results[comb] = get_result_fn(comb)
            else:
                raise ValueError(f"Duplicate result for {comb}")  # Should only need to compute once

    for player in player_pool:
        pool_without_player = [p for p in player_pool if p != player]
        for r in range(0, len(pool_without_player) + 1):
            for comb in itertools.combinations(pool_without_player, r=r):
                perm_with_player = tuple(sorted(comb + (player,)))
                weight = 1 / (n * math.comb(n - 1, len(comb)))
                marginal_contribution = results[perm_with_player] - results[comb]
                shapley_values[player] += weight * marginal_contribution

    return shapley_values


if __name__ == '__main__':
    n = 6
    print(shapley_value(lambda x: len(x), list(range(n))))  # Should be all 1s
    print(shapley_value(lambda x: sum(x), list(range(n))))  # Player n should have a Shapley value of n
    print(shapley_value(lambda x: np.array([len(x), sum(x)]), list(range(n))))
