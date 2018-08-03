from collections import defaultdict
from random import choice

import gym
import gym.spaces

from gym.envs.registration import register

register(
    id='Flood-v0',
    entry_point='gym_flood.envs:FloodEnv',
    timestep_limit=1000,
    reward_threshold=1.0,
    nondeterministic = True
)


def test(itimes=20, jtimes=500, maxmoves=22):
    bestnummoves = maxmoves
    bestmoves = defaultdict(list)
    env = gym.make("Flood-v0")
    observation, possible = env.reset()
    env.render("ansi")
    for i in range(itimes):
        observation, possible = env.reset()
        info = {}
        for j in range(jtimes):
            if "possible" in info:
                action = choice(info["possible"][0])
            else:
                action = choice(possible)
            observation, reward, done, info = env.step(action)
            # print(reward)
            if len(info["moves"]) == bestnummoves and not done:
                print("%d %d failed" % (i, j))
                break
            if done:
                if len(info["moves"]) <= bestnummoves:
                    print("%d %d %d best %s" % (i, j, len(info["moves"]), info))
                    bestnummoves = len(info["moves"])
                    bestmoves[bestnummoves].append(info["moves"])
                    env.render("ansi")
                else:
                    print("%d %d %d not best" % (i, j, len(info["moves"])))
                break
    print(bestnummoves)
    print(bestmoves)

    return bestmoves

# eof
