import gym_flood
import gym

from matplotlib import pyplot as plt

#main function where the execusion should start
def main():
    print ('Hello, let\'s Flood It!!!')

    env = gym.make("Flood-v0")
    observation, possible = env.reset()
    print(observation)
    env.render()

    observation, reward, done, info = env.step(2)
    print(observation)
    env.render()

    env.step(3)
    print(observation)
    env.render()

    plt.show()

main()