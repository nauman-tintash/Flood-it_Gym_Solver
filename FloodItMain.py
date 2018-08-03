import gym_flood
import gym
from random import choice
from collections import defaultdict

from matplotlib import pyplot as plt

#main function where the execusion should start
def main():
    print ('Hello, let\'s Flood It!!!')
    env = gym.make("Flood-v0")
    observation, possible = env.reset()
    env.render()
    bestmoves = 22
    
    current = observation
    previous = observation
    done = False

    stepAction = 0

    env.step(2)
    
    env.render()
    
    while done != True:
    #    for row in range(0,12):
    #        for column in range(len(observation[row])):
    #            current = observation[row][column]
    #            previous = observation[row][column-1]

        stepAction = (stepAction + 1) % 6
        observation, reward, done, info = env.step(stepAction)

        env.render()
                # if (current != previous):
                #     observation, reward, done, info = env.step(current)
                #     env.render()
                #     print (len(info["moves"]))
                # if (len(info["moves"]) == bestmoves):
                #     print ("Failed to complete the game")
                #     stop = True
                #     observation, possible = env.reset()
                #     return
            
    

    plt.show()

main()