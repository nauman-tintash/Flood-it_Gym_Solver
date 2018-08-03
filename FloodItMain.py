import gym_flood
import gym
from random import choice
from collections import defaultdict

#main function where the execusion should start
def main():
    print ('Hello, let\'s Flood It!!!')
    env = gym.make("Flood-v0")
    observation, possible = env.reset()
    env.render("ansi")
    bestmoves = 22
    
    done = None
    current = observation
    previous = observation
    stop = False

    while stop != True:
        for row in range(0,12):
            for column in range(len(observation[row])):
                current = observation[row][column]
                previous = observation[row][column-1]
                if (current != previous):
                    observation, reward, done, info = env.step(current)
                    env.render("ansi")
                    print (len(info["moves"]))
                if (len(info["moves"]) == bestmoves):
                    print ("Failed to complete the game")
                    stop = True
                    observation, possible = env.reset()
                    return
            
    

main()