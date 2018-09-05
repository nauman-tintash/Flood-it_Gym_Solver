import gym
import numpy as np
from time import sleep
import random

def without_RL(env):

    print("Action Space {}".format(env.action_space))
    print("State Space {}".format(env.observation_space))

    env.s = 328  # set environment to illustration's state

    epochs = 0
    penalties, reward = 0, 0

    frames = [] # for animation

    done = False

    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1
        
        # Put each rendered frame into dict for animation
        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
            }
        )

        epochs += 1
        
        
    print("Timesteps taken: {}".format(epochs))
    print("Penalties incurred: {}".format(penalties))
    return frames

def print_frames(frames):
    for i, frame in enumerate(frames):
        #clear_output(wait=True)
        print(frame['frame'].getvalue())
        print("Timestep:", {i + 1})
        print("State: ", frame['state'])
        print("Action:",frame['action'])
        print("Reward: ",frame['reward'])
        sleep(1)

def with_RL(env):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # Hyperparameters
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1


    for i in range(1, 100001):
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values

            next_state, reward, done, info = env.step(action) 
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1
            
        if i % 100 == 0:
            #clear_output(wait=True)
            print("Episode:",i)

    print("Training finished.\n")
    return q_table
def after_qTraining(env,q_table):
    """Evaluate agent's performance after Q-learning"""

    total_epochs, total_penalties = 0, 0
    episodes = 100
    frames = []

    for _ in range(episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0
        
        done = False
        
        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)

            if reward == -10:
                penalties += 1

            epochs += 1
            # Put each rendered frame into dict for animation
            frames.append({
                'frame': env.render(mode='ansi'),
                'state': state,
                'action': action,
                'reward': reward
                }
            )

            

        total_penalties += penalties
        total_epochs += epochs

    print("Results after ",episodes, "episodes")
    print("Average timesteps per episode: ",total_epochs / episodes)
    print("Average penalties per episode: ",total_penalties / episodes)
    return frames


def main():      
    env = gym.make("Taxi-v2").env

    env.render()

    #frames = without_RL(env)  
    q_table = with_RL(env)
    frame = after_qTraining(env,q_table)
    print_frames(frame)

main()