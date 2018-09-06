import gym
import numpy as np
import _pickle as pickle

def downsample(image):
    # Take only alternate pixels - basically halves the resolution of the image (which is fine for us)
    return image[::2, ::2, :]

def remove_color(image):
    """Convert all color (RGB is the third dimension in the image)"""
    return image[:, :, 0]

def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image

def preprocess_observations(input_observation, prev_processed_observation, input_dimensions):
    """ convert the 210x160x3 uint8 frame into a 6400 float vector """
    processed_observation = input_observation[35:195] # crop
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation != 0] = 1 # everything else (paddles, ball) just set to 1
    # Convert from 80 x 80 matrix to 1600 x 1 matrix
    processed_observation = processed_observation.astype(np.float).ravel()

    # subtract the previous frame from the current one so we are only processing on changes in the game
    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation
    else:
        input_observation = np.zeros(input_dimensions)
    # store the previous frame so we can subtract from it next time
    prev_processed_observations = processed_observation
    return input_observation, prev_processed_observations


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def relu(vector):
    vector[vector < 0] = 0
    return vector

def apply_neural_nets(observation_matrix, weights):
    
    """ Based on the observation_matrix and weights, compute the new hidden layer values and the new output layer values"""
    hidden_layer_values = np.dot(weights['W1'], observation_matrix) 
    hidden_layer_values = relu(hidden_layer_values)
    output_layer_values = np.dot(hidden_layer_values, weights['W2']) 
    output_layer_values = sigmoid(output_layer_values)
    
    #probs = output_layer_values / np.sum(output_layer_values, axis=1, keepdims=True)

    return hidden_layer_values, output_layer_values

def choose_action(probability):
    random_value = np.random.uniform()

    prob_thresh = 0
    for i in range(len(probability)):
        prob_thresh += probability[i]

        if random_value <= prob_thresh:
            return i+1

def compute_gradient(gradient_log_p, hidden_layer_values, observation_values, weights):
    delta_L = gradient_log_p
    dC_dw2 = np.dot(hidden_layer_values.T, delta_L)
    # delta_l2 = np.outer(delta_L, weights['W2'])
    delta_l2 = np.dot(delta_L, weights['W2'].T)
    delta_l2 = relu(delta_l2)
    dC_dw1 = np.dot(delta_l2.T, observation_values)

    return {
        'W1': dC_dw1,
        'W2': dC_dw2
    }

def update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate):
    epsilon = 1e-5

    for layer_name in weights.keys():
        g = g_dict[layer_name]
        expectation_g_squared[layer_name] = decay_rate * expectation_g_squared[layer_name] + (1 - decay_rate) * g**2
        weights[layer_name] += (learning_rate * g)/(np.sqrt(expectation_g_squared[layer_name] + epsilon))
        g_dict[layer_name] = np.zeros_like(weights[layer_name]) # reset batch gradient buffer


def discount_rewards(rewards, gamma):
    """ Actions you took 20 steps before the end result are less important to the overall result than an action you took a step ago.
    This implements that logic by discounting the reward on previous actions based on how long ago they were taken"""
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, rewards.size)):
        if rewards[t] != 0:
            running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

def discount_with_rewards(gradient_log_p, episode_rewards, gamma):
    """ discount the gradient with the normalized rewards """
    discounted_episode_rewards = discount_rewards(episode_rewards, gamma)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_episode_rewards -= np.mean(discounted_episode_rewards)
    discounted_episode_rewards /= np.std(discounted_episode_rewards)
    return gradient_log_p * discounted_episode_rewards


def main():
    env = gym.make("IceHockey-v0")
    observation = env.reset() # This gets us the image

    # hyperparameters
    episode_number = 0
    batch_size = 10
    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99
    num_hidden_layer_neurons = 200
    input_dimensions = 80 * 80
    output_dimensions = 5
    learning_rate = 1e-4

    resume = True # resume from previous checkpoint?
    render = False

    reward_sum = 0
    running_reward = None
    prev_processed_observations = None

    if resume:
        weights = pickle.load(open('save_icehockey_weights.p', 'rb'))
        expectation_g_squared = pickle.load(open('save_icehockey_expectation_g_squared.p', 'rb'))
        g_dict = pickle.load(open('save_icehockey_g_dict.p', 'rb'))


    else:
        weights = {
            'W1': np.random.randn(num_hidden_layer_neurons, input_dimensions) / np.sqrt(input_dimensions),
            'W2': np.random.randn(num_hidden_layer_neurons, output_dimensions) / np.sqrt(num_hidden_layer_neurons)
        }

        expectation_g_squared = {}
        g_dict = {}

        for layer_name in weights.keys():
            expectation_g_squared[layer_name] = np.zeros_like(weights[layer_name])
            g_dict[layer_name] = np.zeros_like(weights[layer_name])

    episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []

    while True:
        if render:
            env.render()
        processed_observations, prev_processed_observations = preprocess_observations(observation, prev_processed_observations, input_dimensions)
        hidden_layer_values, output_layer_values = apply_neural_nets(processed_observations, weights)
    
        episode_observations.append(processed_observations)
        episode_hidden_layer_values.append(hidden_layer_values)
        
        action = choose_action(output_layer_values)

        # carry out the chosen action
        observation, reward, done, info = env.step(action)

        reward_sum += reward
        episode_rewards.append(reward)

        Li = []
        yi = action - 1
        Li = np.maximum(0, output_layer_values - output_layer_values[yi] + 1)
        Li[yi] = 0

        loss_function_gradient = Li

        episode_gradient_log_ps.append(loss_function_gradient)

        if done: 
            episode_number += 1
            
             # Combine the following values for the episode
            episode_hidden_layer_values = np.vstack(episode_hidden_layer_values)
            episode_observations = np.vstack(episode_observations)
            episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
            episode_rewards = np.vstack(episode_rewards)

            # Tweak the gradient of the log_ps based on the discounted rewards
            episode_gradient_log_ps_discounted = discount_with_rewards(episode_gradient_log_ps, episode_rewards, gamma)
            gradient = compute_gradient(
                episode_gradient_log_ps_discounted,
                episode_hidden_layer_values,
                episode_observations,
                weights)

             # Sum the gradient for use when we hit the batch size
            for layer_name in gradient:
                g_dict[layer_name] += gradient[layer_name]
            
            if episode_number % batch_size == 0:
                update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate)
            
            episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], [] # reset values
            observation = env.reset() # reset env
            
            running_reward = reward_sum if running_reward is None else running_reward + reward_sum
            print ('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward/episode_number))

            if episode_number % 10 == 0: 
                pickle.dump(weights, open('save_icehockey_weights.p', 'wb'))
                pickle.dump(expectation_g_squared, open('save_icehockey_expectation_g_squared.p', 'wb'))
                pickle.dump(g_dict, open('save_icehockey_g_dict.p', 'wb'))
                
                print('-----------------Saved Data--------------------')

            reward_sum = 0
            prev_processed_observations = None




        
    env.close()
        
main()