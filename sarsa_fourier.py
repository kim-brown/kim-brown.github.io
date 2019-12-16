import gym, itertools
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

# Initializations
num_episodes = 3000  # 1000
num_timesteps = 800  # 200
gym.envs.register(
    id="MountainCarLongerEpisodeLength-v0",
    entry_point="gym.envs.classic_control:MountainCarEnv",
    max_episode_steps=num_timesteps,  # MountainCar-v0 uses 200
    reward_threshold=-110.0,
)
env = gym.make("MountainCarLongerEpisodeLength-v0")
num_actions = env.action_space.n
dim = env.observation_space.high.size

# Parameters
# order of the basis functions
order = 3
# total number of such basis, 16(?)
num_ind = int(pow(order + 1.0, dim))
# multipliers are the coefficient vectors used within the computation of cosine or sine function computation
                           # 16 x 2
multipliers = np.zeros((num_ind, dim))

epsilon = 0.1
Lambda = 0.5
alpha = 0.00005
gamma = 0.99


xbar = np.zeros((2, dim))
xbar[0, :] = env.observation_space.low
xbar[1, :] = env.observation_space.high

ep_length = np.zeros(num_episodes)
np.set_printoptions(precision=2)

# These are the weights which the basis functions are multiplied to get a value
weights = np.zeros((num_ind, num_actions))

# this function returns a normalized state where all variable values are between 0 and 1.
def normalize_state(s):
    y = np.zeros(len(s))
    for i in range(len(s)):
        if s[i] > xbar[1, i]:
            y[i] = 1
        elif s[i] < xbar[0, i]:
            y[i] = 0
        else:
            y[i] = (s[i] - xbar[0, i]) / (xbar[1, i] - xbar[0, i])
    return y


# TODO
# Returns an ndarray basis functions
#  return a 1 x 16 array ?
def phi(state):
    "*** Fill in code to return the computed basis functions! ***"
    #functions = np.ones((2, 1))
    # functions[0] = x
    #functions[0] = s[0]
                                    #  16 x 2
    basis = np.cos(np.dot(np.multiply(multipliers, math.pi), state))
    return np.transpose(basis)


# TODO
# Create the fourier basis coefficients
def create_multipliers():
    global multipliers
    "*** Fill in the code to generate the coefficients for fourier " "basis functions and assign it to the variable multipliers***"
    for i in range(order+1):
        for j in range(order+1):
            multipliers[i*4+j][0] = i
            multipliers[i*4+j][1] = j


# Returns the value of an action at some state
def action_value(input_state, action):
    state_normalized = normalize_state(input_state)
    features = phi(state_normalized)
    Qvalue = np.dot(weights[:, action], features)
    return Qvalue


# Returns an exploratory action for an input
def learning_policy(input_state):
    global weights
    state_normalized = normalize_state(input_state)
    features = phi(state_normalized)
                #                16 x 3
    Qvalues = np.dot(features, weights)
    random_num = np.random.random()
    if random_num < 1. - epsilon:
        action_chosen = Qvalues.argmax()
    else:
        action_chosen = env.action_space.sample()
    return int(action_chosen)


def test_policy(state):
    global weights
    state_normalized = normalize_state(state)
    features = phi(state_normalized)
    Qvalues = np.dot(features, weights)
    return int(Qvalues.argmax())


# TODO
def SARSA_Learning():
    """
        Implement Sarsa-lambda algorithm to update qtable, etable and learning policy.
        Input:
            all parameters

        Output:
            This function does not return an output. It only updates the weights according to the algorithm.

    """
    global weights
    for ep in range(num_episodes):
        print(ep)
        #                   16 x 3
        e = np.zeros((num_ind, num_actions))
        state = env.reset()
        norm_state = normalize_state(state)
        features = phi(norm_state)
        action = learning_policy(state)
        rewards = np.array([0])

        # Each episode
        for t in range(num_timesteps):
            env.render()
            next_state, reward, done, info = env.step(action)
            rewards = np.append(rewards, reward)
            "*** Fill in the rest of the algorithm!***"
            delta = reward + np.multiply(weights, gamma *
                action_value(next_state, next_action)) -
                np.multiply(weights, action_value(state, action))
            weights = np.multiply(weights, alpha * delta * action_value(state, action))
            state = next_state
            action = next_action
            if done:
                break
            """
            for s in range(0, 16):
                next_action = learning_policy(next_state)
                # self.policy[state] = action
                for a in range(1, 3):
                    e[s][a] = np.add(action_value(s, a), np.multiply(e[s][a], gamma * Lambda))
                e[0] = np.zeros(num_actions)
                # 16 x 3
                for a in range(0, 3):
                    delta = reward + gamma * weights[s][a] * action_value(next_state, next_action) - weights[s][a] * action_value(state, action)
                    weights[s][a] = weights[s][a] + alpha * delta * e[s][a]
                state = next_state
                action = next_action
                if done:
                    break
                """
        ep_length[ep] = t
        print(np.sum(rewards))

    np.save("mountain_car_saved_weights.npy", weights)


def SARSA_Test(num_test_episodes):
    for ep in range(num_test_episodes):
        state = env.reset()
        rewards = np.array([0])

        # Each episode
        for t in range(num_timesteps):
            env.render()
            action = test_policy(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            rewards = np.append(rewards, reward)
            if done:
                print((rewards.shape))
                break

        print(np.sum(rewards))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        "*** run without input arguments to learn ***"
        create_multipliers()
        SARSA_Learning()
    elif sys.argv[1] == "test":
        "*** run to test the saved policy with input argument test ***"
        create_multipliers()
        weights = np.load("mountain_car_saved_weights.npy")
        num_test_episodes = 100
        SARSA_Test(num_test_episodes)
    else:
        print("unknown input argument")
