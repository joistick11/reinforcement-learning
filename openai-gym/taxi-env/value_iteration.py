# Value iteration algorithm

import gym
import numpy as np
import time
from gym import wrappers

global number_of_states, number_of_actions, gamma, P


def update_value_function(v_current):
    v_new = np.zeros(number_of_states)
    for s in range(number_of_states):
        profits = np.zeros(number_of_actions)
        for a in range(number_of_actions):
            # The goal is to find the action with best "future" for every state
            for prob, next_state, reward, is_final in P[s][a]:
                profits[a] += prob * (reward + gamma * v_current[next_state] * (not is_final))
        v_new[s] = max(profits)
    return v_new


def create_policy(v_inp):
    policy = np.zeros(number_of_states)
    for s in range(number_of_states):
        profits = np.zeros(number_of_actions)
        for a in range(number_of_actions):
            for prob, next_state, reward, is_final in P[s][a]:
                profits[a] += prob * (reward + gamma * v_inp[next_state])
        policy[s] = np.argmax(profits)
    return policy


if __name__ == '__main__':
    env = gym.make('Taxi-v2')
    env = wrappers.Monitor(env, 'taxi-results', force=True)

    number_of_states = env.env.env.nS  # overall number of states in the env
    number_of_actions = env.env.env.nA  # overall number of actions on each state
    P = env.env.env.P  # transition matrix
    gamma = 0.999999
    number_of_episodes = 1000

    # init value function
    v = np.zeros(number_of_states)

    start = time.time()
    # Evaluating value-function
    while True:
        v_old = v.copy()
        v = update_value_function(v)
        if np.all(v == v_old):  # value function has converged
            print("Value function has converged")
            break

    policy = create_policy(v).astype(np.int)
    end = time.time()
    print("Time: " + str(end - start))

    # apply policy
    overall_reward = 0.0
    for _ in range(0, 2):
        current_state = env.reset()
        while True:
            current_state, reward, done, info = env.step(policy[current_state])
            env.render()
            time.sleep(0.5)
            overall_reward += reward
            if done:
                break

    print("Average reward: " + str(overall_reward / number_of_episodes))
    env.close()
