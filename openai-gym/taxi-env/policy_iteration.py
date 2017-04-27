# Policy iteration algorithm

import gym
import time
from gym import wrappers
import numpy as np

env = gym.make('Taxi-v2')
number_of_states = env.env.nS
number_of_actions = env.env.nA

P = env.env.P  # transition matrix
V = np.zeros(number_of_states)  # value function
pi = np.zeros(number_of_states, dtype=np.int)  # policy
gamma = 0.99


# Returns value function for state S when performing action a
def get_reward(s, a):
    global P, V
    v_for_a = 0
    for prob, next_state, reward, is_final in P[s][a]:
        v_for_a += prob * (reward + gamma * V[next_state] * (not is_final))
    return v_for_a


def run_episode():
    global env, pi
    current_state = env.reset()
    episode_reward = 0.0
    while True:
        # env.render()
        time.sleep(.05)
        current_state, reward, done, info = env.step(pi[current_state])
        episode_reward += reward
        if done:
            break
    return episode_reward


if __name__ == '__main__':
    global env, V, pi, P
    env = wrappers.Monitor(env, 'taxi-results', force=True)

    number_of_episodes = 1000

    start = time.time()
    # Until policy is stable
    while True:
        # Evaluate value-function
        for s in range(number_of_states):
            V[s] = get_reward(s, pi[s])

        # Update policy
        i = 1
        previous_pi = pi.copy()
        for s in range(number_of_states):
            profits = np.zeros(number_of_actions)
            for a in range(number_of_actions):
                for prob, next_state, reward, is_final in P[s][a]:
                    profits[a] += prob * (reward + V[next_state])
            # choose the best action
            pi[s] = np.argmax(profits)

        print(i)
        episode_reward = run_episode()
        print(episode_reward)

        # Done, moving to the testing
        if np.all(previous_pi == pi):
            print("Policy has converged")
            break
    end = time.time()
    print("Time: " + str(end - start))

    overall_reward = 0.0
    for _ in range(0, number_of_episodes):
        overall_reward += run_episode()

    print("Average reward: " + str(overall_reward / number_of_episodes))
    env.close()
