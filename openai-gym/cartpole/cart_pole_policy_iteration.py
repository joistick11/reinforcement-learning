from random import randint

import gym
import math
from gym import wrappers
import numpy as np

global env
env = gym.make('CartPole-v0')

global V, pi
gamma = 0.95
EPISODES_FOR_LEARNING = 2000
EPISODES_FOR_TESTING = 1000
P = {}  # transition matrix

# STAFF RESPONSIBLE FOR DISCRETIZATION
# Number of discrete states (bucket) per state dimension
DESCR_DEMENTION = 12
NUM_BUCKETS = (DESCR_DEMENTION, DESCR_DEMENTION, DESCR_DEMENTION, DESCR_DEMENTION)  # (x, x', theta, theta')
# Number of discrete actions
NUM_ACTIONS = env.action_space.n  # (left, right)
# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
# Manually setting bounds (needed for the x_dot and theta_dot)
STATE_BOUNDS[1] = [-0.5, 0.5]
STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]
# Index of the action
ACTION_INDEX = len(NUM_BUCKETS)


def init():
    global V, pi
    V = [0] * DESCR_DEMENTION
    pi = [0] * DESCR_DEMENTION
    for i in range(0, DESCR_DEMENTION):
        V[i] = [0] * DESCR_DEMENTION
        pi[i] = [0] * DESCR_DEMENTION
    for i in range(0, DESCR_DEMENTION):
        for j in range(0, DESCR_DEMENTION):
            V[i][j] = [0] * DESCR_DEMENTION
            pi[i][j] = [0] * DESCR_DEMENTION
    for i in range(0, DESCR_DEMENTION):
        for j in range(0, DESCR_DEMENTION):
            for k in range(0, DESCR_DEMENTION):
                V[i][j][k] = [0] * DESCR_DEMENTION
                pi[i][j][k] = [randint(0, 1)] * DESCR_DEMENTION


def do_episode():
    global env
    current_state = env.reset()
    # if _ % 20 == 0:
    # update_policy()
    i = 0
    while True:
        i += 1
        action = determine_action(state_to_bucket(current_state))
        previous_state = env.env.env.state
        current_state, reward, done, _ = env.step(action)
        improve_policy(state_to_bucket(current_state), state_to_bucket(previous_state), action, reward)
        if done:
            break


def simulate():
    print("Start simulating")
    # update_policy()
    print("Policy updated")
    do_episode()


def determine_action(state):
    supposed_action = pi[state[0]][state[1]][state[2]][state[3]]
    if state in P:
        possible_actions = P[state]
        # we need to go deeper
        # and choose another action
        if supposed_action in possible_actions and len(possible_actions) == 1:
            return 1 - supposed_action

    return supposed_action


def improve_policy(curr_state, prev_state, action, reward):
    # if there is no information in transition matrix about prev state
    # or about the action done, then add it there
    if prev_state not in P:
        P[prev_state] = {}  # add observation to transitions
    if action not in P[prev_state]:
        P[prev_state][action] = [[1, curr_state, reward]]


def update_policy():
    global pi
    # Until policy is stable
    while True:
        # Evaluate value-function
        for i in range(0, DESCR_DEMENTION):
            for j in range(0, DESCR_DEMENTION):
                for k in range(0, DESCR_DEMENTION):
                    for l in range(0, DESCR_DEMENTION):
                        V[i][j][k][l] = get_reward((i, j, k, l), pi[i][j][k][l])

        # Update policy
        previous_pi = pi.copy()
        for i in range(0, DESCR_DEMENTION):
            for j in range(0, DESCR_DEMENTION):
                for k in range(0, DESCR_DEMENTION):
                    for l in range(0, DESCR_DEMENTION):
                        if (i, j, k, l) not in P:
                            continue  # suppose a starting state
                        profits = np.zeros(2)
                        for a in P[(i, j, k, l)]:
                            for prob, next_s, reward in P[(i, j, k, l)][a]:
                                profits[a] += prob * (reward + gamma * V[next_s[0]][next_s[1]][next_s[2]][next_s[3]])
                        # choose the best action
                        print(profits)
                        pi[i][j][k][l] = np.argmax(profits)

        # Done, moving to the testing
        if np.all(previous_pi == pi):
            # print("Policy has converged")
            break
        else:
            print("Policy has not converged")


# Returns value function for state S when performing action a
def get_reward(s, a):
    v_for_a = 0
    if s not in P:
        return 0
    if a in P[s]:
        for prob, next_s, reward in P[s][a]:
            v_for_a += prob * (reward + gamma * V[next_s[0]][next_s[1]][next_s[2]][next_s[3]])
    return v_for_a


def to_tuple(a):
    try:
        return tuple(to_tuple(i) for i in a)
    except TypeError:
        return a


def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i] - 1) * STATE_BOUNDS[i][0] / bound_width
            scaling = (NUM_BUCKETS[i] - 1) / bound_width
            bucket_index = int(round(scaling * state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


if __name__ == '__main__':
    global env
    env = wrappers.Monitor(env, 'cart-pole-results', force=True)
    init()
    print("Start exploring")
    for _ in range(0, 30):
        for _ in range(0, 600):
            do_episode()
        update_policy()
    print("Start simulating")
    for _ in range(0, 1000):
        do_episode()
