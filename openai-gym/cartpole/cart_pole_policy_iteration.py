from random import randint

import gym
import math
from gym import wrappers
import numpy as np

global env
env = gym.make('CartPole-v0')

STATES = {}  # {obs: sid}
V = []
pi = []
gamma = 0.99
NUM_EPISODES = 5000
P = {}  # transition matrix

# Number of discrete states (bucket) per state dimension
NUM_BUCKETS = (5, 5, 10, 5)  # (x, x', theta, theta')
# Number of discrete actions
NUM_ACTIONS = env.action_space.n # (left, right)
# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
# Manually setting bounds (needed for the x_dot and theta_dot)
STATE_BOUNDS[1] = [-0.5, 0.5]
STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]
# Index of the action
ACTION_INDEX = len(NUM_BUCKETS)


# global last_action


def simulate():
    global env
    env = wrappers.Monitor(env, 'cart-pole-results', force=True)

    for _ in range(NUM_EPISODES):
        current_state = env.reset()
        if _ % 250 == 0:
            print("UPDATING POLICY")
            update_policy()
            print("DONE")
        if state_to_bucket(current_state) not in STATES:
            initialize_new_state(state_to_bucket(current_state))
        while True:
            # env.render()
            action = determine_action(state_to_bucket(current_state))
            # print("Action: " + str(action))
            previous_state = env.env.env.state
            current_state, reward, done, _ = env.step(action)
            improve_policy(state_to_bucket(current_state), state_to_bucket(previous_state), action, reward)
            if done:
                break


def determine_action(state):
    if state in STATES:
        return pi[STATES[state]]
    else:
        return randint(0, 1)


def initialize_new_state(state):
    new_sid = len(V)
    STATES[state] = new_sid  # put new state to states
    V.insert(new_sid, 0)  # put value 0 for this action into V-table
    pi.insert(new_sid, randint(0, 1))  # and add new state to the policy


def improve_policy(curr_state, prev_state, action, reward):
    # it is a new state
    if curr_state not in STATES:
        initialize_new_state(curr_state)

    # if there is no information in transition matrix about prev state
    # or about the action done, then add it there
    if STATES[prev_state] not in P:
        P[STATES[prev_state]] = {}  # add observation to transitions
    if action not in P[STATES[prev_state]]:
        P[STATES[prev_state]][action] = [[1, curr_state, reward]]


def update_policy():
    # Until policy is stable
    while True:
        # Evaluate value-function
        for s in range(len(STATES)):
            V[s] = get_reward(s, pi[s])

        # Update policy
        previous_pi = pi.copy()
        for s in range(len(V)):
            if s not in P:
                continue  # suppose a starting state
            profits = np.zeros(len(V))
            for a in P[s]:
                for prob, next_state, reward in P[s][a]:
                    profits[a] += prob * (reward + gamma * V[STATES[next_state]])
            # choose the best action
            pi[s] = np.argmax(profits)

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
    for prob, next_state, reward in P[s][a]:
        v_for_a += prob * (reward + gamma * V[STATES[next_state]])
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
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


if __name__ == '__main__':
    simulate()
