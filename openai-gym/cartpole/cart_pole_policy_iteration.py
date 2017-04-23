from random import randint
import gym
from gym import wrappers
import numpy as np
import math

gym.envs.register(
    id='CartPole-v1337',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=400,      # MountainCar-v0 uses 200
)
env = gym.make('CartPole-v1337')
# env.env.spec.timestep_limit = 400
# env.env.spec.tags.wrapper_config.TimeLimit.max_episode_steps = 400

V = []
pi = []
P = {}  # transition matrix
TP_A = {}  # TP_A[state][action] -> number of actions A in state S
TP_A_S = {}  # TP_A_S[state][action][STATE] -> number of actions A in state S which leads to S'

gamma = 0.995
EPISODES_FOR_LEARNING = 2000
EPISODES_FOR_TESTING = 1000

# STAFF RESPONSIBLE FOR DISCRETIZATION
# Number of discrete states (bucket) per state dimension
DESCR_DIMENSION = 7
OVERALL_NUMBER_OF_STATES = DESCR_DIMENSION ** 4
NUM_BUCKETS = (DESCR_DIMENSION, DESCR_DIMENSION, DESCR_DIMENSION, DESCR_DIMENSION)  # (x, x', theta, theta')

# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
# Manually setting bounds (needed for the x_dot and theta_dot)
STATE_BOUNDS[1] = [-0.5, 0.5]
STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]


def init():
    global V, pi
    V = [0] * DESCR_DIMENSION
    pi = [0] * DESCR_DIMENSION
    for i in range(0, DESCR_DIMENSION):
        V[i] = [0] * DESCR_DIMENSION
        pi[i] = [0] * DESCR_DIMENSION
    for i in range(0, DESCR_DIMENSION):
        for j in range(0, DESCR_DIMENSION):
            V[i][j] = [0] * DESCR_DIMENSION
            pi[i][j] = [0] * DESCR_DIMENSION
    for i in range(0, DESCR_DIMENSION):
        for j in range(0, DESCR_DIMENSION):
            for k in range(0, DESCR_DIMENSION):
                V[i][j][k] = [0] * DESCR_DIMENSION
                pi[i][j][k] = [0] * DESCR_DIMENSION

    for i in range(0, DESCR_DIMENSION):
        for j in range(0, DESCR_DIMENSION):
            for k in range(0, DESCR_DIMENSION):
                for l in range(0, DESCR_DIMENSION):
                    pi[i][j][k][l] = randint(0, 1)


def simulate(iterations):
    global env
    print("Start simulating")
    for _ in range(0, iterations):
        current_state = env.reset()
        while True:
            state = state_to_bucket(current_state)
            action = pi[state[0]][state[1]][state[2]][state[3]]
            current_state, reward, done, _ = env.step(action)

            if done:
                break


def explore(iterations):
    global env
    print("Start exploring")
    for _ in range(0, iterations):
        current_state = env.reset()
        reward_for_previous_actions = 0
        prev_prev_state = ()
        prev_prev_action = -1
        i = 1
        while True:
            action = determine_action(state_to_bucket(current_state))
            previous_state = env.env.env.state
            current_state, reward, done, _ = env.step(action)

            if prev_prev_state != ():
                if prev_prev_state in P:
                    for prob, next_s, reward in P[prev_prev_state][prev_prev_action]:
                        if next_s == state_to_bucket(previous_state):
                            reward_for_previous_actions += reward * .01
            prev_prev_state = state_to_bucket(previous_state)
            prev_prev_action = action

            # if done and i < 150:
            #     reward = -5

            improve_policy(state_to_bucket(current_state), state_to_bucket(previous_state), action,
                           reward, reward_for_previous_actions)
            if done:
                break
            i += 1
    update_transition_probabilities()
    print("Finishing exploring")


def update_transition_probabilities():
    print("Updating transition probabilities")
    for state in P:
        for action in P[state]:
            new_list_for_state_action = []
            for prob, next_s, reward in P[state][action]:
                overall_number_of_actions = TP_A[state][action]
                number_of_actions_to_next_s = TP_A_S[state][action][next_s]
                actual_prob = 1.0 * number_of_actions_to_next_s / overall_number_of_actions
                new_list_for_state_action.append([actual_prob, next_s, reward])
            P[state][action] = new_list_for_state_action
    print("Finished")


def determine_action(state):
    supposed_action = pi[state[0]][state[1]][state[2]][state[3]]
    if state in P:
        possible_actions = P[state]
        # we need to go deeper
        # and choose another action
        if supposed_action in possible_actions and len(possible_actions) == 1:
            return 1 - supposed_action

    return supposed_action


def improve_policy(curr_state, prev_state, action, reward, reward_for_previous_action):
    # if there is no information in transition matrix about prev state
    # or about the action done, then add it there
    if prev_state not in P:
        P[prev_state] = {}  # add observation to transitions
        TP_A[prev_state] = {}
        TP_A_S[prev_state] = {}

    # new action, no information before, adding to the matrix
    if action not in P[prev_state]:
        P[prev_state][action] = [[0, curr_state, reward + reward_for_previous_action]]
        TP_A[prev_state][action] = 0
        TP_A_S[prev_state][action] = {}

    # not a new action, but the same action may lead to another state
    new_state_for_same_action = True
    for prob, state, reward in P[prev_state][action]:
        if state == curr_state:
            new_state_for_same_action = False
            break
    if new_state_for_same_action:
        P[prev_state][action].append([0, curr_state, reward + reward_for_previous_action])

    if curr_state not in TP_A_S[prev_state][action]:
        TP_A_S[prev_state][action][curr_state] = 0

    TP_A[prev_state][action] += 1
    TP_A_S[prev_state][action][curr_state] += 1


def update_policy():
    global pi
    print("Starting policy update")
    # Until policy is stable
    while True:
        # Evaluate value-function
        for i in range(0, DESCR_DIMENSION):
            for j in range(0, DESCR_DIMENSION):
                for k in range(0, DESCR_DIMENSION):
                    for l in range(0, DESCR_DIMENSION):
                        V[i][j][k][l] = get_reward((i, j, k, l), pi[i][j][k][l])

        # Update policy
        policy_changed = False
        for i in range(0, DESCR_DIMENSION):
            for j in range(0, DESCR_DIMENSION):
                for k in range(0, DESCR_DIMENSION):
                    for l in range(0, DESCR_DIMENSION):
                        if (i, j, k, l) not in P:
                            continue  # suppose a starting state
                        profits = np.zeros(2)
                        for a in P[(i, j, k, l)]:
                            for prob, next_s, reward in P[(i, j, k, l)][a]:
                                profits[a] += prob * (reward + gamma * V[next_s[0]][next_s[1]][next_s[2]][next_s[3]])
                        # choose the best action
                        # print(profits)
                        new_action = np.argmax(profits)
                        if pi[i][j][k][l] != new_action:
                            pi[i][j][k][l] = new_action
                            policy_changed = True

        # Done, moving to the testing
        if not policy_changed:
            # print("Policy has converged")
            break
        else:
            print("Policy has not converged")
    print("Finished with policy update")


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
    global STATE_BOUNDS
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
    # optimal was 10 times for 250 episodes
    for _ in range(0, 10):
        explore(250)
        update_policy()
    simulate(5000)
