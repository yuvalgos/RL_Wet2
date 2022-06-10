import numpy as np
from itertools import product

from matplotlib import pyplot as plt

jobs = np.array([[0.6, 0.5, 0.3, 0.7, 0.1],
                 [1, 4, 6, 2, 9]])

# a map from state index to state. each state is a tuple of size 5 where value
# of '1' represents finished job and '0' represents unfinished job.
def get_state_mapping():
    return np.array([np.array(t) for t in list(product([0, 1], repeat=5))])


def state_to_index(states, state):
    return np.where((states == state).all(axis=1))[0][0]


def compute_state_reward(state):
    # return reward for the state
    r = 0
    for i in range(5):
        if state[i] == 0:
            r += jobs[1, i]
    return r


def compute_policy_value(states, policy):
    value = np.zeros(states.shape[0])

    for i in range(40):
        for i, v in enumerate(value):
            if i == 31:
                continue

            next_job_to_proccess = policy[i]
            finish_job_prob = jobs[0, next_job_to_proccess]

            current_state = states[i]
            next_state_if_finished = states[i]
            next_state_if_finished[next_job_to_proccess] = 1

            new_value_finished_job = compute_state_reward(next_state_if_finished) +\
                                     value[state_to_index(next_state_if_finished)]
            new_value_unfinished_job = compute_state_reward(current_state) +\
                                          value[i]
            value[i] = finish_job_prob * new_value_finished_job + (1 - finish_job_prob) * new_value_unfinished_job

    return value


def get_policy1(states):
    policy = np.zeros(states.shape[0], dtype=int)
    for i in range(policy.shape[0]):
        policy[i] = np.argmax(jobs[1, :] * (1 - states[i, :]))
    return policy


if __name__ == "__main__":
    states = get_state_mapping()

    policy1 = get_policy1(states)
    policy1_value = compute_policy_value(states, policy1)

    plt.plot(np.arange(32), policy1_value)
    plt.show()

    print(policy1_value)
