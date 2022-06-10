import numpy as np
from itertools import product

from matplotlib import pyplot as plt

jobs = np.array([[0.6, 0.5, 0.3, 0.7, 0.1],
                 [1, 4, 6, 2, 9]])

# a map from state index to state. each state is a tuple of size 5 where value
# of '1' represents finished job and '0' represents unfinished job.
states = np.array([np.array(t) for t in list(product([0, 1], repeat=5))])


def state_to_index(state):
    return np.where((states == state).all(axis=1))[0][0]


def compute_state_reward(state):
    # return reward for the state
    r = 0
    for i in range(5):
        if state[i] == 0:
            r += jobs[1, i]
    return r


def compute_policy_value(policy):
    value = np.zeros(states.shape[0])

    for i in range(40):
        for i, v in enumerate(value):
            if i == 31:
                continue

            next_job_to_proccess = policy[i]
            finish_job_prob = jobs[0, next_job_to_proccess]

            current_state = states[i]
            next_state_if_finished = states[i].copy()
            next_state_if_finished[next_job_to_proccess] = 1

            current_state_reward = compute_state_reward(current_state)

            new_value_finished_job = current_state_reward + \
                                     value[state_to_index(next_state_if_finished)]
            new_value_unfinished_job = current_state_reward + \
                                       value[i]
            value[i] = finish_job_prob * new_value_finished_job + (1 - finish_job_prob) * new_value_unfinished_job

    return value


def compute_greedy_policy(value):
    policy = np.zeros(states.shape[0], dtype=int)
    for i in range(policy.shape[0]):
        best_action = 0
        best_action_value = np.inf
        current_state = states[i].copy()
        for action in range(5):
            finish_job_prob = jobs[0, action]
            next_state_if_finished = current_state.copy()
            next_state_if_finished[action] = 1

            new_value_finished_job = compute_state_reward(next_state_if_finished) + \
                                     value[state_to_index(next_state_if_finished)]
            new_value_unfinished_job = compute_state_reward(current_state) + value[i]

            action_value = finish_job_prob * new_value_finished_job + (1 - finish_job_prob) * new_value_unfinished_job
            if action_value < best_action_value:
                best_action_value = action_value
                best_action = action

        policy[i] = best_action

    return policy


def policy_iteration(policy, max_iterations=32):
    state0_values = []

    for i in range(max_iterations):
        old_policy = policy.copy()
        value = compute_policy_value(policy)
        state0_values.append(value[0])
        policy = compute_greedy_policy(value)
        if np.all(policy == old_policy):
            break

    return policy, state0_values


def get_policy1():
    policy = np.zeros(states.shape[0], dtype=int)
    for i in range(policy.shape[0]):
        policy[i] = np.argmax(jobs[1, :] * (1 - states[i, :]))
    return policy


def get_cmu_law_policy():
    policy = np.zeros(states.shape[0], dtype=int)
    for i in range(policy.shape[0]):
        policy[i] = np.argmax(jobs[1, :] * jobs[0, :] * (1 - states[i, :]))
    return policy

if __name__ == "__main__":
    policy1 = get_policy1()
    policy1_value = compute_policy_value(policy1)

    plt.plot(np.arange(32), policy1_value)
    plt.title("Policy1 (take job with highest cost) Value")
    plt.show()

    policy1_optimal, state0_values = policy_iteration(policy1)
    plt.plot(np.arange(len(state0_values)), state0_values)
    plt.title("Value of state0 during policy iteration")
    plt.show()

    print(f"policy iteration on policy1 converged in {len(state0_values) - 1} iterations")

    policy_cmu_law = get_cmu_law_policy()
    print("c mu law policy:", policy_cmu_law)
    print("policy iteration policy:", policy1_optimal)

    policy_cmu_law_value = compute_policy_value(policy_cmu_law)
    plt.plot(np.arange(32), policy_cmu_law_value, label="optimal policy")
    plt.plot(np.arange(32), policy1_value, label="policy c")
    plt.legend()
    plt.xlabel("state")
    plt.ylabel("value")
    plt.title("values")
    plt.show()
