import numpy as np
from matplotlib import pyplot as plt

from c_mu_rule import jobs, states, state_to_index, compute_state_reward, compute_policy_value, get_policy1


def simulate_next_state(sate_idx, action):
    # return next state and reward
    state = states[sate_idx]
    next_state = state.copy()

    if np.random.rand() < jobs[0, action]:
        next_state[action] = 1

    next_state_idx = state_to_index(next_state)

    return next_state_idx, compute_state_reward(state)


def TD0_update(state_id, reward, next_state_id, alpha, value):
    # update value function
    value[state_id] += alpha * (reward + value[next_state_id] - value[state_id])


def TD_lambda_update(states_id, rewards, next_states_id, alpha, value, lambda_):
    delta_lambda = 0
    for i in range(len(states_id)):
        delta_lambda += (lambda_ ** i) * (rewards[i] + value[next_states_id[i]] - value[states_id[i]])

    value[states_id[0]] += alpha * delta_lambda


def alpha_fn_1(state_visit_count):
    return 1 / state_visit_count


def alpha_fn_2(state_visit_count):
    return 0.01


def alpha_fn_3(state_visit_count):
    return 10 / (100 + state_visit_count)


def run_TD0(policy, true_value, alpha_fn, max_iterations=1000):
    # initialize value function
    value = np.zeros(states.shape[0])
    # run TD0 for alpha1:
    value_norm_error = []
    value0_error = []
    state_visit_count = np.zeros(states.shape[0])
    state_id = 0
    for i in range(max_iterations):
        value_norm_error.append(np.linalg.norm(value - true_value, ord=np.inf))
        value0_error.append(np.abs(value[0] - true_value[0]))

        state_visit_count[state_id] += 1
        action = policy[state_id]
        next_state_id, reward = simulate_next_state(state_id, action)
        TD0_update(state_id, reward, next_state_id, alpha_fn(state_visit_count[state_id]), value)

        if next_state_id == 31:
            # terminal state. start over
            state_id = 0
        else:
            state_id = next_state_id

    return value, value_norm_error, value0_error


def plot_error(value_norm_error, value0_error, title):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(value_norm_error)
    ax[0].set_title('Value inf norm error')
    ax[1].plot(value0_error)
    ax[1].set_title('Value0 error')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    policy1 = get_policy1()
    policy1_value = compute_policy_value(policy1)

    num_iterations = 20000

    _, value_norm_error_alpha1, value0_error_alpha1 = run_TD0(policy1, policy1_value, alpha_fn_1, num_iterations)
    plot_error(value_norm_error_alpha1, value0_error_alpha1, "alpha1")

    _, value_norm_error_alpha2, value0_error_alpha2 = run_TD0(policy1, policy1_value, alpha_fn_2, num_iterations)
    plot_error(value_norm_error_alpha2, value0_error_alpha2, "alpha2")

    _, value_norm_error_alpha3, value0_error_alpha3 = run_TD0(policy1, policy1_value, alpha_fn_3, num_iterations)
    plot_error(value_norm_error_alpha3, value0_error_alpha3, "alpha3")

