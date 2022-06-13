import numpy as np
from matplotlib import pyplot as plt

from c_mu_rule import jobs, states, state_to_index, compute_state_reward, compute_policy_value, get_policy1, \
    get_cmu_law_policy


def simulate_next_state(sate_idx, action):
    # return next state and reward
    state = states[sate_idx]
    next_state = state.copy()

    if np.random.rand() < jobs[0, action]:
        next_state[action] = 1

    next_state_idx = state_to_index(next_state)

    return next_state_idx, compute_state_reward(state)


def alpha_fn_1(state_visit_count):
    return 1 / state_visit_count


def alpha_fn_2(state_visit_count):
    return 0.01


def alpha_fn_3(state_visit_count):
    return 10 / (100 + state_visit_count)


def TD0_update(state_id, reward, next_state_id, alpha, value):
    # update value function
    value[state_id] += alpha * (reward + value[next_state_id] - value[state_id])


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


def TD_lambda_update(states_id, rewards, next_states_id, alpha, value, lambda_):
    delta_lambda = 0
    for i in range(len(states_id)):
        delta_lambda += (lambda_ ** i) * (rewards[i] + value[next_states_id[i]] - value[states_id[i]])

    value[states_id[0]] += alpha * delta_lambda


def run_TD_lambda(policy, true_value, alpha_fn, lambda_, max_episodes=200):
    # initialize value function
    value = np.zeros(states.shape[0])
    # run TD0 for alpha1:
    value_norm_error = []
    value0_error = []

    state_visit_count = np.zeros(states.shape[0])
    iter = 0

    for i in range(max_episodes):
        value_norm_error.append(np.linalg.norm(value - true_value, ord=np.inf))
        value0_error.append(np.abs(value[0] - true_value[0]))

        state_id_buffer = []
        reward_buffer = []
        next_state_id_buffer = []
        state_id = 0
        # collect rollout from one episode:
        while state_id != 31:
            iter += 1
            state_visit_count[state_id] += 1

            action = policy[state_id]
            next_state_id, reward = simulate_next_state(state_id, action)
            state_id_buffer.append(state_id)
            reward_buffer.append(reward)
            next_state_id_buffer.append(next_state_id)

            state_id = next_state_id

        for j in range(len(state_id_buffer)):
            TD_lambda_update(state_id_buffer, reward_buffer, next_state_id_buffer,
                             alpha_fn(state_visit_count[state_id_buffer[0]]), value, lambda_)
            state_id_buffer.pop(0)
            reward_buffer.pop(0)
            next_state_id_buffer.pop(0)

    return value, value_norm_error, value0_error


def plot_error(value_norm_error, value0_error, title):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(value_norm_error)
    ax[0].set_title('Value inf norm error')
    ax[1].plot(value0_error)
    ax[1].set_title('Value0 error')
    fig.suptitle(title)
    plt.show()


def get_gridy_value_from_Q(Q):
    value = np.zeros(states.shape[0])
    for i in range(states.shape[0]):
        value[i] = np.min(Q[i])
    return value


def Q_learning(optimal_value, alpha_fn, epsilon, max_iterations=20000):
    value_error_norm = []
    value0_error = []

    # initialize Q table
    Q = np.zeros((32, 5))
    state_action_visit_count = np.zeros_like(Q)

    state_id = 0
    for i in range(max_iterations):
        # calculate errors:
        gridy_value = get_gridy_value_from_Q(Q)
        value_error_norm.append(np.linalg.norm(gridy_value - optimal_value, ord=np.inf))
        value0_error.append(np.abs(gridy_value[0] - optimal_value[0]))

        # choose action according to epsilon greedy policy:
        if np.random.rand() < epsilon:
            action = np.random.randint(0, 5)
        else:
            action = np.argmin(Q[state_id])

        # simulate next state and reward:
        next_state_id, reward = simulate_next_state(state_id, action)
        state_action_visit_count[state_id, action] += 1

        # update Q table:
        alpha = alpha_fn(state_action_visit_count[state_id, action])
        Q[state_id, action] += alpha * (reward + np.min(Q[next_state_id]) - Q[state_id, action])

        if next_state_id == 31:
            # terminal state. start over
            state_id = 0
        else:
            state_id = next_state_id

    return Q, value_error_norm, value0_error


if __name__ == "__main__":
    policy1 = get_policy1()
    policy1_value = compute_policy_value(policy1)

    ###### TD0 ######
    # num_iterations = 20000
    #
    # _, value_norm_error_alpha1, value0_error_alpha1 = run_TD0(policy1, policy1_value, alpha_fn_1, num_iterations)
    # plot_error(value_norm_error_alpha1, value0_error_alpha1, "alpha1")
    #
    # _, value_norm_error_alpha2, value0_error_alpha2 = run_TD0(policy1, policy1_value, alpha_fn_2, num_iterations)
    # plot_error(value_norm_error_alpha2, value0_error_alpha2, "alpha2")
    #
    # _, value_norm_error_alpha3, value0_error_alpha3 = run_TD0(policy1, policy1_value, alpha_fn_3, num_iterations)
    # plot_error(value_norm_error_alpha3, value0_error_alpha3, "alpha3")

    ###### TD lambda ######
    # num_episodes = 200
    #
    # for lambda_ in [0.1, 0.25, 0.5, 0.9, 1]:
    #     value_norm_error_list = []
    #     value0_error_list = []
    #     for i in range(20):
    #         _, value_norm_error_lambda1, value0_error_lambda1 = run_TD_lambda(policy1, policy1_value, alpha_fn_1,
    #                                                                     lambda_=lambda_, max_episodes=num_episodes)
    #         value_norm_error_list.append(value_norm_error_lambda1)
    #         value0_error_list.append(value0_error_lambda1)
    #
    #     value_norm_error_lambda = np.mean(value_norm_error_list, axis=0)
    #     value0_error_lambda = np.mean(value0_error_list, axis=0)
    #
    #     plot_error(value_norm_error_lambda, value0_error_lambda, "lambda = " + str(lambda_))

    ###### Q learning ######
    num_iterations = 30000
    epsilon = 0.1

    optimal_policy = get_cmu_law_policy()
    optimal_policy_value = compute_policy_value(optimal_policy)

    # _, value_norm_error_alpha1, value0_error_alpha1 = Q_learning(optimal_policy_value, alpha_fn_1,
    #                                                              epsilon, num_iterations)
    # plot_error(value_norm_error_alpha1, value0_error_alpha1, "alpha1")
    #
    # _, value_norm_error_alpha2, value0_error_alpha2 = Q_learning(optimal_policy_value, alpha_fn_2,
    #                                                              epsilon, num_iterations)
    # plot_error(value_norm_error_alpha2, value0_error_alpha2, "alpha2")
    #
    # _, value_norm_error_alpha3, value0_error_alpha3 = Q_learning(optimal_policy_value, alpha_fn_3,
    #                                                              epsilon, num_iterations)
    # plot_error(value_norm_error_alpha3, value0_error_alpha3, "alpha3")

    epsilon = 0.01
    _, value_norm_error_alpha3, value0_error_alpha3 = Q_learning(optimal_policy_value, alpha_fn_3,
                                                                 epsilon, num_iterations)
    plot_error(value_norm_error_alpha3, value0_error_alpha3, "alpha3, epsilon = 0.01")