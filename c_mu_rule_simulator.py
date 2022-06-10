import numpy as np
from c_mu_rule import jobs, states, state_to_index, compute_state_reward


def simulate_next_state(sate_idx, action):
    # return next state and reward
    state = states[sate_idx]
    next_state = state.copy()

    if np.random.rand() < jobs[0, action]:
        next_state[action] = 1

    next_state_idx = state_to_index(next_state)

    return next_state_idx, compute_state_reward(state)


if __name__ == "__main__":
