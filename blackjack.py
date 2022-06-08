import numpy as np
import matplotlib.pyplot as plt


# first row is for start card of 2, then goes on until 11
from matplotlib import cm

dealer_sum_probabilities = np.array([[0.12, 0.12, 0.12, 0.11, 0.11, 0.41],
                                    [0.12, 0.12, 0.12, 0.11, 0.10, 0.43],
                                    [0.11, 0.12, 0.11, 0.11, 0.10, 0.45],
                                    [0.10, 0.11, 0.11, 0.10, 0.10, 0.47],
                                    [0.13, 0.10, 0.10, 0.10, 0.09, 0.47],
                                    [0.36, 0.13, 0.08, 0.08, 0.07, 0.29],
                                    [0.12, 0.36, 0.13, 0.07, 0.07, 0.27],
                                    [0.11, 0.12, 0.35, 0.12, 0.06, 0.25],
                                    [0.10, 0.11, 0.11, 0.34, 0.11, 0.23],
                                    [0.10, 0.10, 0.10, 0.10, 0.33, 0.27], ])


def get_init_value():
    # value is a map between a tuple of (X,Y) or "win", "lose", "draw" and a value, we initialize all values with zero
    value = dict()
    for X in range(4, 22):
        for Y in range(2, 12):
            value[(X, Y)] = 0

    # initialize values we already know for the final states to speed up:
    value["win"] = 1
    value["lose"] = -1
    value["draw"] = 0

    return value


def card_probability(card):
    if 2<= card <= 9 or card == 11:
        return 1/13
    else:  # (card == 10)
        return 4/13


def compute_dealer_sum_probabilities(dealer_start_card):
    # return list of probabilities for the dealer sums to be 17, 18, 19, 20, 21, 22
    return dealer_sum_probabilities[dealer_start_card-2]


def p_next_state_at_stick(state):
    # return probability of next state at stick
    curr_dealer_sum_prob = compute_dealer_sum_probabilities(state[1])
    p_win, p_lose, p_draw = 0, 0, 0

    for i, prob in enumerate(curr_dealer_sum_prob):
        sum = i + 17
        if sum > 21:
            p_win += prob
            continue
        if state[0] == sum:
            p_draw += prob
        if state[0] > sum:
            p_win += prob
        if state[0] < sum:
            p_lose += prob

    return p_win, p_lose, p_draw


def p_next_state_at_hit(state):
    # return probability of next state at hit as a dictionary where the key is
    # the next state and the value is the probability

    X = state[0]
    p_next_state = dict()
    p_next_state["lose"] = 0

    for nextX in range(X+2, X+12):
        card = nextX - X
        if nextX > 21:
            p_next_state["lose"] += card_probability(card)
            continue

        p_next_state[(nextX, state[1])] = card_probability(card)

    assert(np.abs(sum(p_next_state.values()) - 1) < 0.00001)
    return p_next_state


def perform_one_value_iteration(value):
    policy_is_stick = dict()

    for state in value.keys():
        if state == "win" or state == "lose" or state == "draw":
            continue
        else:
            p_win, p_lose, p_draw = p_next_state_at_stick(state)
            new_value_stick = p_win * 1 + p_lose * (-1) + p_draw * 0

            p_next_states_at_hit = p_next_state_at_hit(state)
            new_value_hit = 0
            for next_state in p_next_states_at_hit.keys():
                new_value_hit += p_next_states_at_hit[next_state] * value[next_state]

            value[state] = max(new_value_stick, new_value_hit)
            policy_is_stick[state] = new_value_hit < new_value_stick

    return value, policy_is_stick


if __name__ == "__main__":
    value = get_init_value()

    for i in range(200):
        value, policy_is_stick = perform_one_value_iteration(value)

    # plot value:
    player, dealer, V = [], [], []
    for state in value.keys():
        if state == "win" or state == "lose" or state == "draw":
            continue

        player.append(state[0])
        dealer.append(state[1])
        V.append(value[state])
    player = np.array(player)
    dealer = np.array(dealer)
    V = np.array(V)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    bottom = -np.ones_like(V)

    surf = ax.bar3d(dealer, player, bottom, 1, 1, V+1, cmap='viridis')
    ax.set_xlabel("Dealer")
    ax.set_ylabel("Player")
    plt.show()

    policy = np.zeros((18, 10))
    for state in policy_is_stick.keys():
        if state == "win" or state == "lose" or state == "draw":
            continue
        policy[state[0] - 4, state[1] - 2] = policy_is_stick[state]

    plt.imshow(policy, cmap='gray', extent=[2, 12, 4, 22])
    plt.xticks(np.arange(2, 12, 1))
    plt.xlabel("Dealer")
    plt.ylabel("Player")
    plt.show()


