import numpy as np


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


if __name__ ==
    value = get_init_value()

