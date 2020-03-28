import numpy as np


def random_tape(length, max_char=6):
    """returns array of length N of random numbers in range(0,max_char)"""

    tape = np.random.randint(0, high=max_char, size=length)

    return tape
