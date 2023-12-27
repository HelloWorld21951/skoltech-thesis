import time
from collections import defaultdict

import numpy as np


class Timer(object):
    def __init__(self):
        self._starts = defaultdict(float)
        self._deltas = defaultdict(list)
        self._counts = defaultdict(int)

    def tick(self, name=None):
        self._starts[name] = time.time()
        self._counts[name] += 1

    def tock(self, name=None):
        self._deltas[name].append(time.time() - self._starts[name])

    def print(self):
        time_stat = ''
        for key, value in self._deltas.items():
            mean = np.round(1000 * np.mean(value), 1)
            std = np.round(1000 * np.std(value), 1)
            time_stat += ("Duration of " + str(key) + " = " + str(mean) + " +- " + str(std) + " ms" + "\n")
        return time_stat
