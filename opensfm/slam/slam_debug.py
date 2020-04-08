import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

disable_debug = True


class AvgTimings(object):
    def __init__(self):
        self.times = defaultdict(float)
        self.n_mean = defaultdict(int)

    def addTimes(self, timings):
        for (_, (k, v, _)) in timings.items():
            self.times[k] += v
            self.n_mean[k] += 1

    def printAvgTimings(self):
        for (k, v) in self.n_mean.items():
            print("{} with {} runs: {}s".format(k, v, self.times[k]/v))


avg_timings = AvgTimings()
