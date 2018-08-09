from functools import reduce
import inspect
import math
# import numpy as np


# Python class to calculate statistics on datasets
class DataStatistics(object):

    def __init__(self, data):
        self.data = data
        self.data_lengths = list()
        self.length_map = dict()
        self.N = len(self.data)

    # pretty straightforward
    @staticmethod
    def compute_mean(data):
        sum = 0
        for num in data:
            sum += num

        return sum / len(data)

    @staticmethod
    def create_length_map(data):
        length_map = {}

        for message in data:
            if len(message) in length_map:
                length_map[len(message)] += 1
            else:
                length_map[len(message)] = 1

        return length_map

    @classmethod
    def compute_mode(cls, data):
        length_map = cls.create_length_map(data)

        max_length_seen, mode = 0, 0

        for number, count in length_map.items():

            # hopefully it doesn't register this as a tuple
            max_length_seen, mode = (count, number) if count > max_length_seen else (max_length_seen, mode)

        return mode, max_length_seen


    # Returns standard deviation, variance
    @classmethod
    def compute_standard_deviation(cls, data, average=None):
        if average is None:
            average = cls.compute_mean(data)
        try:
            variance = float(reduce((lambda data, sum: sum + (data - average) ** 2), data)) / float(len(data))

        except ZeroDivisionError:
            record_frame = inspect.stack()[0]
            print("Error: Division by zero at function compute_standard_deviation at {} in file {}".format(
                inspect.getframeinfo(record_frame[0]).lineno, __file__))

            return 0, 0

        return math.sqrt(variance), variance




