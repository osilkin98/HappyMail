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

    @staticmethod
    def compute_mean(data):
        sum = 0
        for num in data:
            sum += num

        return sum / len(data)

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

        return



