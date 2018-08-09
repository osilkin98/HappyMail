from functools import reduce
import inspect
import math
# import numpy as np


# Python class to calculate statistics on datasets
class DataStatistics(object):

    def __init__(self, data=None):
        self.data = data
        self.data_lengths = list(map(lambda x: len(x), self.data))
        self.length_map = self.create_length_map(data=self.data)
        self.N = len(self.data)
        self.mean = self.compute_mean(data=self.data)
        self.std_dev, self.variance = self.compute_standard_deviation(data=self.data_lengths, average=self.mean)
        print(self.data)
        self.median = self.compute_median(data=self.data_lengths)
        self.mode, self.mode_count = self.compute_mode(data=self.data)



    # pretty straightforward
    @staticmethod
    def compute_mean(data):
        sum = 0
        for num in data:
            sum += len(num)

        return float(sum) / float(len(data))

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


    # Computes the median by using python's default sorting algorithm which runs in O(n log n)
    @staticmethod
    def compute_median(data):
        data = data.sort()
        N = len(data)

        return float(data[N/2]) if N % 2 == 0 else float(data[N/2 - 1] + data[N/2]) / 2



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




