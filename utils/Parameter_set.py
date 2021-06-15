"""
set Tracking parameter
"""
import numpy as np


class CVMotionModel:
    def __init__(self, T, sigma=1):
        self.F = np.array([[1, 0, T, 0],
                           [0, 1, 0, T],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        self.Q = np.array([[0.25 * T ** 4, 0, 0.5 * T ** 3, 0],
                           [0, 0.25 * T ** 4, 0, 0.5 * T ** 3],
                           [0.5 * T ** 3, 0, T ** 2, 0],
                           [0, 0.5 * T ** 3, 0, T ** 2]]) * sigma


class DisMeasureModel:
    def __init__(self, sigma=1):
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        self.R = np.identity(2) * sigma

