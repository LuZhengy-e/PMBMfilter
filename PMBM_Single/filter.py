"""
Kalman Filter
"""
import numpy as np


class Kalman_Filter:
    def __init__(self, motion_model, measure_model):
        self.F = motion_model.F
        self.H = measure_model.H
        self.Q = motion_model.Q
        self.R = measure_model.R

    def predict(self, x, P):
        pre_x = np.dot(self.F, x)
        pre_P = np.dot(np.dot(self.F, P), self.F.T) + self.Q
        pre_P = (pre_P + pre_P.T) / 2
        return pre_x, pre_P

    def update(self, x, P, z):
        S = np.dot(np.dot(self.H, P), self.H.T) + self.R
        S = (S + S.T) / 2
        K = np.dot(np.dot(P, self.H.T), np.linalg.inv(S))
        delta = z - np.dot(self.H, x)
        I = np.identity(P.shape[1])

        upt_x = x + np.dot(K, delta)
        upt_P = np.dot(I - np.dot(K, self.H), P)
        upt_P = (upt_P + upt_P.T) / 2

        return upt_x, upt_P

