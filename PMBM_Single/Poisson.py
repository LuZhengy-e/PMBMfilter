import numpy as np
from filter import Kalman_Filter


class Poisson(Kalman_Filter):
    def __init__(self, x, P, w, motion_model, meas_model, ps, pd):
        super(Poisson, self).__init__(motion_model=motion_model, measure_model=meas_model)
        self.ps = ps  # Probability of survival
        self.pd = pd  # Probability of detection
        self.x = x
        self.P = P
        self.w = w

    def poisson_predict(self):
        self.w = self.w + np.log(self.ps)
        self.x, self.P = self.predict(self.x, self.P)

    def undetected_poisson_update(self):
        self.w = self.w + np.log(1 - self.pd)

    def detected_poisson_update(self, meas):
        self.w += self.calculate_logweight(self.x, self.P, meas) + np.log(self.pd)
        self.w = np.exp(self.w)
        self.x, self.P = self.update(self.x, self.P, meas)

    def calculate_logweight(self, x, P, z):
        S = np.dot(np.dot(self.H, P), self.H.T) + self.R
        S = (S + S.T) / 2
        delta = z - np.dot(self.H, x)
        l = - 0.5 * np.dot(np.dot(delta.T, np.linalg.inv(S)), delta) \
            - 0.5 * np.log(np.linalg.det(2 * np.pi * S))

        return l

    def calculate_logscale(self, z1, z2):
        R = self.R + self.R.T
        delta = z1 - z2
        Sg = - 0.5 * np.dot(np.dot(delta.T, np.linalg.inv(R)), delta) \
             - 0.5 * np.log(np.linalg.det(2 * np.pi * R))

        return Sg


