import numpy as np
from filter import Kalman_Filter


class Bernoulli(Kalman_Filter):
    def __init__(self, x, P, r, id, motion_model, meas_model, ps, pd, intensity_c):
        super(Bernoulli, self).__init__(motion_model=motion_model, measure_model=meas_model)
        self.ps = ps  # Probability of survival
        self.pd = pd  # Probability of detection
        self.intensity_c = intensity_c
        self.x = x
        self.P = P
        self.r = r
        self.id = id

    def bern_predict(self):
        self.r = self.r * self.ps
        self.x, self.P = self.predict(self.x, self.P)

    def undetected_bern_update(self):
        unpd = 1 - self.pd  # undetect probability
        self.r = self.r * unpd / (1 - self.r + self.r * unpd)
        l = np.log((1 - self.r + self.r * unpd))

        return l

    def detected_bern_update(self, meas):
        l = self.calculate_logweight(self.x, self.P, meas) + np.log(self.pd) + np.log(self.r)
        self.r = 1
        self.x, self.P = self.update(self.x, self.P, meas)

        return l

    def calculate_logweight(self, x, P, z):
        S = np.dot(np.dot(self.H, P), self.H.T) + self.R
        S = (S + S.T) / 2
        delta = z - np.dot(self.H, x)
        l = - 0.5 * np.dot(np.dot(delta.T, np.linalg.inv(S)), delta) \
            - 0.5 * np.log(np.linalg.det(2 * np.pi * S))

        return l
