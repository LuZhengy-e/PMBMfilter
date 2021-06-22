import numpy as np
from filter import Kalman_Filter


class Bernoulli(Kalman_Filter):
    def __init__(self, x, P, r, id, motion_model, meas_model, ps, pd, intensity_c):
        super(Bernoulli, self).__init__(motion_model=motion_model, measure_model=meas_model)
        self.ps = ps  # Probability of survival
        self.pd1 = pd  # Probability of detection
        self.pd2 = pd
        self.intensity_c = intensity_c
        self.x = x
        self.P = P
        self.r = r
        self.id = id

    def bern_predict(self):
        self.r = self.r * self.ps
        self.x, self.P = self.predict(self.x, self.P)

    def undetected_bern_update(self):
        unpd = (1 - self.pd1) * (1 - self.pd2)  # undetect probability
        self.r = self.r * unpd / (1 - self.r + self.r * unpd)
        l = np.log((1 - self.r + self.r * unpd))

        return l

    def detected_bern_update(self, meas):
        merge = False
        Sg = 0
        if meas[0] is None:
            z = meas[1]
            pd = (1 - self.pd1) * self.pd2
        elif meas[1] is None:
            z = meas[0]
            pd = (1 - self.pd2) * self.pd1
        else:
            Sg = self.calculate_logscale(meas[0], meas[1])
            z = (meas[0] + meas[1]) / 2
            self.R /= 2
            pd = self.pd1 * self.pd2
            merge = True

        l = self.calculate_logweight(self.x, self.P, z) + np.log(pd) + np.log(self.r) + Sg
        self.r = 1
        self.x, self.P = self.update(self.x, self.P, z)
        if merge:
            self.R *= 2

        return l

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
