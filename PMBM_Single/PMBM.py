import os
import sys
from copy import deepcopy

import numpy as np
from scipy.stats.distributions import chi2
from scipy.optimize import linear_sum_assignment as linear_assignment

ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)

from Bernoulli import Bernoulli

MAX_VALUE = 1e8


class PMBMfilter():
    def __init__(self, motion_model, meas_model, birth_model, ps, pd, intensity_c,
                 max_hypo_num=100, max_object_num=100):
        self.motion_model = motion_model
        self.meas_model = meas_model
        self.birth_model = birth_model
        self.ps = ps
        self.pd = pd
        self.intensity_c = intensity_c
        self.max_hypo_num = max_hypo_num
        self.max_object_num = max_object_num
        self.undetected_objects = []
        self.detected_objects = []
        self.max_id = 0  # save current id, or id will switch in prune
        self.gate_rate = 0.9999

    def predict(self):
        print("---------predict----------")
        for poi_comp in self.undetected_objects:
            poi_comp.poisson_predict()
        self.undetected_objects.extend(deepcopy(self.birth_model))

        # use PMB first not PMBM
        for bern_comp in self.detected_objects:
            bern_comp.bern_predict()

    def update(self, z: list):
        print("-----------update---------")
        gate_size = chi2.ppf(self.gate_rate, 4)  # gate size use X^2 distribution
        z = z[0] + z[1]
        m = len(z)  # measurement number of each sensor
        nu, nd = len(self.undetected_objects), len(self.detected_objects)
        used_z_u = [False] * m
        used_z_d = [False] * m
        z_in_gate_u = [[False] * m for _ in range(nu)]
        z_in_gate_d = [[False] * m for _ in range(nd)]

        # if measurement in the gate of undetected objects
        for i in range(nu):
            for j in range(m):
                if self.calculate_gate(self.undetected_objects[i], gate_size, z[j]):
                    z_in_gate_u[i][j] = True
                    used_z_u[j] = True

        # if measurement in the gate of detected objects
        if nd != 0:
            for i in range(nd):
                for j in range(m):
                    if self.calculate_gate(self.detected_objects[i], gate_size, z[j]):
                        z_in_gate_d[i][j] = True
                        used_z_d[j] = True

            # update detected objects
            m_d = sum(used_z_d)
            z_in_gate_d = [[z_in_gate_d[i][j] for j in range(m) if used_z_d[j]] for i in range(nd)]
            z_d = [z[i] for i in range(m) if used_z_d[i]]

            hypo_table = [[None] * (nd) for _ in range(m_d + 1)]
            like_table = - np.ones(shape=(m_d + 1, nd), dtype=float) * MAX_VALUE
            for i in range(m_d + 1):
                for j in range(nd):
                    if i == 0:  # undectected object
                        obj = deepcopy(self.detected_objects[j])
                        like_table[i, j] = obj.undetected_bern_update()
                        hypo_table[i][j] = deepcopy(obj)
                    else:
                        obj = deepcopy(self.detected_objects[j])
                        if not (z_in_gate_d[j][i - 1]):
                            continue
                        like_table[i, j] = obj.detected_bern_update(z_d[i - 1])
                        hypo_table[i][j] = deepcopy(obj)

            # undetected objects
            z_in_gate_d_and_u = [[z_in_gate_u[i][j] for j in range(m) if used_z_d[j]] for i in range(nu)]
            hypo_table_u = [None] * (m_d + 1)
            like_table_u = np.ones(shape=(m_d + 1), dtype=float) * np.log(self.intensity_c)
            for i in range(m_d + 1):
                undetected_objs = [None] * nu
                for j in range(nu):
                    if i == 0:
                        continue
                    else:
                        if not z_in_gate_d_and_u[j][i - 1]:
                            continue
                        obj = deepcopy(self.undetected_objects[j])
                        obj.detected_poisson_update(z_d[i - 1])
                        undetected_objs[j] = deepcopy(obj)

                if not any(undetected_objs):
                    continue
                x, P, rho = self.componentmatching(undetected_objs)
                new_obj = Bernoulli(x, P, 1, 0, self.motion_model, self.meas_model,
                                    self.ps, self.pd, self.intensity_c)
                hypo_table_u[i] = deepcopy(new_obj)
                like_table_u[i] = np.log(rho)

            """
            use munkres to get best combinations of objects and meas
            """
            # construct confirm matrix
            res = 0
            L1 = np.ones((m_d, nd), dtype=float) * -MAX_VALUE
            L2 = np.ones((m_d, m_d), dtype=float) * -MAX_VALUE
            for r in range(m_d):
                for c in range(nd):
                    L1[r, c] = like_table[r + 1, c] - like_table[0, c]
                    res += like_table[0, c]
                L2[r, r] = like_table_u[r + 1]
            L = -np.concatenate((L1, L2), axis=1)
            row_, col_ = linear_assignment(L)
            log_weight = - L[row_, col_].sum() + res  # not used in this case

            self.detected_objects.clear()
            col = col_.tolist()
            for obj_idx in range(nd):
                if obj_idx not in col:
                    obj = hypo_table[0][obj_idx]
                    if not obj:
                        continue
                    self.detected_objects.append(deepcopy(obj))
                else:
                    meas_idx = col.index(obj_idx)
                    obj = hypo_table[meas_idx + 1][obj_idx]
                    if not obj:
                        continue
                    self.detected_objects.append(deepcopy(obj))

            for meas_idx in range(m_d):
                _meas_idx = meas_idx + nd
                if _meas_idx not in col:
                    continue
                new_obj = hypo_table_u[meas_idx + 1]
                if not new_obj:
                    continue
                new_obj.id = self.max_id
                self.max_id += 1
                self.detected_objects.append(deepcopy(new_obj))

        # undected meas update
        used_z_u_not_d = [used_z_u[j] and not used_z_d[j] for j in range(m)]
        z_u = [z[i] for i in range(m) if used_z_u_not_d[i]]
        z_in_gate_u_not_d = [[z_in_gate_u[i][j] for j in range(m) if used_z_u_not_d[j]] for i in range(nu)]
        m_u = sum(used_z_u_not_d)
        hypo_table_u_not_d = [None] * (m_u + 1)
        like_table_u_not_d = np.ones((m_u + 1)) * np.log(self.intensity_c)
        for i in range(m_u + 1):
            undetected_objs = [None] * nu
            for j in range(nu):
                if i == 0:
                    obj = deepcopy(self.undetected_objects[j])
                    obj.undetected_poisson_update()
                    hypo_table_u_not_d[i] = deepcopy(obj)

                else:
                    if not z_in_gate_u_not_d[j][i - 1]:
                        continue
                    obj = deepcopy(self.undetected_objects[j])
                    obj.detected_poisson_update(z_u[i - 1])
                    undetected_objs[j] = deepcopy(obj)

            if not any(undetected_objs):
                continue
            x, P, rho = self.componentmatching(undetected_objs)
            new_obj = Bernoulli(x, P, 1, 0, self.motion_model, self.meas_model,
                                self.ps, self.pd, self.intensity_c)
            hypo_table_u_not_d[i] = deepcopy(new_obj)
            like_table_u_not_d[i] = np.log(rho)

        for meas_idx in range(m_u):
            new_obj = hypo_table_u_not_d[meas_idx + 1]
            if not new_obj:
                continue
            new_obj.id = self.max_id
            self.detected_objects.append(deepcopy(new_obj))
            self.max_id += 1

        # undetected undetected objects
        for poi_comp in self.undetected_objects:
            poi_comp.undetected_poisson_update()

    def prune(self, prune_hypo, prune_bern, prune_poisson):
        print("-----------prune----------")
        undetected_objects = []
        for poi_comp in self.undetected_objects:
            if poi_comp.w > np.log(prune_poisson):
                undetected_objects.append(deepcopy(poi_comp))
        self.undetected_objects = undetected_objects

        detected_objects = []
        for bern_comp in self.detected_objects:
            if bern_comp.r > prune_bern:
                detected_objects.append(deepcopy(bern_comp))
        self.detected_objects = detected_objects

    def estimate(self, extract):
        extract_object = []
        for bern_comp in self.detected_objects:
            if bern_comp.r > extract:
                extract_object.append(deepcopy(bern_comp))

        return len(self.detected_objects), extract_object

    @staticmethod
    def calculate_gate(obj, gate_size, z):
        S = np.dot(np.dot(obj.H, obj.P), obj.H.T) + obj.R
        S = (S + S.T) / 2
        delta = z - np.dot(obj.H, obj.x)
        Mdistance = np.dot(np.dot(delta.T, np.linalg.inv(S)), delta)
        z_in_gate = Mdistance < gate_size

        return z_in_gate

    @staticmethod
    def componentmatching(poisson_obj_set):
        w = 0
        x = 0
        for obj in poisson_obj_set:
            if not obj:
                continue
            w += obj.w
            x += obj.w * obj.x
            shape = obj.P.shape

        x = x / w
        P = np.zeros(shape=shape)

        for obj in poisson_obj_set:
            if not obj:
                continue
            P += obj.w * (obj.P + np.dot((obj.x - x), (obj.x - x).T))

        P = P / w
        return x, P, w
