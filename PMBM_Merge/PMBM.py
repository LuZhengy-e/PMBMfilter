import numpy as np
import sys
import os
from copy import deepcopy
from scipy.stats.distributions import chi2

ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)

from Poisson import Poisson
from Bernoulli import Bernoulli
from utils.Parameter_set import CVMotionModel, DisMeasureModel


class PMBMfilter():
    def __init__(self, motion_model, meas_model, birth_model, ps, pd, intensity_c,
                 max_hypo_num=100, max_object_num=100):
        self.motion_model = motion_model
        self.meas_model = meas_model
        self.birth_model = birth_model
        self.ps = ps
        self.pd1 = pd
        self.pd2 = pd
        self.intensity_c = intensity_c
        self.max_hypo_num = max_hypo_num
        self.max_object_num = max_object_num
        self.undetected_objects = []
        self.detected_objects = []
        self.max_id = 0  # save current id, or id will switch in prune
        self.gate_rate = 0.999

    def predict(self):
        print("---------predict----------")
        for poi_comp in self.undetected_objects:
            poi_comp.poisson_predict()
        self.undetected_objects.extend(deepcopy(self.birth_model))

        # use PMB first not PMBM
        for bern_comp in self.detected_objects:
            bern_comp.bern_predict()

    def update(self, z):
        print("-----------update---------")
        gate_size = chi2.ppf(self.gate_rate, 2)  # gate size use X^2 distribution
        z1, z2 = z[0], z[1]
        m1, m2 = len(z1), len(z2)  # measurement number of each sensor
        nu, nd = len(self.undetected_objects), len(self.detected_objects)
        used_z1_u, used_z2_u = [False] * m1, [False] * m2
        used_z1_d, used_z2_d = [False] * m1, [False] * m2
        z1_in_gate_u, z2_in_gate_u = [[False] * m1 for _ in range(nu)], [[False] * m2 for _ in range(nu)]
        z1_in_gate_d, z2_in_gate_d = [[False] * m1 for _ in range(nd)], [[False] * m2 for _ in range(nd)]

        # if measurement in the gate of undetected objects
        for i in range(nu):
            for j in range(m1):
                if self.calculate_gate(self.undetected_objects[i], gate_size, z1[j]):
                    z1_in_gate_u[i][j] = True
                    used_z1_u[j] = True

            for j in range(m2):
                if self.calculate_gate(self.undetected_objects[i], gate_size, z2[j]):
                    z2_in_gate_u[i][j] = True
                    used_z2_u[j] = True

        # if measurement in the gate of detected objects
        for i in range(nd):
            for j in range(m1):
                if self.calculate_gate(self.detected_objects[i], gate_size, z1[j]):
                    z1_in_gate_d[i][j] = True
                    used_z1_d[j] = True

            for j in range(m2):
                if self.calculate_gate(self.detected_objects[i], gate_size, z2[j]):
                    z2_in_gate_d[i][j] = True
                    used_z2_d[j] = True

        # update detected objects
        m1_d = sum(used_z1_d)
        m2_d = sum(used_z2_d)
        z1_in_gate_d = [[z1_in_gate_d[i][j] for j in range(m1) if used_z1_d[j]] for i in range(nd)]
        z2_in_gate_d = [[z2_in_gate_d[i][j] for j in range(m2) if used_z2_d[j]] for i in range(nd)]
        z1_d = [z1[i] for i in range(m1) if used_z1_d[i]]
        z2_d = [z2[i] for i in range(m2) if used_z2_d[i]]

        hypo_table = [[[None] * (m2_d + 1) for _ in range(nd)] for _ in range(m1_d + 1)]
        like_table = - np.ones(shape=(m1_d + 1, nd, m2_d + 1), dtype=float) * np.inf
        for i in range(m1_d + 1):
            for j in range(nd):
                for k in range(m2_d + 1):
                    if i == 0 and k == 0:  # undectected object
                        obj = deepcopy(self.detected_objects[j])
                        like_table[i, j, k] = obj.undetected_bern_update()
                        hypo_table[i][j][k] = deepcopy(obj)
                    elif i == 0 and k != 0:
                        obj = deepcopy(self.detected_objects[j])
                        if not z2_in_gate_d[j][k]:
                            continue
                        like_table[i, j, k] = obj.detected_bern_update([[], z2_d[k - 1]])
                        hypo_table[i][j][k] = deepcopy(obj)
                    elif i != 0 and k == 0:
                        obj = deepcopy(self.detected_objects[j])
                        if not z1_in_gate_d[j][i]:
                            continue
                        like_table[i, j, k] = obj.detected_bern_update([z1_d[i - 1], []])
                        hypo_table[i][j][k] = deepcopy(obj)
                    else:
                        obj = deepcopy(self.detected_objects[j])
                        if not (z1_in_gate_d[j][i] and z2_in_gate_d[j][k]):
                            continue
                        like_table[i, j, k] = obj.detected_bern_update([z1_d[i - 1], z2_d[k - 1]])
                        hypo_table[i][j][k] = deepcopy(obj)

        # undetected objects
        z1_in_gate_d_and_u = [[z1_in_gate_u[i][j] for j in range(m1) if used_z1_d[j]] for i in range(nu)]
        z2_in_gate_d_and_u = [[z2_in_gate_u[i][j] for j in range(m2) if used_z2_d[j]] for i in range(nu)]
        hypo_table_u = [[None] * (m2_d + 1) for _ in range(m1_d + 1)]
        like_table_u = np.ones(shape=(m1_d + 1, m2_d + 1), dtype=float) * np.log(self.intensity_c)
        for i in range(m1_d + 1):
            for k in range(m2_d + 1):
                undetected_objs_0 = [None] * nu
                undetected_objs_1 = [None] * nu
                undetected_objs_01 = [None] * nu
                for j in range(nu):
                    if i == 0 and k == 0:
                        obj = deepcopy(self.undetected_objects[j])
                        obj.undetected_poisson_update()
                        hypo_table_u[i][k] = deepcopy(obj)

                    elif i == 0 and k != 0:
                        if not z2_in_gate_d_and_u[j][k]:
                            continue
                        obj = deepcopy(self.undetected_objects[j])
                        obj.detected_poisson_update([[], z2_d[k - 1]])
                        undetected_objs_1[j] = deepcopy(obj)

                    elif i != 0 and k == 0:
                        if not z1_in_gate_d_and_u[j][i]:
                            continue
                        obj = deepcopy(self.undetected_objects[j])
                        obj.detected_poisson_update([z1_d[i - 1], []])
                        undetected_objs_0[j] = deepcopy(obj)

                    else:
                        if not (z1_in_gate_d_and_u[j][i] and z2_in_gate_d_and_u[j][k]):
                            continue
                        obj = deepcopy(self.undetected_objects[j])
                        obj.detected_poisson_update(z1_d[i - 1], z2_d[k - 1])
                        undetected_objs_01[j] = deepcopy(obj)

                if i == 0 and k != 0:
                    if not undetected_objs_1:
                        continue
                    x, P, rho = self.componentmatching(undetected_objs_1)
                    exp_weight = 1 + rho
                    new_r = rho / exp_weight
                    new_obj = Bernoulli(x, P, new_r, 0, self.motion_model, self.meas_model,
                                        self.ps, self.pd1, self.intensity_c)
                    hypo_table_u[i][k] = deepcopy(new_obj)
                    like_table_u[i, k] = np.log(exp_weight)

                elif i != 0 and k == 0:
                    if not undetected_objs_0:
                        continue
                    x, P, rho = self.componentmatching(undetected_objs_1)
                    exp_weight = self.intensity_c + rho
                    new_r = rho / exp_weight
                    new_obj = Bernoulli(x, P, new_r, 0, self.motion_model, self.meas_model,
                                        self.ps, self.pd1, self.intensity_c)
                    hypo_table_u[i][k] = deepcopy(new_obj)
                    like_table_u[i, k] = np.log(exp_weight)

                elif i != 0 and k != 0:
                    if not undetected_objs_01:
                        continue
                    x, P, rho = self.componentmatching(undetected_objs_01)
                    new_obj = Bernoulli(x, P, 1, 0, self.motion_model, self.meas_model,
                                        self.ps, self.pd1, self.intensity_c)
                    hypo_table_u[i][k] = deepcopy(new_obj)
                    like_table_u[i, k] = np.log(rho)

        # permutation all meas2
        used_z2_d_and_u = [used_z2_u[k] for k in range(m2) if used_z2_d[k]]
        all_probably_permutation = []
        meas_index_list = list(range(m2_d))
        self.permutation(meas_index_list, used_z2_d_and_u, 0, [], all_probably_permutation)
        for single_z2 in all_probably_permutation:
            pass

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

        return extract

    @staticmethod
    def calculate_gate(obj, gate_size, z):
        S = np.dot(np.dot(obj.H, obj.P), obj.H.T) + obj.R
        S = (S + S.T) / 2
        delta = z - np.dot(obj.H, obj.x)
        Mdistance = np.dot(np.dot(delta.T, np.linalg.inv(S)), delta)
        z_in_gate = Mdistance < gate_size

        return z_in_gate

    @staticmethod
    def permutation(meas_list, used_d_and_u, k, path, result):
        if not meas_list:
            return
        result.append(path)
        for i in range(k, len(meas_list)):
            if not used_d_and_u[i]:
                continue
            PMBMfilter.permutation(meas_list, used_d_and_u, i + 1, path + [meas_list[i]], result)

    @staticmethod
    def combination(confirm_matrix):
        return confirm_matrix

    @staticmethod
    def componentmatching(poisson_obj_set):
        w = 0
        x = 0
        for obj in poisson_obj_set:
            if not obj:
                continue
            w += obj.w
            x += obj.w * obj.x

        x = x / w
        P = np.zeros(shape=poisson_obj_set[0].P.shape)

        for obj in poisson_obj_set:
            P += obj.w * (obj.P + np.dot((obj.x - x), (obj.x - x).T))

        P = P / w
        return x, P, w
