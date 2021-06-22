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
        self.pd1 = pd
        self.pd2 = pd
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

    def update(self, z):
        print("-----------update---------")
        gate_size = chi2.ppf(self.gate_rate, 4)  # gate size use X^2 distribution
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
        if nd != 0:
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
            like_table = - np.ones(shape=(m1_d + 1, nd, m2_d + 1), dtype=float) * MAX_VALUE
            for i in range(m1_d + 1):
                for j in range(nd):
                    for k in range(m2_d + 1):
                        if i == 0 and k == 0:  # undectected object
                            obj = deepcopy(self.detected_objects[j])
                            like_table[i, j, k] = obj.undetected_bern_update()
                            hypo_table[i][j][k] = deepcopy(obj)
                        elif i == 0 and k != 0:
                            obj = deepcopy(self.detected_objects[j])
                            if not z2_in_gate_d[j][k - 1]:
                                continue
                            like_table[i, j, k] = obj.detected_bern_update([None, z2_d[k - 1]])
                            hypo_table[i][j][k] = deepcopy(obj)
                        elif i != 0 and k == 0:
                            obj = deepcopy(self.detected_objects[j])
                            if not z1_in_gate_d[j][i - 1]:
                                continue
                            like_table[i, j, k] = obj.detected_bern_update([z1_d[i - 1], None])
                            hypo_table[i][j][k] = deepcopy(obj)
                        else:
                            obj = deepcopy(self.detected_objects[j])
                            if not (z1_in_gate_d[j][i - 1] and z2_in_gate_d[j][k - 1]):
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
                            continue

                        elif i == 0 and k != 0:
                            if not z2_in_gate_d_and_u[j][k - 1]:
                                continue
                            obj = deepcopy(self.undetected_objects[j])
                            obj.detected_poisson_update([None, z2_d[k - 1]])
                            undetected_objs_1[j] = deepcopy(obj)

                        elif i != 0 and k == 0:
                            if not z1_in_gate_d_and_u[j][i - 1]:
                                continue
                            obj = deepcopy(self.undetected_objects[j])
                            obj.detected_poisson_update([z1_d[i - 1], None])
                            undetected_objs_0[j] = deepcopy(obj)

                        else:
                            if not (z1_in_gate_d_and_u[j][i - 1] and z2_in_gate_d_and_u[j][k - 1]):
                                continue
                            obj = deepcopy(self.undetected_objects[j])
                            obj.detected_poisson_update([z1_d[i - 1], z2_d[k - 1]])
                            undetected_objs_01[j] = deepcopy(obj)

                    if i == 0 and k != 0:
                        if not any(undetected_objs_1):
                            continue
                        x, P, rho = self.componentmatching(undetected_objs_1)
                        exp_weight = 1 + rho
                        new_r = rho / exp_weight
                        new_obj = Bernoulli(x, P, new_r, 0, self.motion_model, self.meas_model,
                                            self.ps, self.pd1, self.intensity_c)
                        hypo_table_u[i][k] = deepcopy(new_obj)
                        like_table_u[i, k] = np.log(exp_weight)

                    elif i != 0 and k == 0:
                        if not any(undetected_objs_0):
                            continue
                        x, P, rho = self.componentmatching(undetected_objs_0)
                        exp_weight = self.intensity_c + rho
                        new_r = rho / exp_weight
                        new_obj = Bernoulli(x, P, new_r, 0, self.motion_model, self.meas_model,
                                            self.ps, self.pd1, self.intensity_c)
                        hypo_table_u[i][k] = deepcopy(new_obj)
                        like_table_u[i, k] = np.log(exp_weight)

                    elif i != 0 and k != 0:
                        if not any(undetected_objs_01):
                            continue
                        x, P, rho = self.componentmatching(undetected_objs_01)
                        new_obj = Bernoulli(x, P, 1, 0, self.motion_model, self.meas_model,
                                            self.ps, self.pd1, self.intensity_c)
                        hypo_table_u[i][k] = deepcopy(new_obj)
                        like_table_u[i, k] = np.log(rho)

            # permutation all meas2
            used_z1_d_and_u = [used_z1_u[i] for i in range(m1) if used_z1_d[i]]
            used_z2_d_and_u = [used_z2_u[k] for k in range(m2) if used_z2_d[k]]
            all_probably_permutation = []
            meas_index_list = list(range(m2_d))
            self.permutation(meas_index_list, used_z2_d_and_u, 0, [], all_probably_permutation)

            weight_list = []
            assign_list = []
            for single_z2 in all_probably_permutation:
                init_weight = 0
                res_meas_idx_list = [idx for idx in meas_index_list if idx not in single_z2]
                res_cz = np.log(self.intensity_c) * len(res_meas_idx_list)
                for idx in single_z2:
                    init_weight += like_table_u[0, idx + 1]

                """
                get all possible combination hypothesis:
                1. use murty to get all combinations of meas1 and objeccts
                2. use munkres to get best combinations of objects and meas2
                """
                # construct confirm matrix
                C1 = np.ones(shape=(m1_d, nd))
                C2 = np.ones(shape=(m1_d, m1_d)) * MAX_VALUE
                for i in range(m1_d):
                    for j in range(nd):
                        if not z1_in_gate_d[j][i]:
                            C1[i, j] = MAX_VALUE
                    C2[i, i] = 1
                C = np.concatenate((C1, C2), axis=1)
                all_probably_combination = self.combination(C)
                # for each combination, construct loss matrix
                max_log_weight = -MAX_VALUE
                max_assign = []
                for row, col in all_probably_combination:
                    new_obj_meas_list = [i for i in list(col) if i >= nd]  # save new obj2meas1 index
                    num_new_objs = np.sum(col >= nd)
                    L1 = np.ones((nd + num_new_objs, len(res_meas_idx_list)), dtype=float) * -MAX_VALUE
                    L2 = np.ones((nd + num_new_objs, nd + num_new_objs), dtype=float) * -MAX_VALUE
                    for r in range(nd + num_new_objs):
                        for c in range(len(res_meas_idx_list)):
                            if r < nd and (r not in col):
                                idx_2 = res_meas_idx_list[c]
                                if z2_in_gate_d[r][idx_2]:
                                    L1[r, c] = like_table[0, r, idx_2 + 1] - np.log(self.intensity_c)
                            elif r < nd and (r in col):
                                idx_2 = res_meas_idx_list[c]
                                idx_1 = list(col).index(r)
                                if z2_in_gate_d[r][idx_2]:
                                    L1[r, c] = like_table[idx_1 + 1, r, idx_2 + 1] - np.log(self.intensity_c)
                            elif r >= nd:
                                idx_2 = res_meas_idx_list[c]
                                idx_1 = list(col).index(new_obj_meas_list[r - nd])
                                if used_z2_d_and_u[idx_2]:
                                    L1[r, c] = like_table_u[idx_1 + 1, idx_2 + 1] - np.log(self.intensity_c)
                        if r < nd and (r not in col):
                            L2[r, r] = like_table[0, r, 0]
                        elif r < nd and (r in col):
                            idx_1 = list(col).index(r)
                            L2[r, r] = like_table[idx_1 + 1, r, 0]
                        elif r >= nd:
                            idx_1 = list(col).index(new_obj_meas_list[r - nd])
                            L2[r, r] = like_table_u[idx_1 + 1, 0]
                    L = -np.concatenate((L1, L2), axis=1)
                    row_, col_ = linear_assignment(L)
                    log_weight = - L[row_, col_].sum() + res_cz
                    if log_weight > max_log_weight:
                        max_log_weight = log_weight
                        best_assign = []
                        for i in range(nd + num_new_objs):
                            if i < nd and (i not in col):
                                meas_idx_1 = -1
                                obj_idx = i
                                if col_[i] < len(res_meas_idx_list):
                                    meas_idx_2 = res_meas_idx_list[col_[i]]
                                else:
                                    meas_idx_2 = -1
                            elif i < nd and (i in col):
                                meas_idx_1 = list(col).index(i)
                                obj_idx = i
                                if col_[i] < len(res_meas_idx_list):
                                    meas_idx_2 = res_meas_idx_list[col_[i]]
                                else:
                                    meas_idx_2 = -1
                            else:
                                meas_idx_1 = list(col).index(new_obj_meas_list[i - nd])
                                obj_idx = -1
                                if col_[i] < len(res_meas_idx_list):
                                    meas_idx_2 = res_meas_idx_list[col_[i]]
                                else:
                                    meas_idx_2 = -1
                            best_assign.append((meas_idx_1, obj_idx, meas_idx_2))
                        max_assign = best_assign

                weight_list.append(max_log_weight + init_weight)
                assign_list.append(max_assign)

            max_weight = max(weight_list)
            index = weight_list.index(max_weight)
            final_assign = assign_list[index]
            self.detected_objects.clear()
            for meas_idx_1, obj_idx, meas_idx_2 in final_assign:
                if obj_idx >= 0:
                    new_obj = hypo_table[meas_idx_1 + 1][obj_idx][meas_idx_2 + 1]
                    self.detected_objects.append(deepcopy(new_obj))
                else:
                    new_obj = hypo_table_u[meas_idx_1 + 1][meas_idx_2 + 1]
                    if not new_obj:
                        continue
                    new_obj.id = self.max_id
                    self.max_id += 1
                    self.detected_objects.append(deepcopy(new_obj))

            best_permutation = all_probably_permutation[index]
            for meas_idx_2 in best_permutation:
                new_obj = hypo_table_u[0][meas_idx_2 + 1]
                if not new_obj:
                    continue
                new_obj.id = self.max_id
                self.max_id += 1
                self.detected_objects.append(deepcopy(new_obj))

        # undected meas update
        used_z1_u_not_d = [used_z1_u[j] and not used_z1_d[j] for j in range(m1)]
        used_z2_u_not_d = [used_z2_u[j] and not used_z2_d[j] for j in range(m2)]
        z1_u = [z1[i] for i in range(m1) if used_z1_u_not_d[i]]
        z2_u = [z2[i] for i in range(m2) if used_z2_u_not_d[i]]
        z1_in_gate_u_not_d = [[z1_in_gate_u[i][j] for j in range(m1) if used_z1_u_not_d[j]] for i in range(nu)]
        z2_in_gate_u_not_d = [[z2_in_gate_u[i][j] for j in range(m2) if used_z2_u_not_d[j]] for i in range(nu)]
        m1_u = sum(used_z1_u_not_d)
        m2_u = sum(used_z2_u_not_d)
        hypo_table_u_not_d = [[None] * (m2_u + 1) for _ in range(m1_u + 1)]
        like_table_u_not_d = np.ones((m1_u + 1, m2_u + 1)) * np.log(self.intensity_c)
        for i in range(m1_u + 1):
            for k in range(m2_u + 1):
                undetected_objs_0 = [None] * nu
                undetected_objs_1 = [None] * nu
                undetected_objs_01 = [None] * nu
                for j in range(nu):
                    if i == 0 and k == 0:
                        obj = deepcopy(self.undetected_objects[j])
                        obj.undetected_poisson_update()
                        hypo_table_u_not_d[i][k] = deepcopy(obj)

                    elif i == 0 and k != 0:
                        if not z2_in_gate_u_not_d[j][k - 1]:
                            continue
                        obj = deepcopy(self.undetected_objects[j])
                        obj.detected_poisson_update([None, z2_u[k - 1]])
                        undetected_objs_1[j] = deepcopy(obj)

                    elif i != 0 and k == 0:
                        if not z1_in_gate_u_not_d[j][i - 1]:
                            continue
                        obj = deepcopy(self.undetected_objects[j])
                        obj.detected_poisson_update([z1_u[i - 1], None])
                        undetected_objs_0[j] = deepcopy(obj)

                    else:
                        if not (z1_in_gate_u_not_d[j][i - 1] and z2_in_gate_u_not_d[j][k - 1]):
                            continue
                        obj = deepcopy(self.undetected_objects[j])
                        obj.detected_poisson_update([z1_u[i - 1], z2_u[k - 1]])
                        undetected_objs_01[j] = deepcopy(obj)

                if i == 0 and k != 0:
                    if not any(undetected_objs_1):
                        continue
                    x, P, rho = self.componentmatching(undetected_objs_1)
                    exp_weight = 1 + rho
                    new_r = rho / exp_weight
                    new_obj = Bernoulli(x, P, new_r, 0, self.motion_model, self.meas_model,
                                        self.ps, self.pd1, self.intensity_c)
                    hypo_table_u_not_d[i][k] = deepcopy(new_obj)
                    like_table_u_not_d[i, k] = np.log(exp_weight)

                elif i != 0 and k == 0:
                    if not any(undetected_objs_0):
                        continue
                    x, P, rho = self.componentmatching(undetected_objs_0)
                    exp_weight = self.intensity_c + rho
                    new_r = rho / exp_weight
                    new_obj = Bernoulli(x, P, new_r, 0, self.motion_model, self.meas_model,
                                        self.ps, self.pd1, self.intensity_c)
                    hypo_table_u_not_d[i][k] = deepcopy(new_obj)
                    like_table_u_not_d[i, k] = np.log(exp_weight)

                elif i != 0 and k != 0:
                    if not any(undetected_objs_01):
                        continue
                    x, P, rho = self.componentmatching(undetected_objs_01)
                    new_obj = Bernoulli(x, P, 1, 0, self.motion_model, self.meas_model,
                                        self.ps, self.pd1, self.intensity_c)
                    hypo_table_u_not_d[i][k] = deepcopy(new_obj)
                    like_table_u_not_d[i, k] = np.log(rho)

        # permutation all meas2_u_not_d
        meas_index_list_u = list(range(m2_u))
        all_probably_permutation_u = []
        self.permutation(meas_index_list_u, [True] * m2_u, 0, [], all_probably_permutation_u)

        weight_list_u = []
        assign_list_u = []
        for single_z2 in all_probably_permutation_u:
            init_weight = 0
            res_meas_idx_list = [idx for idx in meas_index_list_u if idx not in single_z2]
            res_cz = np.log(self.intensity_c) * len(res_meas_idx_list)
            for idx in single_z2:
                init_weight += like_table_u_not_d[0, idx + 1]
            """
            use munkres to create new objects
            """
            L1 = np.ones((m1_u, len(res_meas_idx_list))) * -MAX_VALUE
            L2 = np.ones((m1_u, m1_u)) * -MAX_VALUE
            for i in range(m1_u):
                for k in range(len(res_meas_idx_list)):
                    idx_2 = res_meas_idx_list[k]
                    L1[i, k] = like_table_u_not_d[i + 1, idx_2 + 1] - np.log(self.intensity_c)
                L2[i, i] = like_table_u_not_d[i + 1, 0]
            L = - np.concatenate((L1, L2), axis=1)
            row_u, col_u = linear_assignment(L)
            best_weight = - -L[row_u, col_u].sum() + res_cz
            weight_list_u.append(best_weight + init_weight)
            best_assign_u = []
            for i in range(m1_u):
                if col_u[i] < len(res_meas_idx_list):
                    meas_idx_2_u = res_meas_idx_list[col_u[i]]
                    best_assign_u.append((i, meas_idx_2_u))
                else:
                    best_assign_u.append((i, -1))
            assign_list_u.append(best_assign_u)

        max_weight_u = max(weight_list_u)
        index_u = weight_list_u.index(max_weight_u)
        final_assign_u = assign_list_u[index_u]
        for meas_idx_1, meas_idx_2 in final_assign_u:
            new_obj = hypo_table_u_not_d[meas_idx_1 + 1][meas_idx_2 + 1]
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
    def permutation(meas_list, used_d_and_u, k, path, result):
        result.append(path)
        if not meas_list:
            return
        for i in range(k, len(meas_list)):
            if not used_d_and_u[i]:
                continue
            PMBMfilter.permutation(meas_list, used_d_and_u, i + 1, path + [meas_list[i]], result)

    @staticmethod
    def combination(confirm_matrix):
        # murty algorithm
        row, col = linear_assignment(confirm_matrix)
        cost = confirm_matrix[row, col].sum()
        best_assign = []
        best_costs = []
        node_list = [confirm_matrix]
        cost_list = [cost]
        assign_list = [(row, col)]
        while cost_list:
            min_cost = min(cost_list)
            best_costs.append(min_cost)
            min_index = cost_list.index(min_cost)
            cost_list.pop(min_index)
            assignment = assign_list.pop(min_index)
            best_assign.append(assignment)
            pre_matrix = node_list.pop(min_index)

            for i in range(confirm_matrix.shape[0]):
                node = deepcopy(pre_matrix)
                m, n = assignment[0][i], assignment[1][i]
                node[m, n] = MAX_VALUE
                row_, col_ = linear_assignment(node)
                cost_ = node[row_, col_].sum()
                if cost_ < MAX_VALUE:
                    node_list.append(node)
                    assign_list.append((row_, col_))
                    cost_list.append(cost_)

                cur_value = pre_matrix[m, n]
                pre_matrix[m, :] = MAX_VALUE
                pre_matrix[:, n] = MAX_VALUE
                pre_matrix[m, n] = cur_value

        return best_assign

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
