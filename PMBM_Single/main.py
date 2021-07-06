import numpy as np
from matplotlib import pyplot as plt
import argparse
import sys
import os
import logging
from tqdm import tqdm
from copy import deepcopy
from PMBM import PMBMfilter
from Poisson import Poisson

ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)

from utils.Parameter_set import CVMotionModel, DisMeasureModel
from utils import Plot

COLORMAP = ['red', 'yellow', 'green', 'blue', 'purple', 'black', 'magenta', 'cyan']


def generate_one_traj(traj_init, Motion_model, T, noise=False):
    """
    traj_init: a initial dict of one trajectory
    Motion_model: motion model
    noise: if the model is accurate
    return: a dict of trajectory including state and start time
    """
    start_time = traj_init['start time']
    end_time = traj_init['end time']
    state = []
    cur_state = traj_init['start state']
    for i in range(int((end_time - start_time) / T)):
        state.append(cur_state)
        cur_state = np.dot(Motion_model.F, cur_state)
        if noise:
            cur_state = np.random.multivariate_normal(cur_state, Motion_model.Q)

    traj_complete = {'state': state, 'start time': start_time, 'end time': end_time}

    return traj_complete


def generate_measurement(traj, num_sensors, pd, Meas_Model):
    """
    traj: trajectory point
    num_sensors: number of sensors
    pd: probability of detection
    Meas_Model: measurement model
    return: measurement list
    """
    meas = [[] for _ in range(num_sensors)]
    xy = np.dot(Meas_Model.H, traj)
    for i in range(num_sensors):
        if np.random.rand() < pd:
            z = np.random.multivariate_normal(xy, Meas_Model.R)
            meas[i].append(z)
    return meas


def main(traj: dict, meas: list):
    """
    traj: the dict of all trajectory, element[idx] is also a dict,
    including start time, end time and state list

    meas: the list of measurement of each time
    each traj_point: np.array(4, )
    each meas: np.array(2, )
    """
    # parameter initial
    min_start_time = sys.maxsize
    max_end_time = 0
    for idx in range(traj['num']):
        traj_idx = traj[idx]
        start_time = traj_idx['start time']
        end_time = traj_idx['end time']
        min_start_time = start_time if start_time < min_start_time else min_start_time
        max_end_time = end_time if end_time > max_end_time else max_end_time

    T = (max_end_time - min_start_time) / len(meas)
    ps = 0.9  # survival probability
    pd = 0.8  # detection probability
    intensity_c = 1e-6  # clutter intensity

    # initial model, including motion model,meas model,birth model
    motion_model = CVMotionModel(T)
    meas_model = DisMeasureModel()
    birth_state = np.array([[0, 100, 100, 0],
                            [0, 100, 0, 100],
                            [5, -5, -5, 5],
                            [5, -5, 5, -5]])
    birth_cov = 1 * np.identity(4)
    birth_weight = [0.5, 0.5, 0.5, 0.5]
    num_birth = birth_state.shape[1]
    birth_model = []
    for i in range(num_birth):
        new_poisson = Poisson(x=birth_state[:, i], P=birth_cov, w=np.log(birth_weight[i]),
                              motion_model=motion_model, meas_model=meas_model, ps=ps, pd=pd)
        birth_model.append(deepcopy(new_poisson))

    pmbm = PMBMfilter(motion_model, meas_model, birth_model,
                      ps, pd, intensity_c)
    prune_hypo, prune_poisson, prune_bern = 1e-4, 1e-4, 1e-4

    total_time = len(meas)
    tracking = [[] for _ in range(total_time)]
    number = [0 for _ in range(total_time)]
    for t in tqdm(range(total_time)):
        pmbm.predict()
        pmbm.update(z=meas_list[t])
        pmbm.prune(prune_hypo, prune_bern, prune_poisson)
        number[t], tracking[t] = pmbm.estimate(0.5)

    # from IPython import embed
    # embed()
    fig1 = plt.figure()
    # plt.ion()
    for t in range(total_time):
        for idx in range(traj['num']):
            start_time = traj_dict[idx]['start time']
            end_time = traj_dict[idx]['end time']
            if start_time <= t * T < end_time:
                state = traj_dict[idx]['state'][t - int(start_time / T)]
                plt.scatter(x=state[0], y=state[1], c=COLORMAP[idx % 8], s=3)

        for meas_i in meas_list[t]:
            for meas in meas_i:
                plt.scatter(x=meas[0], y=meas[1], c='black', s=2, marker='*')

        for obj in tracking[t]:
            idx = obj.id
            state = obj.x
            plt.scatter(x=state[0], y=state[1], c=COLORMAP[idx % 8], s=7, marker='s')

    plt.xlabel(xlabel='x position/m')
    plt.ylabel(ylabel='y position/m')
    plt.show()

    fig2 = plt.figure()
    real_number = []
    for t in range(total_time):
        num_cur = 0
        for idx in range(traj['num']):
            start_time = traj_dict[idx]['start time']
            end_time = traj_dict[idx]['end time']
            if start_time <= t * T < end_time:
                num_cur += 1
        real_number.append(num_cur)

        plt.scatter(x=t, y=number[t], s=5, marker='o', c='red')
    plt.plot(list(range(total_time)), real_number)
    plt.xlabel(xlabel='time')
    plt.ylabel(ylabel='number')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='self', help='where the input from')
    parser.add_argument("--output-dir", type=str, help='output dictionary')
    args = parser.parse_args()
    np.random.seed(157)

    traj_dict = {}
    meas_list = []
    if args.input == 'self':
        T = 0.5
        motion_model = CVMotionModel(T, sigma=0.5)
        meas_model = DisMeasureModel(sigma=1.5)
        init_traj = [{'start time': 0, 'end time': 100, 'start state': np.array([0, 0, 6, 8]).T},
                     {'start time': 0, 'end time': 100, 'start state': np.array([100, 100, -6, -8]).T},
                     {'start time': 5, 'end time': 100, 'start state': np.array([1, 0, 5, 5]).T},
                     {'start time': 10, 'end time': 75, 'start state': np.array([101, 99, -8, -6]).T},
                     {'start time': 0, 'end time': 100, 'start state': np.array([101, 0, -6, 5]).T}]
        # Data generation
        num_traj = len(init_traj)
        for idx in range(num_traj):
            traj_idx = init_traj[idx]
            traj_dict[idx] = generate_one_traj(traj_idx, motion_model, T, noise=True)
        traj_dict['num'] = num_traj

        # Measurement generation
        max_end_time = 100
        pd = 0.8
        num_sensors = 2
        num_clutter = 15
        for t in range(0, int(max_end_time / T)):
            meas_t = [[] for _ in range(num_sensors)]
            for idx in range(num_traj):
                start_time = traj_dict[idx]['start time']
                end_time = traj_dict[idx]['end time']
                if start_time <= t * T < end_time:
                    state = traj_dict[idx]['state'][t - int(start_time / T)]
                    meas_t_idx = generate_measurement(state, num_sensors, pd, meas_model)
                    for i in range(num_sensors):
                        meas_t[i].extend(meas_t_idx[i])

            # create clutters
            for i in range(num_sensors):
                for j in range(num_clutter):
                    z_c = (np.random.rand(2) - 0.5) * 2000
                    meas_t[i].append(z_c)
            meas_list.append(meas_t)

    elif args.input == 'sumo':
        pass
    else:
        print("-----------Wrong parameters----------")
        exit(-1)

    main(traj_dict, meas_list)
