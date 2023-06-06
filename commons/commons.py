from numpy.linalg import norm
import numpy as np
from enum import Enum
import math
import commons.parameters as pr
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class SimulationMode(Enum):
    ORIGINAL_DMP = 0
    ONLY_ROBOT = 1
    DMP_WITH_ROBOT = 2


def nm(vec):
    return norm(vec, ord=np.inf)


def nm2(vec):
    return norm(vec, ord=2)


def radius_to_degree(vec):
    out = np.zeros([len(vec)])
    for i in range(len(out)):
        out[i] = (vec[i] + 6.28) if vec[i] < 0 else vec[i]
    return out/3.14*180


def calculate_dmp_basis(s):
    weight = np.zeros([len(pr.ORIGINAL_DMP_BASIS_CENTERS), ])
    for j in range(len(pr.ORIGINAL_DMP_BASIS_CENTERS)):
        weight[j] = math.exp(-(s - pr.ORIGINAL_DMP_BASIS_CENTERS[j]) ** 2 / (2 * (pr.ORIGINAL_DMP_BASIS_VARIANCE[j])))
    basis = weight / sum(weight)
    return weight, basis


def push(obj):
    if torch.is_tensor(obj):
        if not obj.is_cuda:
            obj = obj.to(device)
    else:
        print("Given parameter is not a tensor!")
    return obj


def pull(obj):
    if torch.is_tensor(obj):
        if obj.is_cuda:
            obj = obj.cpu()
    else:
        print("Given parameter is not a tensor!")
    return obj


def dist_to_obstacle(x_to_obst_vec):
    if x_to_obst_vec[2] > 0:
        dist_to_obst = nm2(x_to_obst_vec)
    else:
        dist_to_obst = nm2(x_to_obst_vec[:2])
    return dist_to_obst


def potential_field(dist_to_goal, field_range, dist_tolerance):

    if dist_to_goal <= dist_tolerance:
        value = 100000
    elif (dist_to_goal > dist_tolerance) and (dist_to_goal < field_range):
        value = 1/(dist_to_goal-dist_tolerance)**2 - 1/(field_range-dist_tolerance)**2
    else:
        value = 0
    return value


def calculate_reward_dmp_with_obstacle(counter, pos, acc, goal_pos, obst_pos):
    dist_to_goal = nm2(pos - goal_pos)

    # if dist_to_obstacle < 0.02 + pr.OBST_RADIUS:
    #    done = True
    #    reward = -1000000
    if counter >= int(pr.TOTAL_TIME/pr.SAMPLING_TIME):
        truncated = True
        terminated = False
        if dist_to_goal > 0.01:
            reward = - (dist_to_goal - 0.01) ** 2 * 100000
        else:
            reward = 0
    else:
        truncated = False
        if dist_to_goal > 0.01:
            terminated = False
            reward = - nm2(acc) ** 2 * 0.001 - dist_to_goal ** 2 * 10 \
                     - pr.COLLISION_GAIN * potential_field(dist_to_obstacle(pos - obst_pos),
                                                           field_range=pr.COLLISION_FIELD_RANGE,
                                                           dist_tolerance=pr.OBST_RADIUS + pr.COLLISION_TOLERANCE) \
                     - 0.001 * potential_field(pos[2] + 0.05, field_range=0.05, dist_tolerance=0.01)
        else:
            terminated = True
            reward = 0
    return reward, terminated, truncated
