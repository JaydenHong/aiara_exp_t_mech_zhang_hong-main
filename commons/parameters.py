import numpy as np

SAMPLING_TIME = 0.2
TOTAL_TIME = 5     # 12 # 10
DOF = 3

DMP_ALPHA = 10      # 8 # 10
DMP_BETA = 1.2      # 0.6 # 2.5
DMP_OMEGA = 6       # 4 # 8 # 20
RL_GAMMA = 0.99
DMP_TAU = 0.25

ORIGINAL_DMP_BASIS_CENTERS = np.array([1, 0.6294, 0.3962, 0.2494, 0.1569,
                                       0.0988, 0.0622, 0.0391, 0.0246, 0.0155], dtype=np.float32)
ORIGINAL_DMP_BASIS_VARIANCE = np.array([41.6667, 16.5359, 6.5359, 2.5840, 1.0235,
                                        0.4054, 0.1606, 0.0636, 0.0252, 0.0252], dtype=np.float32) / 100
ORIGINAL_DMP_DIM = 10

COLLISION_TOLERANCE = 0.05
COLLISION_GAIN = 0.001
COLLISION_FIELD_RANGE = 0.08
OBST_RADIUS = 0.12

INITIAL_ANGULAR_POSITION_IN_RADIUS = [0, 0.25, -2.38, 0, -0.55, 1.57]
GOAL_ANGULAR_POSITION_IN_RADIUS = [0, 1.0, -0.68, 0, -1.5, 1.57]
OBSTACLE_STATIC_POSITION = [0.5, 0.1, 0.176]

region_x = 0.6
region_y = 0.5
region_z = 0.4

INIT_POS_FIXED = [0, 0, 0.05]
GOAL_POS_FIXED = [0.30, 0.35, 0.08]
OBST_POS_FIXED = [0.14, 0.16, 0.16]

HUMAN_DEMO_DIR = '/home/zengjie/Documents/dataset_2022_icra_human_dmp_zhang/2. obstacle avoidance_Aug22_2022/'
SAVE_DATA_DIR = '/home/zengjie/Documents/data/'
