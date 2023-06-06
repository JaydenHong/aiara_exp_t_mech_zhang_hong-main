import gymnasium as gym
import numpy as np
from commons.commons import nm, nm2, calculate_reward_dmp_with_obstacle
import commons.parameters as pr
import copy

Ts = pr.SAMPLING_TIME
alpha = pr.DMP_ALPHA
beta = pr.DMP_BETA
gamma = pr.RL_GAMMA
omega = pr.DMP_OMEGA
tau = pr.DMP_TAU
DoF = pr.DOF


class DMPEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, seed=0):

        # self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.act_dim = (DoF, )
        self.init_position = None
        self.goal_position = None
        self.obst_position = None

        # Configure DMP simulation
        self.x = None
        self.xd = None
        self.xdd = None
        self.s = None
        self.counter = None

        options = {'init_pos': np.array(pr.INIT_POS_FIXED).astype(np.float32),
                   'goal_pos': np.array(pr.GOAL_POS_FIXED).astype(np.float32),
                   'obst_pos': np.array(pr.OBST_POS_FIXED).astype(np.float32)}

        obs, _ = self.reset(options=options)
        self.obs_dim = np.shape(obs)
        self.observation_space = gym.spaces.box.Box(-np.inf, np.inf, self.obs_dim, dtype=np.float32, seed=seed)
        self.action_space = gym.spaces.box.Box(-5, 5, self.act_dim, dtype=np.float32, seed=seed)

    def step(self, action):

        #si = 1 - 1 / (1 + math.exp(-(0.2 * (self.counter - 50))))
        #si *= 2.5

        self.xdd = tau * alpha * (beta * (self.goal_position - self.x) - self.xd) + self.s * nm(
                    self.goal_position - self.init_position) * action.astype(np.float32)
        self.s += - tau * Ts * omega * self.s #+ tau * Ts * omega * si
        self.x += Ts * self.xd
        self.xd += Ts * self.xdd
        self.counter += 1

        reward, terminated, truncated = calculate_reward_dmp_with_obstacle(self.counter, self.x, self.xdd,
                                                                           self.goal_position, self.obst_position)
        obs = np.append(self.x, self.xd)
        obs = np.append(obs, self.x - self.obst_position)
        obs = np.append(obs, self.s)

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):

        self.init_position = options['init_pos'].astype(np.float32)
        self.goal_position = options['goal_pos'].astype(np.float32)
        self.obst_position = options['obst_pos'].astype(np.float32)

        self.x = copy.copy(self.init_position)
        self.xd = np.zeros([DoF, ], dtype=np.float32)
        self.xdd = np.zeros([DoF, ], dtype=np.float32)
        self.s = np.array([1], dtype=np.float32)
        self.counter = 0
        obs = np.append(self.x, self.xd)
        obs = np.append(obs, self.x - self.obst_position)
        obs = np.append(obs, self.s)
        return obs, {}

    def render(self, mode='human'):
        return

    def close(self):
        return
