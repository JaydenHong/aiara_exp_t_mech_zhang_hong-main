import torch
import dmp_sim_obst
import numpy as np
import gymnasium as gym
import commons.parameters as pr
from commons.logx import EpochLogger
import draw_trajectory

# Calculate the maximum steps of one trajectory. You can make the TOTAL_TIME even longer.
# The step size is 0.02s. If Gen3 takes a longer time for one step, the motion will be slow.
# You can adjust the motion speed by skipping some set points.
max_ep_len = int(pr.TOTAL_TIME/pr.SAMPLING_TIME)

# This 'option' variable is a dictionary that contains the initial position, goal position, and obstacle position.
# options = {'init_pos': np.array(pr.init_pos, dtype=np.float32),
#            'goal_pos': np.array(pr.goal_pos, dtype=np.float32) + np.array([0.02, -0.01, 0.01], dtype=np.float32),
#            'obst_pos': np.array(pr.obst_pos, dtype=np.float32) + np.array([-0.01, -0.02, 0], dtype=np.float32)}


def execute(init_pos, goal_pos, obst_pos, file_name='progress.txt', file_path='saved_data'):

    options = {'init_pos': np.array(init_pos, dtype=np.float32),
               'goal_pos': np.array(goal_pos, dtype=np.float32),
               'obst_pos': np.array(obst_pos, dtype=np.float32)}
    print("options", options)
    # Render the dmp environment
    # If you do not like to use gymnasium, you can also rap-up the dmp model as a normal function
    env = gym.make('dmp-v0')
    
    # I use the spinning-up logger to record the data in 'saved_data/progress.txt'.
    # You can use a different manner to log the data.

    logger = EpochLogger(output_dir=file_path, output_fname=file_name)
    
    # Load the trained policy
    # ac = torch.load('best_policy.pt',map_location ='cpu')
    ac = torch.load('best_policy.pt', map_location='cuda')
    
    obs, _ = env.reset(options=options)
    ter, tru, ep_ret, ep_len = 0, 0, 0, 0
    while not (ter or tru or (max_ep_len == ep_len)):
        action = ac.act(torch.tensor(obs, dtype=torch.float32))
        obs, r, ter, tru, _ = env.step(action)
        ep_ret += r
        ep_len += 1
        logger.log_tabular('Time', ep_len*pr.SAMPLING_TIME)
        for i in range(3):
            # option['init_pos'] is the 3-dim initial position
            logger.log_tabular('InitPos_' + str(i + 1), options['init_pos'][i])
            # option['goal_pos'] is the 3-dim goal position
            logger.log_tabular('GoalPos_' + str(i + 1), options['goal_pos'][i])
            # option['obst_pos'] is the 3-dim obstacle position
            logger.log_tabular('ObstPos_' + str(i + 1), options['obst_pos'][i])
            # obs[0:3] is the 3-dim position of the end-effector
            logger.log_tabular('Pos_' + str(i + 1), obs[i])
            # obs[3:6] is the 3-dim linear velocity of the end-effector
            logger.log_tabular('Vel_' + str(i + 1), obs[i+3])
        # ep_ret is the total reward of this trajectory
        logger.log_tabular('TestEpRet', ep_ret)
        logger.dump_tabular()


if __name__ == '__main__':
    execute(pr.INIT_POS_FIXED, pr.GOAL_POS_FIXED, pr.OBST_POS_FIXED)
