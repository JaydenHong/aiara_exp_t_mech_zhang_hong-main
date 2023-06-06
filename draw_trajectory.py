import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import commons.parameters as pr

option_dir = 'BC_32'


def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
    z = np.arange(0, height_z, 0.05)
    theta = np.arange(0, 2*np.pi, 0.05)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid


def plot_path(file_name='progress.txt'):

    fig = plt.figure(figsize=(3, 2.5))
    ax = plt.axes(projection='3d')

    for name in file_name:
        exp_data = pd.read_table('saved_data00/'+name)

        if 'Pos_1' in exp_data.columns:
            InitPos_x = exp_data.InitPos_1[0]
            InitPos_y = exp_data.InitPos_2[0]
            InitPos_z = exp_data.InitPos_3[0]
            GoalPos_x = exp_data.GoalPos_1[0]
            GoalPos_y = exp_data.GoalPos_2[0]
            GoalPos_z = exp_data.GoalPos_3[0]
            ObstPos_x = exp_data.ObstPos_1[0]
            ObstPos_y = exp_data.ObstPos_2[0]
            ObstPos_z = exp_data.ObstPos_3[0]

            CurrPos_x = exp_data.Pos_1.to_numpy()
            CurrPos_y = exp_data.Pos_2.to_numpy()
            CurrPos_z = exp_data.Pos_3.to_numpy()

            ax.plot3D(CurrPos_x, CurrPos_y, CurrPos_z, color=(21 / 255, 21 / 255, 81 / 255), linewidth=2, linestyle='--')
            ax.plot([InitPos_x], [InitPos_y], [InitPos_z], alpha=0.8, color=(0.2, 0.5, 1), marker="o", markersize=8)
            ax.plot([GoalPos_x], [GoalPos_y], [GoalPos_z], alpha=0.8, color=(128 / 255, 0, 127 / 255), marker="*",
                    markersize=10)
            Xc, Yc, Zc = data_for_cylinder_along_z(ObstPos_x, ObstPos_y, 0.045, ObstPos_z)
            ax.plot_surface(Xc, Yc, Zc, color=(255 / 255, 212 / 255, 128 / 255), alpha=0.2)

        elif 'x' in exp_data.columns:
            CurrPos_x = exp_data.x.to_numpy()
            CurrPos_y = exp_data.y.to_numpy()
            CurrPos_z = exp_data.z.to_numpy()

            ax.plot3D(CurrPos_x, CurrPos_y, CurrPos_z, color=(21/255, 21/255, 200/255), linewidth=1)

        # 3d range wrt cam #
        ax.set_xlim3d(-0.1, 0.9)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(0, 1)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    # plt.show()


if __name__ == '__main__':
    filenames = ['executed_trajectory1.txt', 'executed_trajectory2.txt', 'executed_trajectory3.txt']
    plot_path(filenames)
    # for i in range(1,6):
    #     plot_path(['computed_trajectory{}.txt'.format(i), 'executed_trajectory{}.txt'.format(i)])
    plt.show()

