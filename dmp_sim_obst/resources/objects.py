import pybullet as p
import os
import numpy as np
import math


class Goal:
    def __init__(self, client, base_position):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'simple_goal.urdf')
        self.id = self.client.loadURDF(fileName=f_name, basePosition=base_position)


class Obstacle:
    def __init__(self, client, base_position, velocity):
        self.client = client
        self.id = self.client.loadURDF(fileName=os.path.join(os.path.dirname(__file__), 'obstacle_cylinder.urdf'),
                                       basePosition=base_position, useFixedBase=1)
        self.client.resetBaseVelocity(self.id, linearVelocity=velocity)


class Plane:
    def __init__(self, client):
        self.client = client
        self.id = self.client.loadURDF(fileName=os.path.join(os.path.dirname(__file__), 'plane.urdf'),
                                       basePosition=[0, 0, 0])


GRIPPER_OPEN = 0.085
GRIPPER_CLOSE = 0.040
GRIPPER_RATIOS = [-1, 1, 1, -1, 1, -1]


def _open_angle(open_width):
    return 0.715 - math.asin((open_width - 0.010) / 0.1143)


class Robot:
    def __init__(self, client):
        self.client = client
        self.id = p.loadURDF(fileName=os.path.join(os.path.dirname(__file__), 'kinova_gen3.urdf'),
                             basePosition=[0, 0, 0],
                             useFixedBase=1,
                             flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

        self.links_all_dict = {p.getBodyInfo(self.id)[0].decode('UTF-8'): -1, }
        self.joints_all_dict = {}
        self.joints_all_info_list = []
        self.joints_controllable_list = []

        for i in range(p.getNumJoints(self.id)):
            info = self.client.getJointInfo(self.id, i)
            # Pair link name and joint id
            self.links_all_dict[info[12].decode('UTF-8')] = i
            # Pair joint name and its id
            self.joints_all_dict.update({info[1].decode("utf-8"): i})

            self.joints_all_info_list.append(dict({}))
            self.joints_all_info_list[i].update({'id': info[0]})
            self.joints_all_info_list[i].update({'name': info[1].decode("utf-8")})
            # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            self.joints_all_info_list[i].update({'type': info[2]})
            self.joints_all_info_list[i].update({'damping': info[6]})
            self.joints_all_info_list[i].update({'friction': info[7]})
            self.joints_all_info_list[i].update({'lower_limit': info[8]})
            self.joints_all_info_list[i].update({'upper_limit': info[9]})
            self.joints_all_info_list[i].update({'max_force': info[10]})
            self.joints_all_info_list[i].update({'max_velocity': info[11]})
            self.joints_all_info_list[i].update({'is_control': self.joints_all_info_list[i]['type'] != p.JOINT_FIXED})

            if self.joints_all_info_list[i]['is_control']:
                self.joints_controllable_list.append(i)

        self.robot_joints_list = ['Actuator1', 'Actuator2', 'Actuator3', 'Actuator4', 'Actuator5', 'Actuator6']
        self.robot_joints_controllable_id_list = [self.joints_all_dict[i] for i in self.robot_joints_list
                                                  if self.joints_all_info_list[self.joints_all_dict[i]]['is_control']]

        self.dof = len(self.robot_joints_controllable_id_list)
        self.end_effector_id = self.links_all_dict['EndEffector_Link']

        self.gripper_joints_list = ['left_inner_finger_joint', 'right_inner_finger_joint',
                                    'left_outer_knuckle_joint', 'right_outer_knuckle_joint',
                                    'left_inner_knuckle_joint', 'right_inner_knuckle_joint']
        self.gripper_joints_id_list = [self.joints_all_dict[i] for i in self.gripper_joints_list]

        self.reset_robot(np.zeros([self.dof, ]))

    def reset_robot(self, angles):
        self.reset_gripper()
        for i in range(self.dof):
            p.resetJointState(self.id, self.robot_joints_controllable_id_list[i], angles[i])

        p.setJointMotorControlArray(
            bodyUniqueId=self.id,
            jointIndices=self.robot_joints_controllable_id_list,
            controlMode=p.VELOCITY_CONTROL,
            forces=np.zeros([self.dof, ]))

    def set_joint_angles(self, angles):
        p.setJointMotorControlArray(
            bodyUniqueId=self.id,
            jointIndices=self.robot_joints_controllable_id_list,
            controlMode=p.POSITION_CONTROL,
            targetPositions=angles)

    def get_joint_state(self):
        angles = np.zeros([self.dof, ])
        velocities = np.zeros([self.dof, ])
        for i in np.arange(0, self.dof):
            angles[i], velocities[i], _, _ = p.getJointState(self.id, self.robot_joints_controllable_id_list[i])
        return angles, velocities

    def get_end_effector_pose(self):
        position = np.array(p.getLinkState(self.id, linkIndex=self.end_effector_id, computeForwardKinematics=1)[0])
        quaternion = np.array(p.getLinkState(self.id, linkIndex=self.end_effector_id, computeForwardKinematics=1)[1])
        return position, quaternion

    def get_end_effector_linear_velocity(self):
        linear_velocity = np.array(p.getLinkState(self.id, linkIndex=self.end_effector_id, computeLinkVelocity=1)[6])
        return linear_velocity

    def get_tool_pose(self):
        local_frame_position = p.getLinkState(self.id, linkIndex=self.end_effector_id, computeForwardKinematics=1)[4]
        local_frame_quaternion = p.getLinkState(self.id, linkIndex=self.end_effector_id, computeForwardKinematics=1)[5]
        tool_offset_to_end_effector = [0, 0, 0.118]
        position, quaternion = p.multiplyTransforms(positionA=local_frame_position,
                                                    orientationA=local_frame_quaternion,
                                                    positionB=tool_offset_to_end_effector,
                                                    orientationB=[0, 0, 0, 1])
        return np.array(position), np.array(quaternion)

    def get_tip_pose(self):
        position = (np.array(p.getLinkState(self.id, linkIndex=self.links_all_dict['left_inner_finger'],
                                            computeForwardKinematics=1)[0]) +
                    np.array(p.getLinkState(self.id, linkIndex=self.links_all_dict['right_inner_finger'],
                                            computeForwardKinematics=1)[0]))/2
        _, quaternion = self.get_end_effector_pose()
        return position, quaternion

    def move_fingers(self, open_angle):

        p.setJointMotorControlArray(
            bodyUniqueId=self.id,
            jointIndices=self.gripper_joints_id_list,
            controlMode=p.POSITION_CONTROL,
            targetPositions=np.array(GRIPPER_RATIOS)*open_angle)

    def open_gripper(self):
        self.move_fingers(_open_angle(GRIPPER_OPEN))

    def close_gripper(self):
        self.move_fingers(_open_angle(GRIPPER_CLOSE))

    def reset_gripper(self):
        reset_angle = _open_angle(GRIPPER_OPEN)
        for i in range(len(self.gripper_joints_id_list)):
            p.resetJointState(self.id, self.gripper_joints_id_list[i], GRIPPER_RATIOS[i]*reset_angle)
