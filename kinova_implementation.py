import numpy as np
import pandas as pd
import sys
import os
import time
import threading
import multiprocessing as mp
import cv2
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2
from kortex_api.Exceptions.KServerException import KServerException
from kortex_api.autogen.messages import DeviceConfig_pb2, Session_pb2, DeviceManager_pb2, VisionConfig_pb2
import ObjectDetection
import calculate_trajectory
from virtual_camera import VirtualCamera
from filepath_generator import filepath

FILE_PATH, FILE_NUM = filepath()

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20

# Object info
CUBE2_HEIGHT = 0.048
CUBE3_HEIGHT = 0.0585
CUBE4_HEIGHT = 0.065

# One Step behind the Goal pos to grip (45mm offset)
Z_OFFSET_SAFE = 0.07
CUBE2_Z_SAFE = CUBE2_HEIGHT + Z_OFFSET_SAFE
CUBE3_Z_SAFE = CUBE3_HEIGHT + Z_OFFSET_SAFE
CUBE4_Z_SAFE = CUBE3_HEIGHT + Z_OFFSET_SAFE

# Goal pos to grip (half point)
CUBE2_Z_GRIP = CUBE2_HEIGHT / 2
CUBE3_Z_GRIP = CUBE3_HEIGHT / 2
CUBE4_Z_GRIP = CUBE4_HEIGHT / 2

# Obstacle info
CUP_HEIGHT = 0.152
# CUP_HEIGHT_SAFE = CUP_HEIGHT + CUBE3_Z_GRIP  # Half length
# CUP_HEIGHT_SAFE = CUP_HEIGHT + CUBE3_Z_SAFE  # Full length = 0.24
CUP_HEIGHT_SAFE = 0.3
CUP_RADIUS_SAFE = 0.075

# Vision Position
VISION_POS = [0.356, 0.106, 0.562]

# Flags
movement_completed =False
record_enabled =True

option_dir = 'BC_32'

def nm2(vec):
    return np.linalg.norm(vec, ord=2)

def get_coordinate_with_blend_radius(positions, orientations):
    positions = [np.array(pos) for pos in positions]
    distance = [nm2(positions[i] - positions[i + 1]) for i in range(len(positions) - 1)]
    tangent_distance = [min(distance[i], distance[i + 1]) for i in range(len(positions) - 2)]
    radius = [tangent_distance[i] / 3 for i in range(len(positions) - 2)]
    radius.insert(0, 0)
    radius.append(0)
    new_coordinate = [wp.tolist() + [radius[i]] + orientations[i] for i, wp in enumerate(positions)]
    # new_coordinate = [wp.tolist() + [0] + orientations[i] for i, wp in enumerate(positions)]

    return new_coordinate


class pose:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.theta_x = 0
        self.theta_y = 0
        self.theta_z = 0
        self.joint = [0] * 7


def FK(base, joint_angles):
    # joint_angles = base.GetMeasuredJointAngles()
    return base.ComputeForwardKinematics(joint_angles)


def IK(base, ref_joint, pose_cartesian):
    # get robot's pose (by using forward kinematics)
    try:
        ref_joint = base.GetMeasuredJointAngles()
    except KServerException as ex:
        print("Unable to get current robot pose")
        print("Error_code:{} , Sub_error_code:{} ".format(ex.get_error_code(), ex.get_error_sub_code()))
        print("Caught expected error: {}".format(ex))
        return False

    # Object containing cartesian coordinates and Angle Guess
    input_IkData = Base_pb2.IKData()

    # Fill the IKData Object with the cartesian coordinates that need to be converted
    input_IkData.cartesian_pose.x = pose_cartesian.x
    input_IkData.cartesian_pose.y = pose_cartesian.y
    input_IkData.cartesian_pose.z = pose_cartesian.z
    input_IkData.cartesian_pose.theta_x = pose_cartesian.theta_x
    input_IkData.cartesian_pose.theta_y = pose_cartesian.theta_y
    input_IkData.cartesian_pose.theta_z = pose_cartesian.theta_z

    # Fill the IKData Object with the guessed joint angles
    for joint_angle in ref_joint.joint_angles:
        jAngle = input_IkData.guess.joint_angles.add()
        jAngle.value = joint_angle.value

    try:
        print("Computing Inverse Kinematics using joint angles and pose...")
        computed_joint_angles = base.ComputeInverseKinematics(input_IkData)
    except KServerException as ex:
        print("Unable to compute inverse kinematics")
        print("Error_code:{} , Sub_error_code:{} ".format(ex.get_error_code(), ex.get_error_sub_code()))
        print("Caught expected error: {}".format(ex))
        return False, None

    print("Joint ID : Joint Angle")

    for joint_identifier, joint_angle in enumerate(computed_joint_angles.joint_angles):
        print(joint_identifier, " : ", joint_angle.value)

    return True, computed_joint_angles


def vision_get_device_id(device_manager):
    vision_device_id = 0

    # Getting all device routing information (from DeviceManagerClient service)
    all_devices_info = device_manager.ReadAllDevices()

    vision_handles = [hd for hd in all_devices_info.device_handle if hd.device_type == DeviceConfig_pb2.VISION]
    if len(vision_handles) == 0:
        print("Error: there is no vision device registered in the devices info")
    elif len(vision_handles) > 1:
        print("Error: there are more than one vision device registered in the devices info")
    else:
        handle = vision_handles[0]
        vision_device_id = handle.device_identifier
        print("Vision module found, device Id: {0}".format(vision_device_id))

    return vision_device_id


# Autofocus continuously
def autofocus(vision_config, vision_device_id):
    sensor_focus_action = VisionConfig_pb2.SensorFocusAction()
    sensor_focus_action.sensor = VisionConfig_pb2.SENSOR_COLOR
    sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_START_CONTINUOUS_FOCUS
    vision_config.DoSensorFocusAction(sensor_focus_action, vision_device_id)


# Function to record video
def record_video():
    global record_enabled

    # access the virtual camera
    frame_queue = mp.Queue()
    virtual_camera = VirtualCamera(frame_queue)
    virtual_camera.start()
    # view the frames coming off the camera
    print("haha")
    while record_enabled:
        frame = frame_queue.get()
        if frame is None:
            break
        cv2.imshow("Virtual Camera outside process", frame)
        cv2.waitKey(1)


REC_IDX = 1

def record_actual_pos(base, sampling_frequency):
    global REC_IDX
    global movement_completed
    start_time = time.time()
    file_path = FILE_PATH
    file_name = "executed_trajectory{}.txt".format(REC_IDX)
    file_path_and_name = os.path.join(file_path, file_name)
    REC_IDX += 1
    time_list = []
    pose_list = []
    i = 0
    while not movement_completed:
        i += 1
        try:
            time_list.append(time.time() - start_time)
            input_joint_angles = base.GetMeasuredJointAngles()
            pose_list.append(base.ComputeForwardKinematics(input_joint_angles))
            # print(current_time, pose.x, pose.y, pose.z, pose.theta_x, pose.theta_y, pose.theta_z)
        except Exception as exception:
            print("FK failed:")
        remaining_time = sampling_frequency * i - (time.time() - start_time)
        if remaining_time > 0:
            time.sleep(remaining_time)

    with open(file_path_and_name, "w") as file:
        file.write("Time\tx\ty\tz\trx\try\trz\n")

        for time_val, pose in zip(time_list, pose_list):
            line = "{:.2f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n"\
                .format(time_val, pose.x, pose.y, pose.z, pose.theta_x, pose.theta_y, pose.theta_z)
            file.write(line)


class ActionAbortError(Exception):
    pass


def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """

    def check(notification, e=e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
                or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
        if notification.action_event == Base_pb2.ACTION_ABORT:
            raise ActionAbortError("Action was aborted.")

    return check


def move_to_action_position(base, action_name):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

    # Move arm to ready position
    print("Moving the arm to a safe position")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None
    for action in action_list.action_list:
        if action.name == action_name:
            action_handle = action.handle

    if action_handle == None:
        print("Can't reach safe position. Exiting")
        return False

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteActionFromReference(action_handle)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Safe position reached")
    else:
        print("Timeout on action notification wait")
    return finished


def populateCartesianCoordinate(waypointInformation):
    waypoint = Base_pb2.CartesianWaypoint()
    waypoint.pose.x = waypointInformation[0]
    waypoint.pose.y = waypointInformation[1]
    waypoint.pose.z = waypointInformation[2]
    waypoint.blending_radius = waypointInformation[3]
    waypoint.pose.theta_x = waypointInformation[4]
    waypoint.pose.theta_y = waypointInformation[5]
    waypoint.pose.theta_z = waypointInformation[6]
    waypoint.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE

    return waypoint


def move_to_cartesian_trajectory(base, base_cyclic, waypointsDefinition, log=True):
    global movement_completed

    # Set servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

    waypoints = Base_pb2.WaypointList()

    waypoints.duration = 0
    waypoints.use_optimal_blending = False

    index = 0
    for waypointDefinition in waypointsDefinition:
        waypoint = waypoints.waypoints.add()
        waypoint.name = "waypoint_" + str(index)
        waypoint.cartesian_waypoint.CopyFrom(populateCartesianCoordinate(waypointDefinition))
        index = index + 1

    print(waypoint)

    #### IK check if needed

    # Verify validity of waypoints
    result = base.ValidateWaypointList(waypoints)

    if (len(result.trajectory_error_report.trajectory_error_elements) == 0):
        e = threading.Event()
        notification_handle = base.OnNotificationActionTopic(check_for_end_or_abort(e),
                                                             Base_pb2.NotificationOptions())
        if log is True:
            movement_completed = False
            record_thread = threading.Thread(target=record_actual_pos, args=(base, 0.1))
            record_thread.start()
        print("Moving cartesian trajectory...")

        try:
            base.ExecuteWaypointTrajectory(waypoints)
            print("Waiting for trajectory to finish ...")
            finished = e.wait(TIMEOUT_DURATION)
        except ActionAbortError as e:
            finished = False
        except Exception as e:
            print("An unexpected error occurred:", str(e))
            finished = False
        finally:
            base.Unsubscribe(notification_handle)

        if log is True:
            movement_completed = True
            record_thread.join()

        return finished
    else:
        raise ValueError("Invalid Waypoints")


SEQ_IDX = 1

def get_waypoints(init_pos, init_ori, goal_pos, goal_ori, obst_pos, pre_init_pos=None, post_goal_pos=None):
    global SEQ_IDX
    # calculate Trajectory
    print("poses:", init_pos, goal_pos, obst_pos)
    file_path = FILE_PATH
    file_name = 'computed_trajectory{}.txt'.format(SEQ_IDX)
    file_path_and_name = os.path.join(file_path, file_name)
    SEQ_IDX += 1
    calculate_trajectory.execute(init_pos, goal_pos, obst_pos, file_name=file_name, file_path=file_path)
    exp_data = pd.read_table(file_path_and_name)

    # Read positions
    CurrPos_x = exp_data.Pos_1.to_numpy()
    CurrPos_y = exp_data.Pos_2.to_numpy()
    CurrPos_z = exp_data.Pos_3.to_numpy()
    positions = [np.array(i) for i in zip(CurrPos_x, CurrPos_y, CurrPos_z)]
    positions.append(np.array(goal_pos))

    # Get orientations
    init_orientation = np.array(init_ori)
    goal_orientation = np.array(goal_ori)
    orientations = np.linspace(init_orientation, goal_orientation, len(positions)).tolist()

    # Append pre_init_pos at the beginning if init safe position is good to have blend radius
    if pre_init_pos is not None:
        positions.insert(0, pre_init_pos)
        orientations.insert(0, init_ori)

    # Append post_goal_pos at the end if goal safe position is good to have blend radius
    if post_goal_pos is not None:
        positions.append(post_goal_pos)
        orientations.append(goal_ori)

    # Get Blend radius
    return get_coordinate_with_blend_radius(positions, orientations)


def SendGripperCommands(base, position=0.0):
    # Create the GripperCommand we will send
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()

    # Position control
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    # finger.finger_identifier = 1
    finger.value = position
    print("Going to position {:0.2f}...".format(finger.value))
    base.SendGripperCommand(gripper_command)

    # Check if gripper is at desired position
    gripper_request = Base_pb2.GripperRequest()
    gripper_request.mode = Base_pb2.GRIPPER_POSITION
    previous_grip = -1
    while True:
        gripper_measure = base.GetMeasuredGripperMovement(gripper_request)
        if len(gripper_measure.finger):
            current_grip = gripper_measure.finger[0].value

            print("Current position is : {0}".format(current_grip))
            if abs(current_grip - position) < 0.01 or\
                    (abs(previous_grip - current_grip) < 0.0001 and abs(previous_grip - current_grip)>0) :
                break
            previous_grip = current_grip
        else:  # Else, no finger present in answer, end loop
            break

    print("gripping is done")



if __name__ == '__main__':

    # Turning on virtual camera
    frame_queue = mp.Queue()
    print(FILE_PATH)
    virtual_camera = VirtualCamera(frame_queue, file_num=FILE_NUM)
    virtual_camera.start()
    print("waiting for virtual camera...")
    # time.sleep(8)
    os.mkdir(FILE_PATH)

    # Import the utilities helper module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse arguments
    args = utilities.parseConnectionArguments()
    args.ip = '192.168.1.10'

    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:

        # Create required services
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

        # Execute movement sequence

        # 0. Move to Home pos
        # move_to_action_position(base, "Home")
        SendGripperCommands(base, position=0.0)

        # 2. Move to vis pos
        move_to_action_position(base, "vision_pos_joint")
        time.sleep(1)

        # 3. Get position
        positions, rotations, sizes = ObjectDetection.execute(file_num=FILE_NUM)
        print(positions)

        cube_s_size, cube_m_size, cube_l_size = sizes
        # sizes can be later used to determine how much gripper should open, not implemented yet though

        heights = [CUBE2_Z_GRIP, CUBE3_Z_GRIP, CUBE3_Z_GRIP, CUBE3_Z_GRIP, CUP_HEIGHT_SAFE]
        # goal height (=height(4) need to be changed to CUBE4_Z_GRIP instead of CUBE3_Z_GRIP in case of using 4x4x4 cube
        # or be adjusted correspond to sizes input.
        cube_s_pos, cube_m_pos, cube_l_pos, goal_pos, obst_pos = [p+[h] for p, h in zip(positions, heights)]
        print(cube_s_pos, cube_m_pos, cube_l_pos, goal_pos, obst_pos)
        cube_s_ori, cube_m_ori, cube_l_ori = [[180, 0.01, rot] for rot in rotations]


        goal_ori = [180, 0.01, 90]
        z_offset_safe = np.array([0, 0, Z_OFFSET_SAFE])
        z_offset_l = np.array([0, 0, CUBE3_HEIGHT])
        z_offset_m = np.array([0, 0, CUBE3_HEIGHT])
        z_offset_s = np.array([0, 0, CUBE2_HEIGHT])

        # 4. Move to vis_safe pos
        move_to_action_position(base, "vision_safe_joint")

        # 5. generate trajectories

        seq = []
        grip = []
        input_joint_angles = base.GetMeasuredJointAngles()
        current_cartesian = base.ComputeForwardKinematics(input_joint_angles)
        cur_pos = [current_cartesian.x, current_cartesian.y, current_cartesian.z]
        cur_ori = [current_cartesian.theta_x, current_cartesian.theta_y, current_cartesian.theta_z]

        # Seq1(vis_safe -> cube L)
        print(cur_ori)
        print(cube_l_ori)
        seq.append(get_waypoints(pre_init_pos=None,
                                 init_pos=cur_pos, init_ori=cur_ori,
                                 goal_pos=cube_l_pos + z_offset_safe, goal_ori=cube_l_ori,
                                 post_goal_pos=cube_l_pos,
                                 obst_pos=obst_pos))
        # grip.append(0.0)
        grip.append(0.375)

        # Seq2(cube L -> goal)
        seq.append(get_waypoints(pre_init_pos=cube_l_pos,
                                 init_pos=cube_l_pos + z_offset_safe, init_ori=cube_l_ori,
                                 goal_pos=goal_pos + z_offset_safe, goal_ori=goal_ori,
                                 post_goal_pos=goal_pos,
                                 obst_pos=obst_pos))
        grip.append(0.0)

        # Seq3 (goal -> cube M)
        seq.append(get_waypoints(pre_init_pos=goal_pos,
                                 init_pos=goal_pos + z_offset_safe, init_ori=goal_ori,
                                 goal_pos=cube_m_pos + z_offset_safe, goal_ori=cube_m_ori,
                                 post_goal_pos=cube_m_pos,
                                 obst_pos=obst_pos))
        # grip.append(0.0)
        grip.append(0.375)

        # Seq4 (cube M -> goal)
        seq.append(get_waypoints(pre_init_pos=cube_m_pos,
                                 init_pos=cube_m_pos + z_offset_safe, init_ori=cube_m_ori,
                                 goal_pos=goal_pos + z_offset_safe + z_offset_l, goal_ori=goal_ori,
                                 post_goal_pos=goal_pos + z_offset_l,
                                 obst_pos=obst_pos))
        grip.append(0.0)

        # Seq5 (goal -> cube S)
        seq.append(get_waypoints(pre_init_pos=goal_pos + z_offset_l,
                                 init_pos=goal_pos + z_offset_safe + z_offset_l, init_ori=goal_ori,
                                 goal_pos=cube_s_pos + z_offset_safe, goal_ori=cube_s_ori,
                                 post_goal_pos=cube_s_pos,
                                 obst_pos=obst_pos))
        # grip.append(0.0)
        grip.append(0.495)

        # Seq6 (cube S -> goal)
        seq.append(get_waypoints(pre_init_pos=cube_s_pos,
                                 init_pos=cube_s_pos + z_offset_safe, init_ori=cube_s_ori,
                                 goal_pos=goal_pos + z_offset_safe + z_offset_m + z_offset_l, goal_ori=goal_ori,
                                 post_goal_pos=goal_pos + z_offset_m + z_offset_s,
                                 obst_pos=obst_pos))
        grip.append(0.0)

        # 6. execute trajectories

        for i, waypoints in enumerate(seq):
            try:
                result = move_to_cartesian_trajectory(base, base_cyclic, waypointsDefinition=waypoints)
                if result:
                    print("Trajectory execution completed.")
                    SendGripperCommands(base, position=grip[i])
                else:
                    print("Trajectory execution aborted or timed out.")
                    sys.exit(1)
            except ValueError as e:
                print("Error:", str(e))
                sys.exit(1)

        input_joint_angles = base.GetMeasuredJointAngles()
        current_cartesian = base.ComputeForwardKinematics(input_joint_angles)
        safe_pos = [[current_cartesian.x, current_cartesian.y, current_cartesian.z + 0.1, 0,
                    current_cartesian.theta_x, current_cartesian.theta_y, current_cartesian.theta_z]]

        move_to_cartesian_trajectory(base, base_cyclic, waypointsDefinition=safe_pos, log=False)

        # 7. Return home

        virtual_camera.stop()
        move_to_action_position(base, "Home")