import cv2
import numpy as np
from scipy import stats
from ultralytics import YOLO

MODE = 0

# Loop through the video frames
def process_image(img, contrast, brightness):
    # contrast adjustment
    alpha_contrast = 131 * (contrast + 127) / (127 * (131 - contrast))
    gamma_contrast = 127 * (1 - alpha_contrast)
    img = cv2.addWeighted(img, alpha_contrast, img, 0, gamma_contrast)

    # brightness adjustment
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_brightness = (highlight - shadow) / 255
        gamma_brightness = shadow
        img = cv2.addWeighted(img, alpha_brightness, img, 0, gamma_brightness)
    return img


# Detect lines using hough transform
def get_lines(img):
    dst = cv2.Canny(img, 200, 400, L2gradient=True)
    HOUGHLINE_THRESHOLD = 45
    while True:
        lines = cv2.HoughLines(dst, 1, np.pi/180, HOUGHLINE_THRESHOLD, None, 0, 0)
        if lines is None:
            lines = []

        # detect at least 4 lines or lower the threshold
        if len(lines) >= 4:
            break
        elif HOUGHLINE_THRESHOLD == 2:
            print("line detection failed")
            break
        else:
            HOUGHLINE_THRESHOLD -= 1

    # get angles only
    angles = lines[:, :, 1].ravel().tolist()
    # get rotation from avg of positive angles and the perpendicular angles of negative angles
    angles = [theta - np.pi/2 if theta > np.pi/2 else theta for theta in angles]
    # in case of 2 deg and 88 deg case(still there are two groups in the first quadrant)
    # threshold ~= 80 deg difference
    if (max(angles) - min(angles)) > np.pi/2 * 0.9:
        angles = [theta - np.pi/2 if theta > np.pi/4 else theta for theta in angles]

    rotation = stats.trim_mean(angles, 0.25)

    # turn ccw if theta > pi/4
    if rotation > np.pi/4:
        rotation -= np.pi / 2

    # # #draw lines
    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    #         cv2.line(img, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    # cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", img)
    # cv2.waitKey(0)
    return rotation


def get_transformation(cube_centers, goal_center, cup_center):

    # Cube2
    ref_vision_cube2 = np.float32([(86.617, 96.518), (76.482, 571.51), (1194.1, 585.38), (1217.1, 107.71)])
    ref_robot_cube2 = np.float32([(51.7, 42.9), (27, 40.5), (31, -16.2), (55.8, -16)]) * 0.01
    T_cube2 = cv2.getPerspectiveTransform(ref_vision_cube2, ref_robot_cube2)

    # Cube3
    ref_vision_cube3 = np.float32([(61.577, 82.77), (65.084, 581.93), (1210.2, 592.83), (1205.8, 93.25)])
    ref_robot_cube3 = np.float32([(52, 44), (26.7, 40.6), (31.2, -16.6), (56.4, -15)]) * 0.01
    T_cube3 = cv2.getPerspectiveTransform(ref_vision_cube3, ref_robot_cube3)

    # Cup
    ref_vision_cup = np.float32([(160.8, 168.6), (142.2, 549.52), (1136.74, 539.66), (1132.2, 183.0)])
    ref_robot_cup = np.float32([(47.5, 36.2), (29.6, 35), (34.1, -10.9), (50.6, -9.7)]) * 0.01

    T_cup = cv2.getPerspectiveTransform(ref_vision_cup, ref_robot_cup)

    # Goal
    ref_vision_goal = np.float32([(137.45, 101.87), (84.8, 587.2), (1202.6, 589.0), (1160.25, 127.05)])
    ref_robot_goal = np.float32([(52.3, 42.3), (25.6, 41.5), (30.5, -18.4), (55.4, -14.8)]) * 0.01

    T_goal = cv2.getPerspectiveTransform(ref_vision_goal, ref_robot_goal)

    cube2_robot =cv2.perspectiveTransform(np.float32([[cube_centers[0]]]), T_cube2)
    cube3_robot =cv2.perspectiveTransform(np.float32([cube_centers[1:]]), T_cube3)
    # goal_robot =cv2.perspectiveTransform(np.float32([[goal_center]]), T_cube3)
    # cup_robot = cv2.perspectiveTransform(np.float32([[cup_center]]), T_cube3)
    goal_robot =cv2.perspectiveTransform(np.float32([[goal_center]]), T_goal)
    cup_robot = cv2.perspectiveTransform(np.float32([[cup_center]]), T_cup)
    coordinates = cube2_robot.reshape(1, 2).tolist() +\
                  cube3_robot.reshape(2, 2).tolist() +\
                  goal_robot.reshape(1, 2).tolist() +\
                  cup_robot.reshape(1, 2).tolist()

    return coordinates


def execute():
    # Load the YOLOv8 model
    model = YOLO('/home/aiara/Documents/yolo_cube/runs/detect/train/weights/best.pt')
    # model = YOLO('yolov8m.pt')
    # Open the video file
    video_path = "rtsp://192.168.1.10/color"
    cap = cv2.VideoCapture(video_path)

    cups = []
    goals = []
    isCubeDetected = False
    NUM_CUBES = 3

    while True:

        success, frame = cap.read()

        if success:
            # take enough samples for goal and cup poses
            if not (len(goals) > 9 and len(cups) > 9 and isCubeDetected):

                # Detect cubes
                if isCubeDetected is False:
                    results = model(frame)
                    # Check num of bboxes
                    if len(results[0]) >= NUM_CUBES:
                        # check duplicates
                        bbox_wh = np.array(results[0].boxes.xywh.tolist())
                        cube_centers = [np.array(center) for center in zip(bbox_wh[:, 0], bbox_wh[:, 1])]
                        print(cube_centers)
                        remove = []
                        for i in range(NUM_CUBES):
                            for j in range(i + 1, NUM_CUBES):
                                if np.linalg.norm(cube_centers[i] - cube_centers[j]) < 40:
                                    remove.append(j)
                        bbox_wh = [item for idx, item in enumerate(bbox_wh) if idx not in remove]
                        cube_centers = [item for idx, item in enumerate(cube_centers) if idx not in remove]
                        print(cube_centers)
                        # check number of detected cubes after removal of duplicates
                        if len(bbox_wh) == NUM_CUBES:
                            isCubeDetected = True


                # visualize on annotate frame
                annotated_frame = results[0].plot(labels=True)

                # get goal pos

                frame1hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                GREEN_MIN = (40, 90, 160)
                GREEN_MAX = (65, 150, 255)
                mask = cv2.inRange(frame1hsv, GREEN_MIN, GREEN_MAX)
                frame_green = cv2.bitwise_and(frame1hsv, frame1hsv, mask=mask)
                frame1 = cv2.cvtColor(frame_green, cv2.COLOR_BGR2GRAY)
                # frame1 = process_image(frame1, contrast=50, brightness=-15)
                frame1 = cv2.GaussianBlur(frame1, (3, 3), 0)
                # frame1 = cv2.Canny(frame1, 80, 150, L2gradient=True)
                goal_threshold = 30
                while True:
                    goal = cv2.HoughCircles(frame1, cv2.HOUGH_GRADIENT, 1.2, 100, param1=150, param2=goal_threshold,
                                            minRadius=10, maxRadius=25)
                    if goal is None:
                        goal_threshold -= 1
                        if goal_threshold == 10:
                            break
                    else:
                        break

                # get cup pos
                frame2 = process_image(frame, contrast=10, brightness=-50)
                frame2 = cv2.GaussianBlur(frame2, (11, 11), 0)
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                cup_threshold = 30
                while True:
                    cup = cv2.HoughCircles(frame2, cv2.HOUGH_GRADIENT, 1.2, 100, param1=200, param2=cup_threshold,
                                           minRadius=95, maxRadius=110)
                    if cup is None:
                        cup_threshold -= 1
                        if cup_threshold == 20:
                            break
                    else:
                        break

                # visualize on the annotated frame
                if goal is not None:
                    goals.append(goal[0][0])
                    cv2.circle(frame_green, (int(goals[-1][0]), int(goals[-1][1])), int(goals[-1][2]), (0, 0, 255), thickness=2)

                if cup is not None:
                    cups.append(cup[0][0])
                    cv2.circle(annotated_frame, (int(cups[-1][0]), int(cups[-1][1])), int(cups[-1][2]), (255, 0, 0), thickness=2)

                cv2.imshow("YOLOv8 Inference", annotated_frame)
                # cv2.imshow("YOLOv8 Inference", frame_green)

                print("is cube detected:", isCubeDetected,
                      ", goal detection:", len(goals),
                      ", cup detection:", len(cups))

            else:  # Detected
                # get trimmed mean for goal and cup
                goal_center = [stats.trim_mean(np.array(goals)[:, 0], 0.25), stats.trim_mean(np.array(goals)[:, 1], 0.25)]
                goal_r = stats.trim_mean(np.array(goals)[:, 2], 0.25)
                cup_center = [stats.trim_mean(np.array(cups)[:, 0], 0.25), stats.trim_mean(np.array(cups)[:, 1], 0.25)]
                cup_r = stats.trim_mean(np.array(cups)[:, 2], 0.25)
                print('goal:', goal_center, 'radius:', goal_r)
                print('cup:', cup_center, 'radius:', cup_r)
                print('bbox:', len(results[0].boxes.xywh.tolist()))

                # visualize on the annotated frame
                annotated_frame = results[0].plot(labels=True)
                cv2.circle(annotated_frame, (int(cup_center[0]), int(cup_center[1])), int(cup_r), (255, 0, 0), thickness=2)
                cv2.circle(annotated_frame, (int(goal_center[0]), int(goal_center[1])), int(goal_r), (0, 0, 255), thickness=2)
                cv2.imshow("YOLOv8 Inference", annotated_frame)

                # get rotation angle for each bbox
                # bbox_wh = np.intp(bbox_wh)  # bbox to crop images
                try:
                    cube_angles = [get_lines(
                        frame[int(bbox_wh[i][1] - bbox_wh[i][3] / 2):int(bbox_wh[i][1] + bbox_wh[i][3] / 2),
                        int(bbox_wh[i][0] - bbox_wh[i][2] / 2):int(bbox_wh[i][0] + bbox_wh[i][2] / 2)])
                        for i in range(3)]
                    # get size of each cube
                    cube_sizes = [max(bbox_wh[i][2], bbox_wh[i][3]) / (np.cos(abs(theta)) + np.sin(abs(theta)))
                                  for i, theta in enumerate(cube_angles)]

                    cube_centers, cube_angles, cube_sizes = zip(*sorted(zip(cube_centers, cube_angles, cube_sizes), key=lambda x: x[2]))
                    print("cube_sizes(px):", cube_sizes)
                    print("cube_centers(px):", cube_centers)
                    rotations_robot = [93 - x * 180 / np.pi for x in cube_angles]
                    print("cube_rotations:", rotations_robot)
                    positions_robot = get_transformation(cube_centers, goal_center, cup_center)
                    print("cube_centers(m):", positions_robot[0:3])

                    # # pause here after detection
                    # cv2.waitKey(0)

                    sizes_robot = [x for x in cube_angles]
                    cap.release()
                    cv2.destroyAllWindows()
                    return positions_robot, rotations_robot, sizes_robot
                    # return [cube_s_pos, cube_m_pos, cube_l_pos, goal_pos, cup_pos],
                    #        [cube_s_rot, cube_m_rot, cube_l_rot],
                    #        [cube_s_size, cube_m_size, cube_l_size]

                except:
                    isCubeDetected = False  # Detect again
                    annotated_frame = frame

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


    return False


if __name__ == '__main__':
    result = execute()
    print(result)
