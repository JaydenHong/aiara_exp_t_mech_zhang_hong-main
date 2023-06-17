import time

import cv2
import multiprocessing as mp
import os

class VirtualCamera:
    def __init__(self, frame_queue, fps=30, file_num=0):
        self.frame_queue = frame_queue
        self.fps = fps
        self.process = None
        self.file_num = file_num

    def start(self):
        self.process = mp.Process(target=self._run)
        self.process.start()

    def stop(self):
        self.process.terminate()

    def _run(self):
        video_path = "rtsp://192.168.1.10/color"
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        frame_size = (frame_width, frame_height)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        file_name = f"robot_cam{self.file_num:02}.avi"
        folder_name = "saved_video"
        file_path = os.path.join(folder_name, file_name)
        out = cv2.VideoWriter(file_path, fourcc, self.fps, frame_size)

        while True:
            ret, frame = cap.read()

            # Display the frame here
            # cv2.imshow("Virtual Camera inside process", frame)

            if ret:
                # check if the frame queue has any frames in it
                cv2.imshow("Virtual Camera", frame)
                out.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

            # if ret:
            #     # check if the frame queue has any frames in it
            #     # if not self.frame_queue.empty():
            #     #     # if it does, clear it
            #     #     while not self.frame_queue.empty():
            #     #         self.frame_queue.get()
            #     # self.frame_queue.put(frame)
            #     out.write(frame)
            #     cv2.imshow("Virtual Camera", frame)
            # else:
            #     break

        cap.release()
        out.release()
        self.frame_queue.put(None)


def main():
    # access the virtual camera
    frame_queue = mp.Queue()
    virtual_camera = VirtualCamera(frame_queue)
    virtual_camera.start()
    # view the frames coming off the camera
    # while True:
    #     frame = frame_queue.get()
    #     if frame is None:
    #         break
    #     cv2.imshow("Virtual Camera outside process", frame)
    #     if cv2.waitKey(1) == 'q':
    #         break
    time.sleep(5)
    virtual_camera.stop()


if __name__ == "__main__":
    main()
