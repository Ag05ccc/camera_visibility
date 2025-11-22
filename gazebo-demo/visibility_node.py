#!/usr/bin/env python
import rospy
import numpy as np
import cv2

from sensor_msgs.msg import CameraInfo, Image
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge
from tf.transformations import quaternion_matrix

CAMERA_MODEL_NAME = "moving_camera"
BOX_MODEL_NAME    = "target_box"

class VisibilityVisualizer(object):
    def __init__(self):
        # Intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.img_w = None
        self.img_h = None

        # Poses
        self.cam_pose = None       # geometry_msgs/Pose
        self.box_pose = None       # geometry_msgs/Pose

        # Latest image
        self.bridge = CvBridge()
        self.latest_img = None
        self.latest_img_header = None

        # Subscribers
        rospy.Subscriber("/moving_camera/camera/camera_info", CameraInfo, self.camera_info_cb)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_states_cb)
        rospy.Subscriber("/moving_camera/camera/image_raw", Image, self.image_cb)

        # Publisher for annotated image
        self.vis_pub = rospy.Publisher("/moving_camera/visibility_image",
                                       Image, queue_size=1)
        
        # Video recording
        self.video_writer = None
        self.video_filename = rospy.get_param("~video_filename",
                                              "visibility_output2.avi")
        self.video_fps = rospy.get_param("~video_fps", 30.0)

        rospy.loginfo("VisibilityVisualizer initialized, waiting for data...")

    def camera_info_cb(self, msg):
        K = msg.K
        self.fx = K[0]
        self.fy = K[4]
        self.cx = K[2]
        self.cy = K[5]
        self.img_w = msg.width
        self.img_h = msg.height


    def model_states_cb(self, msg):
        # Camera pose
        try:
            cam_idx = msg.name.index(CAMERA_MODEL_NAME)
            self.cam_pose = msg.pose[cam_idx]
        except ValueError:
            pass

        # Box pose
        try:
            box_idx = msg.name.index(BOX_MODEL_NAME)
            self.box_pose = msg.pose[box_idx]
        except ValueError:
            pass

    def image_cb(self, msg):
        # Store latest image
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr("cv_bridge error: %s", e)
            return

        self.latest_img = cv_img
        self.latest_img_header = msg.header

        # Process and publish annotated image
        if self.ready():
            annotated = self.draw_box_on_image(cv_img.copy())
            out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            out_msg.header = msg.header
            self.vis_pub.publish(out_msg)
            h, w, _ = annotated.shape
        else:
            return
        # Initialize VideoWriter lazily when we know width/height
        if self.video_writer is None:
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or 'MJPG'
            self.video_writer = cv2.VideoWriter(self.video_filename,
                                                fourcc,
                                                self.video_fps,
                                                (w, h))

        # Write frame to video
        if self.video_writer is not None:
            self.video_writer.write(annotated)

    def ready(self):
        return (self.fx is not None and
                self.cam_pose is not None and
                self.box_pose is not None and
                self.latest_img is not None)

    @staticmethod
    def pose_to_matrix(pose):
        q = pose.orientation
        t = pose.position
        Q = [q.x, q.y, q.z, q.w]
        T = quaternion_matrix(Q)           # 4x4
        T[0:3, 3] = [t.x, t.y, t.z]
        return T

    def draw_box_on_image(self, img):
        """
        Compute the 2D bounding box of target_box and draw it on img.
        Returns the annotated image.
        """
        # Build transforms
        T_world_cam = self.pose_to_matrix(self.cam_pose)
        T_cam_world = np.linalg.inv(T_world_cam)
        T_world_box = self.pose_to_matrix(self.box_pose)

        # Box size 1x1x1 from SDF: size=1 1 1, centered at link origin
        half = 0.5
        corners_box = np.array([
            [ half,  half,  half, 1.0],
            [ half,  half, -half, 1.0],
            [ half, -half,  half, 1.0],
            [ half, -half, -half, 1.0],
            [-half,  half,  half, 1.0],
            [-half,  half, -half, 1.0],
            [-half, -half,  half, 1.0],
            [-half, -half, -half, 1.0],
        ]).T  # shape (4, 8)

        # Box frame -> world
        corners_world = np.dot(T_world_box, corners_box)      # (4,8)

        # World -> camera
        corners_cam = np.dot(T_cam_world, corners_world)      # (4,8)

        us = []
        vs = []

        for i in range(corners_cam.shape[1]):
            Xg, Yg, Zg, _ = corners_cam[:, i]

            # Convert from Gazebo camera frame (X forward, Y right, Z up)
            # to optical/OpenCV frame (Z forward, X right, Y down)
            Zc = Xg
            Xc = -Yg
            Yc = -Zg

            if Zc <= 0.0:
                # Behind camera in optical frame
                continue

            u = self.fx * (Xc / Zc) + self.cx
            v = self.fy * (Yc / Zc) + self.cy


            if 0 <= u < self.img_w and 0 <= v < self.img_h:
                us.append(u)
                vs.append(v)

        if len(us) == 0:
            # No visible corners -> nothing to draw
            rospy.loginfo_throttle(1.0, "Box not visible in image.")
            return img

        u_min = int(min(us))
        u_max = int(max(us))
        v_min = int(min(vs))
        v_max = int(max(vs))

        # Draw rectangle
        cv2.rectangle(img, (u_min, v_min), (u_max, v_max), (0, 255, 0), 2)
        # Optionally label it
        cv2.putText(img, "target_box",
                    (u_min, max(v_min - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

        rospy.loginfo_throttle(
            0.5,
            "Box bbox: u=[%d,%d], v=[%d,%d]" % (u_min, u_max, v_min, v_max)
        )

        return img

def main():
    rospy.init_node("visibility_node")
    VisibilityVisualizer()
    rospy.loginfo("Visibility node running, publishing /moving_camera/visibility_image")
    rospy.spin()

if __name__ == "__main__":
    main()
