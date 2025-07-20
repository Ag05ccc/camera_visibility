#!/usr/bin/env python
import rospy
from sensor_msgs.msg import NavSatFix, Imu
import math
import geopy.distance
from tf.transformations import euler_from_quaternion
from camera_visibility import CameraVisibility
DLON = 0.001

class CameraFootprint:
    def __init__(self):

        # Define camera 
        # self.cam = CameraVisibility(
        #     lat=0.0, lon=0.0, alt=0.0,          # camera at the Equator, sea‑level
        #     yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
        #     fov_h_deg=90.0, fov_v_deg=60.0)
        
        self.cam = CameraVisibility(
            lat=0, lon=0, alt=0,
            yaw_deg=0, pitch_deg=0, roll_deg=0,
            fov_h_deg=90,
            fov_v_deg=60
        )
                
        # GPS
        self.uav0_lat = self.uav0_lon = self.uav0_alt = 0.0
        self.uav1_lat = self.uav1_lon = self.uav1_alt = 0.0

        # Orientation
        self.uav0_rpy = [None, None, None]  # roll, pitch, yaw in degrees
        self.uav1_rpy = [None, None, None]

        # Camera settings (defaults)
        self.cam_pitch = 0  # in degrees (forward-looking)
        self.fov_horizontal = 90
        self.fov_vertical = 60

        # Subscriptions
        rospy.Subscriber('/uav0/mavros/global_position/global', NavSatFix, self.uav0_gps_cb)
        rospy.Subscriber('/uav1/mavros/global_position/global', NavSatFix, self.uav1_gps_cb)

        rospy.Subscriber('/uav0/mavros/imu/data', Imu, self.uav0_imu_cb)
        rospy.Subscriber('/uav1/mavros/imu/data', Imu, self.uav1_imu_cb)

        rospy.Timer(rospy.Duration(0.1), self.check_visibility)


        rospy.loginfo("CameraVisibility initialized and listening to GPS and IMU data")

    def check_visibility(self, event):
        if None in [self.cam.lat, self.cam.lon, self.cam.alt,
                    self.uav1_lat, self.uav1_lon, self.uav1_alt]:
            return

        visible = self.cam.can_see(self.uav1_lat, self.uav1_lon, self.uav1_alt)
        if visible:
            rospy.loginfo("✅ UAV1 is VISIBLE from UAV0 (geometric).")
        else:
            rospy.loginfo("❌ UAV1 is NOT visible from UAV0 (geometric).")

    def uav0_gps_cb(self, msg: NavSatFix):
        self.uav0_lat = msg.latitude
        self.uav0_lon = msg.longitude
        self.uav0_alt = msg.altitude
        
        self.cam.lat = msg.latitude
        self.cam.lon = msg.longitude
        self.cam.alt = msg.altitude

        # rospy.loginfo("uav0_lat: %.3f°, uav0_lon: %.3f°, uav0_alt: %.3f°", self.uav0_lat, self.uav0_lon, self.uav0_alt)

        # can_see_result = self.cam.can_see(self.uav1_lat, self.uav1_lon, self.uav1_alt)

    def uav1_gps_cb(self, msg: NavSatFix):
        self.uav1_lat = msg.latitude
        self.uav1_lon = msg.longitude
        self.uav1_alt = msg.altitude
        # rospy.loginfo("uav1_lat: %.3f°, uav1_lon: %.3f°, uav1_alt: %.3f°", self.uav1_lat, self.uav1_lon, self.uav1_alt)
        

    def uav0_imu_cb(self, msg):
        r, p, y = self.quaternion_to_euler(msg.orientation)
        self.uav0_rpy = [math.degrees(r), math.degrees(p), math.degrees(y)]
        self.cam.roll_deg  = self.uav0_rpy[0]
        self.cam.pitch_deg = self.uav0_rpy[1]
        self.cam.yaw_deg   = self.uav0_rpy[2] + 180

    def uav1_imu_cb(self, msg):
        r, p, y = self.quaternion_to_euler(msg.orientation)
        self.uav1_rpy = [math.degrees(r), math.degrees(p), math.degrees(y)]

    def quaternion_to_euler(self, q):
        return euler_from_quaternion([
            q.x,
            q.y,
            q.z,
            q.w
        ])


if __name__ == '__main__':
    rospy.init_node('camera_visibility_dual', anonymous=True)
    CameraFootprint()
    rospy.spin()
