#!/usr/bin/env python
import rospy
from sensor_msgs.msg import NavSatFix, Imu
import math
import geopy.distance
from tf.transformations import euler_from_quaternion

class CameraVisibility:
    def __init__(self):
        # GPS
        self.uav0_lat = self.uav0_lon = self.uav0_alt = None
        self.uav1_lat = self.uav1_lon = self.uav1_alt = None

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

        rospy.loginfo("CameraVisibility initialized and listening to GPS and IMU data")

    def uav0_gps_cb(self, msg):
        self.uav0_lat = msg.latitude
        self.uav0_lon = msg.longitude
        self.uav0_alt = msg.altitude
        self.check_visibility()

    def uav1_gps_cb(self, msg):
        self.uav1_lat = msg.latitude
        self.uav1_lon = msg.longitude
        self.uav1_alt = msg.altitude
        self.check_visibility()

    def uav0_imu_cb(self, msg):
        r, p, y = self.quaternion_to_euler(msg.orientation)
        self.uav0_rpy = [math.degrees(r), math.degrees(p), math.degrees(y)]

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

    def check_visibility(self):
        if None in [self.uav0_lat, self.uav0_lon, self.uav0_alt,
                    self.uav1_lat, self.uav1_lon, self.uav1_alt,
                    self.uav0_rpy[2]]:  # require yaw
            return

        # Camera yaw = UAV0's yaw
        cam_yaw = (self.uav0_rpy[2] + 90) % 360

        # Bearing from UAV0 to UAV1
        bearing = self.calculate_bearing((self.uav0_lat, self.uav0_lon),
                                         (self.uav1_lat, self.uav1_lon))
        distance = geopy.distance.geodesic((self.uav0_lat, self.uav0_lon),
                                           (self.uav1_lat, self.uav1_lon)).meters
        delta_alt = self.uav1_alt - self.uav0_alt
        elevation = math.degrees(math.atan2(delta_alt, distance))

        # Visibility check
        visible = self.angle_in_fov(bearing, cam_yaw, self.fov_horizontal) and \
                  self.angle_in_fov(elevation, self.cam_pitch, self.fov_vertical)

        if visible:
            rospy.loginfo("✅ UAV1 is VISIBLE from UAV0.")
        else:
            rospy.loginfo("❌ UAV1 is NOT visible from UAV0.")

        rospy.loginfo("Bearing: %.2f°, Yaw: %.2f°, Elevation: %.2f°", bearing, cam_yaw, elevation)

    def calculate_bearing(self, pointA, pointB):
        lat1 = math.radians(pointA[0])
        lat2 = math.radians(pointB[0])
        dlon = math.radians(pointB[1] - pointA[1])

        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - \
            math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360

    def angle_in_fov(self, angle, center, fov):
        diff = (angle - center + 180) % 360 - 180
        return abs(diff) <= (fov / 2)

if __name__ == '__main__':
    rospy.init_node('camera_visibility_dual', anonymous=True)
    CameraVisibility()
    rospy.spin()
