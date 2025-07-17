#!/usr/bin/env python3
"""camera_visibility_ros_node.py

Real‑time *local flat‑Earth* camera FOV check for ROS.

This node subscribes to:
    * **/camera_pose**  (geometry_msgs/PoseStamped) – camera position & attitude in a fixed world frame (e.g., ``map`` or ``odom``) that you treat as local ENU (X=East, Y=North, Z=Up).
    * **/target_point** (geometry_msgs/PointStamped) – target location in the *same* world frame.
    * **/camera_info**  (sensor_msgs/CameraInfo) – optional. If present we use true intrinsics to project 3‑D to pixels. If absent we fall back to node params ``fov_h_deg`` & ``fov_v_deg`` + ``image_width``/``image_height`` to build a pinhole model.

It publishes:
    * **/target_visible** (std_msgs/Bool) – True if target lies within camera frustum (in front + inside FOV).
    * **/target_pixel**   (geometry_msgs/Point32) – pixel coordinates ``(u, v)`` if visible; ``(-1, -1)`` if not.

Optional (debug) topics can be added easily – see TODOs at bottom.

Coordinate / Axis conventions
-----------------------------
We re‑use the **LocalCameraVisibility** math from your core module, where camera body axes are:

    X_cam = forward (optical axis)
    Y_cam = right
    Z_cam = up

World frame is ENU (X=East, Y=North, Z=Up).

Euler definitions:
    * yaw   – rotation about world +Z (CCW from +X toward +Y).
    * pitch – rotation about camera *right* (+Y_cam) positive nose‑up.
    * roll  – rotation about camera *forward* (+X_cam) positive right‑wing‑down.

> **What if my ROS camera_optical_frame follows REP‑103/REP‑105 (+Z forward, +X right, +Y down)?**
>
> Set param ``ros_optical_convention:=true`` and we internally remap to the math convention above.

Usage
-----
::

    rosrun your_pkg camera_visibility_ros_node.py _fov_h_deg:=62.0 _fov_v_deg:=48.0 _image_width:=1920 _image_height:=1080

or with a launch file (recommended).

Dependencies: ``rospy``, ``tf_transformations`` (or ``tf_conversions``), ``numpy``.

"""

from __future__ import annotations
import math
import numpy as np
import rospy
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, PointStamped, Point32, Quaternion
from sensor_msgs.msg import CameraInfo

# ---------------------------------------------------------------------------
# Local Camera Visibility math (stripped & inlined to avoid import hassle)
# ---------------------------------------------------------------------------

def _deg2rad(d: float) -> float:
    return math.radians(d)


def _rodrigues(v: np.ndarray, axis: np.ndarray, ang: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    c = math.cos(ang)
    s = math.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


def rotmat_yaw_pitch_roll(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Return 3×3 DCM that maps *camera* vectors into *world* ENU.

    Camera body axes assumed X=fwd, Y=right, Z=up. See module docstring.
    """
    # Start canonical basis in world coords
    fwd = np.array([1.0, 0.0, 0.0])
    right = np.array([0.0, 1.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])

    # 1) yaw about world Z
    ang = _deg2rad(yaw_deg)
    fwd = _rodrigues(fwd, np.array([0, 0, 1]), ang)
    right = _rodrigues(right, np.array([0, 0, 1]), ang)
    up = _rodrigues(up, np.array([0, 0, 1]), ang)

    # 2) pitch about *right*
    ang = _deg2rad(pitch_deg)
    fwd = _rodrigues(fwd, right, ang)
    up = _rodrigues(up, right, ang)

    # 3) roll about *fwd*
    ang = _deg2rad(roll_deg)
    right = _rodrigues(right, fwd, ang)
    up = _rodrigues(up, fwd, ang)

    # Orthonormalize
    fwd = fwd / np.linalg.norm(fwd)
    right = right - np.dot(right, fwd) * fwd
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    up /= np.linalg.norm(up)
    return np.column_stack((fwd, right, up))  # world vectors of cam axes


class LocalCameraVisibility(object):
    __slots__ = (
        "x",
        "y",
        "z",
        "yaw_deg",
        "pitch_deg",
        "roll_deg",
        "fov_h_deg",
        "fov_v_deg",
    )

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        fov_h_deg: float = 60.0,
        fov_v_deg: float = 40.0,
    ) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.yaw_deg = yaw_deg
        self.pitch_deg = pitch_deg
        self.roll_deg = roll_deg
        self.fov_h_deg = fov_h_deg
        self.fov_v_deg = fov_v_deg

    # --- convenience ---------------------------------------------------
    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def dcm(self) -> np.ndarray:
        return rotmat_yaw_pitch_roll(self.yaw_deg, self.pitch_deg, self.roll_deg)

    # --- core test -----------------------------------------------------
    def can_see(self, tx: float, ty: float, tz: float):
        """Return (visible: bool, horiz_deg, vert_deg, range_m)."""
        vec = np.array([tx - self.x, ty - self.y, tz - self.z], dtype=float)
        rng = np.linalg.norm(vec)
        if rng <= 0.0:
            return True, 0.0, 0.0, 0.0  # same point
        R = self.dcm()  # world->cam via transpose
        vec_cam = R.T @ vec
        if vec_cam[0] <= 0.0:  # behind camera forward axis
            return False, None, None, rng
        horiz_ang = math.degrees(math.atan2(vec_cam[1], vec_cam[0]))
        vert_ang = math.degrees(math.atan2(vec_cam[2], math.hypot(vec_cam[0], vec_cam[1])))
        vis = (abs(horiz_ang) <= self.fov_h_deg * 0.5) and (
            abs(vert_ang) <= self.fov_v_deg * 0.5
        )
        return vis, horiz_ang, vert_ang, rng


# ---------------------------------------------------------------------------
# Helpers: quaternion→Euler (ENU), REP‑105 optical→body, projection
# ---------------------------------------------------------------------------

try:
    from tf.transformations import euler_from_quaternion
except ImportError:
    # Minimal internal fallback (YXZ order afterwards) – we will approximate by default tf's XYZ (roll,pitch,yaw) intrinsic
    # but since user sets ros_optical_convention we treat yaw about Z, pitch about Y, roll about X for convenience.
    def euler_from_quaternion(q):  # type: ignore
        x, y, z, w = q
        # ROS standard: returns roll(X), pitch(Y), yaw(Z)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        return roll, pitch, yaw


def quat_to_ypr(q: Quaternion) -> tuple[float, float, float]:
    """Convert a ROS geometry_msgs/Quaternion to (yaw,pitch,roll) *degrees* in ENU frame.

    We interpret the quaternion as expressing the rotation from world->camera_body where
    camera_body axes follow ROS optical convention if param ``ros_optical_convention`` is True; otherwise
    we assume it already matches the math convention (X=fwd,Y=right,Z=up).
    """
    roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])  # radians
    # euler_from_quaternion returns roll(X), pitch(Y), yaw(Z) in the world frame of the message.
    # Convert to degrees and reorder yaw,pitch,roll.
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def optical_to_body_quat(q: Quaternion) -> Quaternion:
    """If ROS optical convention (+Z forward, +X right, +Y down) needs conversion to body (+X fwd, +Y right, +Z up).

    The transform from optical->body is fixed:
        R_body_opt = [[0,  0, 1],
                      [1,  0, 0],
                      [0, -1, 0]]
    (common OpenCV→ROS mapping variant). For simplicity we approximate via Euler swaps.
    NOTE: For precise work, use tf2 to apply a fixed static transform.
    """
    # Convert to matrix, apply fixed, back to quat – done inline for brevity.
    import tf.transformations as tft  # type: ignore

    m_opt = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
    R_body_opt = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)
    m_body = m_opt @ R_body_opt  # map optical axes into body frame
    q_body = tft.quaternion_from_matrix(m_body)
    return Quaternion(x=q_body[0], y=q_body[1], z=q_body[2], w=q_body[3])


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def project_angles_to_pixel(
    horiz_deg: float,
    vert_deg: float,
    width: int,
    height: int,
    fov_h_deg: float,
    fov_v_deg: float,
) -> tuple[int, int]:
    """Map angular offsets (deg) to pixel coordinates assuming symmetrical pinhole.

    0,0 pixel is upper‑left (OpenCV/ROS Image). u increases right, v increases *down*.
    vert_deg positive = up in camera; we invert sign for pixel row.
    """
    u = (horiz_deg / (fov_h_deg * 0.5)) * (width * 0.5) + (width * 0.5)
    v = (-vert_deg / (fov_v_deg * 0.5)) * (height * 0.5) + (height * 0.5)
    return int(round(u)), int(round(v))


def project_camvec_to_pixel(vec_cam: np.ndarray, K: np.ndarray) -> tuple[int, int]:
    """Project a camera‑frame 3‑D vector using intrinsic matrix K.

    ``vec_cam`` expected in camera *body* coords (X forward, Y right, Z up).
    We temporarily map to OpenCV coords (X right, Y down, Z forward) for standard pinhole:
        Xcv = +Y_cam
        Ycv = -Z_cam
        Zcv = +X_cam
    """
    Xcv = vec_cam[1]
    Ycv = -vec_cam[2]
    Zcv = vec_cam[0]
    if Zcv <= 0:
        return -1, -1
    x = Xcv / Zcv
    y = Ycv / Zcv
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    u = fx * x + cx
    v = fy * y + cy
    return int(round(u)), int(round(v))


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class CameraVisibilityNode(object):
    def __init__(self):
        # Params ---------------------------------------------------------
        self.fov_h_deg = rospy.get_param("~fov_h_deg", 60.0)
        self.fov_v_deg = rospy.get_param("~fov_v_deg", 40.0)
        self.image_width = int(rospy.get_param("~image_width", 640))
        self.image_height = int(rospy.get_param("~image_height", 480))
        self.ros_optical_convention = rospy.get_param("~ros_optical_convention", False)
        self.clip_to_image = rospy.get_param("~clip_to_image", True)
        self.max_range = rospy.get_param("~max_range", 3000.0)

        # State ----------------------------------------------------------
        self.have_cam = False
        self.have_tgt = False
        self.cam_pose = None  # type: PoseStamped | None
        self.tgt_point = None  # type: PointStamped | None
        self.K = None  # type: np.ndarray | None  # 3x3 intrinsics

        # Publishers -----------------------------------------------------
        self.pub_visible = rospy.Publisher("target_visible", Bool, queue_size=1)
        self.pub_pixel = rospy.Publisher("target_pixel", Point32, queue_size=1)

        # Subscribers ----------------------------------------------------
        rospy.Subscriber("camera_pose", PoseStamped, self._cb_cam_pose, queue_size=1)
        rospy.Subscriber("target_point", PointStamped, self._cb_tgt_point, queue_size=1)
        rospy.Subscriber("camera_info", CameraInfo, self._cb_camera_info, queue_size=1)

        rospy.loginfo("camera_visibility_ros_node: started.")

    # -- Callbacks ------------------------------------------------------
    def _cb_camera_info(self, msg: CameraInfo):
        self.image_width = msg.width
        self.image_height = msg.height
        K = np.array(msg.K, dtype=float).reshape(3, 3)
        self.K = K

    def _cb_cam_pose(self, msg: PoseStamped):
        self.cam_pose = msg
        self.have_cam = True
        self._try_compute()

    def _cb_tgt_point(self, msg: PointStamped):
        self.tgt_point = msg
        self.have_tgt = True
        self._try_compute()

    # -- Core update ----------------------------------------------------
    def _try_compute(self):
        if not (self.have_cam and self.have_tgt):
            return
        cam = self.cam_pose
        tgt = self.tgt_point

        # Ensure same frame
        if cam.header.frame_id != tgt.header.frame_id:
            rospy.logwarn_throttle(5.0, "camera_visibility: frame mismatch (%s vs %s)" % (cam.header.frame_id, tgt.header.frame_id))
            # In production you would TF transform here; we just bail.
            return

        # Camera position
        cx = cam.pose.position.x
        cy = cam.pose.position.y
        cz = cam.pose.position.z

        # Camera orientation
        q = cam.pose.orientation
        if self.ros_optical_convention:
            q = optical_to_body_quat(q)  # convert optical->body
        yaw_deg, pitch_deg, roll_deg = quat_to_ypr(q)

        # Target position
        tx = tgt.point.x
        ty = tgt.point.y
        tz = tgt.point.z

        # Setup camera model -------------------------------------------------
        # If intrinsics present, compute fovs from them *once*.
        if self.K is not None:
            fx = self.K[0, 0]
            fy = self.K[1, 1]
            cxp = self.K[0, 2]
            cyp = self.K[1, 2]
            # Convert fx->fov_h roughly: fov = 2*atan(width/(2*fx))
            self.fov_h_deg = math.degrees(2.0 * math.atan(self.image_width / (2.0 * fx)))
            self.fov_v_deg = math.degrees(2.0 * math.atan(self.image_height / (2.0 * fy)))
        else:
            cxp = self.image_width * 0.5
            cyp = self.image_height * 0.5

        camvis = LocalCameraVisibility(
            x=cx,
            y=cy,
            z=cz,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            fov_h_deg=self.fov_h_deg,
            fov_v_deg=self.fov_v_deg,
        )

        visible, horiz_deg, vert_deg, rng = camvis.can_see(tx, ty, tz)

        if rng > self.max_range:
            visible = False

        # Publish Bool ---------------------------------------------------
        self.pub_visible.publish(Bool(data=visible))

        # Publish pixel --------------------------------------------------
        if visible:
            if self.K is not None and horiz_deg is not None:
                # project via intrinsics; need vec_cam
                vec = np.array([tx - cx, ty - cy, tz - cz], dtype=float)
                vec_cam = camvis.dcm().T @ vec  # world->cam
                u, v = project_camvec_to_pixel(vec_cam, self.K)
            else:
                u, v = project_angles_to_pixel(
                    horiz_deg, vert_deg, self.image_width, self.image_height, self.fov_h_deg, self.fov_v_deg
                )
            if self.clip_to_image:
                if not (0 <= u < self.image_width and 0 <= v < self.image_height):
                    # outside actual image; treat as not visible
                    visible = False
                    self.pub_visible.publish(Bool(data=False))
                    u, v = -1, -1
        else:
            u, v = -1, -1

        self.pub_pixel.publish(Point32(x=float(u), y=float(v), z=0.0))

    # ------------------------------------------------------------------
    def spin(self):
        rospy.spin()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    rospy.init_node("camera_visibility_ros_node")
    node = CameraVisibilityNode()
    node.spin()


if __name__ == "__main__":
    main()
