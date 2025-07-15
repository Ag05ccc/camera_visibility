import numpy as np
from math import sin, cos, atan2, sqrt, radians

class CameraVisibility:
    """Check whether a ground point is in the camera’s FOV."""

    # ───────────────────────────────────────────────────────────── WGS‑84 constants
    _a = 6_378_137.0             # semi‑major axis  [m]
    _f = 1 / 298.257223563
    _b = _a * (1 - _f)
    _e2 = _f * (2 - _f)

    # ──────────────────────────────────────────────────────────── basic conversions
    @staticmethod
    def _geodetic_to_ecef(lat_deg, lon_deg, h):
        lat, lon = map(radians, (lat_deg, lon_deg))
        N = CameraVisibility._a / sqrt(1 - CameraVisibility._e2 * sin(lat)**2)
        x = (N + h) * cos(lat) * cos(lon)
        y = (N + h) * cos(lat) * sin(lon)
        z = (N * (1 - CameraVisibility._e2) + h) * sin(lat)
        return np.array([x, y, z])

    @staticmethod
    def _ecef_to_enu(ecef, lat0_deg, lon0_deg, h0):
        """Return ENU vector from camera to target."""
        lat0, lon0 = map(radians, (lat0_deg, lon0_deg))
        x0, y0, z0 = CameraVisibility._geodetic_to_ecef(lat0_deg, lon0_deg, h0)
        dx, dy, dz = ecef - np.array([x0, y0, z0])
        e = -sin(lon0) * dx +  cos(lon0) * dy
        n = -sin(lat0) * cos(lon0) * dx - sin(lat0) * sin(lon0) * dy + cos(lat0) * dz
        u =  cos(lat0) * cos(lon0) * dx + cos(lat0) * sin(lon0) * dy + sin(lat0) * dz
        return np.array([e, n, u])

    @staticmethod
    def _enu_to_camera(v_enu, yaw_deg, pitch_deg, roll_deg):
        """Rotate ENU vector into the camera body frame (+X forward, +Y right, +Z down)."""
        ya, pi, ro = map(radians, (yaw_deg, pitch_deg, roll_deg))

        R_yaw   = np.array([[ cos(ya), sin(ya), 0],
                            [-sin(ya), cos(ya), 0],
                            [       0,       0, 1]])

        R_pitch = np.array([[ cos(pi), 0, -sin(pi)],
                            [       0, 1,        0],
                            [ sin(pi), 0,  cos(pi)]])

        R_roll  = np.array([[1,        0,         0],
                            [0,  cos(ro),  sin(ro)],
                            [0, -sin(ro),  cos(ro)]])

        return R_roll @ R_pitch @ R_yaw @ v_enu

    # ───────────────────────────────────────────────────────────────────── interface
    def __init__(self, lat, lon, alt,
                 yaw_deg, pitch_deg, roll_deg,
                 fov_h_deg, fov_v_deg):
        self.lat, self.lon, self.alt = lat, lon, alt
        self.yaw, self.pitch, self.roll = yaw_deg, pitch_deg, roll_deg
        self.fov_h = radians(fov_h_deg)
        self.fov_v = radians(fov_v_deg)

    def can_see(self, tgt_lat, tgt_lon, tgt_alt=0.0):
        """Return True if P₁ is inside the FOV rectangle."""
        ecef_tgt = self._geodetic_to_ecef(tgt_lat, tgt_lon, tgt_alt)
        v_enu    = self._ecef_to_enu(ecef_tgt, self.lat, self.lon, self.alt)
        v_cam    = self._enu_to_camera(v_enu,
                                       self.yaw, self.pitch, self.roll)

        # If target is behind the sensor (x ≤ 0) it is invisible
        if v_cam[0] <= 0:
            return False

        # Angular deviation from optical axis
        az   = atan2(v_cam[1], v_cam[0])            # horizontal (rad)
        el   = atan2(-v_cam[2], sqrt(v_cam[0]**2 + v_cam[1]**2))  # vertical (rad)

        return (abs(az) <= self.fov_h / 2) and (abs(el) <= self.fov_v / 2)

# ───────────────────────────────────────────────────────────── usage example
if __name__ == "__main__":
    cam = CameraVisibility(
        lat=40.00000, lon=29.00000, alt=100.0,   # camera position (m)
        yaw_deg= 30.0, pitch_deg=-10.0, roll_deg=0.0,
        fov_h_deg=60.0, fov_v_deg=40.0)

    print("Target visible:",
          cam.can_see(40.0008, 29.0007, 10.0))     # → True / False