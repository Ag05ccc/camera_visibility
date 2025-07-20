
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple
import numpy as np

# Spherical Earth radius (m)
R_EARTH = 6378137.0

# -----------------------------------------------------------------------------
# Utility conversions
# -----------------------------------------------------------------------------

def deg2rad(d: float) -> float:
    return math.radians(d)


def rad2deg(r: float) -> float:
    return math.degrees(r)


def latlonalt_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Convert geodetic (spherical approx) to ECEF XYZ (meters)."""
    lat = deg2rad(lat_deg)
    lon = deg2rad(lon_deg)
    r = R_EARTH + alt_m
    x = r * math.cos(lat) * math.cos(lon)
    y = r * math.cos(lat) * math.sin(lon)
    z = r * math.sin(lat)
    return np.array([x, y, z], dtype=float)


def enu_axes(lat_deg: float, lon_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local unit vectors (East, North, Up) in ECEF coords at lat/lon."""
    lat = deg2rad(lat_deg)
    lon = deg2rad(lon_deg)
    # Up
    up = np.array([
        math.cos(lat) * math.cos(lon),
        math.cos(lat) * math.sin(lon),
        math.sin(lat),
    ], dtype=float)
    # East = d/dlon normalized
    east = np.array([-math.sin(lon), math.cos(lon), 0.0], dtype=float)
    # North = Up × East?  Actually, we want orthonormal: north = np.cross(up, east)
    north = np.cross(up, east)
    # Normalize
    east /= np.linalg.norm(east)
    north /= np.linalg.norm(north)
    up /= np.linalg.norm(up)
    return east, north, up


def rodrigues_rotate(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate vector *v* about unit *axis* by angle (rad)."""
    axis = axis / np.linalg.norm(axis)
    v = np.asarray(v, dtype=float)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


@dataclass
class CameraVisibility:
    lat: float
    lon: float
    alt: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    fov_h_deg: float
    fov_v_deg: float

    def _camera_basis_in_enu(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute camera Forward/Right/Up unit vectors expressed in **ENU**.

        Conventions:
        - Forward (optical axis) = East when yaw=pitch=roll=0.
        - Yaw rotates about *Up* (+CCW East→North).
        - Pitch rotates about *Right* (+nose up).
        - Roll rotates about *Forward* (+right‑wing‑down).
        """
        # east, north, up = enu_axes(self.lat, self.lon)
        # fwd = east.copy()
        # right = np.cross(fwd, up); right /= np.linalg.norm(right)  # roughly South when yaw=0
        # up_c = up.copy()

        east, north, up = enu_axes(self.lat, self.lon)

        # PX4 forward = body X axis = ENU North
        fwd = north.copy()
        right = east.copy()         # PX4 right = body Y = ENU East
        up_c = up.copy()            # PX4 up = body Z = ENU Up

        # Yaw about Up
        if self.yaw_deg:
            ang = deg2rad(self.yaw_deg)
            fwd = rodrigues_rotate(fwd, up_c, ang)
            right = rodrigues_rotate(right, up_c, ang)
            # up_c unchanged

        # Pitch about Right (nose up positive -> rotate Forward toward Up)
        if self.pitch_deg:
            ang = deg2rad(self.pitch_deg)
            fwd = rodrigues_rotate(fwd, right, ang)
            up_c = rodrigues_rotate(up_c, right, ang)
            # right unchanged

        # Roll about Forward (right‑wing‑down positive rotates Up toward Right)
        if self.roll_deg:
            ang = deg2rad(self.roll_deg)
            right = rodrigues_rotate(right, fwd, ang)
            up_c = rodrigues_rotate(up_c, fwd, ang)
            # fwd unchanged

        # Re‑orthonormalize (drift protection)
        fwd /= np.linalg.norm(fwd)
        right -= fwd * np.dot(fwd, right); right /= np.linalg.norm(right)
        up_c = np.cross(fwd, right); up_c /= np.linalg.norm(up_c)
        return fwd, right, up_c

    # ------------------------------------------------------------------
    # Core visibility check
    # ------------------------------------------------------------------
    def can_see(self, tgt_lat: float, tgt_lon: float, tgt_alt: float) -> bool:
        """Return True if target lies within camera FOV (no range limits)."""
        cam_ecef = latlonalt_to_ecef(self.lat, self.lon, self.alt)
        tgt_ecef = latlonalt_to_ecef(tgt_lat, tgt_lon, tgt_alt)
        vec = tgt_ecef - cam_ecef
        # Express in ENU
        east, north, up = enu_axes(self.lat, self.lon)
        enu_mat = np.vstack([east, north, up]).T  # columns
        vec_enu = enu_mat.T @ vec  # coords in ENU frame

        # Camera basis in ENU
        fwd, right, up_c = self._camera_basis_in_enu()
        cam_mat = np.vstack([fwd, right, up_c]).T
        # ENU -> Camera
        # (cam_mat columns are cam axes in ENU, so inverse = transpose)
        vec_cam = cam_mat.T @ (enu_mat @ vec_enu)  # Wait, we double transform? simplify below
        # Actually we already have vec_enu (coords). To express in camera: dot with axes
        vec_cam = np.array([
            np.dot(vec_enu, fwd),
            np.dot(vec_enu, right),
            np.dot(vec_enu, up_c),
        ])

        if vec_cam[0] <= 0:  # behind forward axis
            return False

        # Horizontal / vertical angles
        horiz_ang = math.degrees(math.atan2(vec_cam[1], vec_cam[0]))  # right vs forward
        vert_ang = math.degrees(math.atan2(vec_cam[2], math.hypot(vec_cam[0], vec_cam[1])))
        return (abs(horiz_ang) <= self.fov_h_deg * 0.5) and (abs(vert_ang) <= self.fov_v_deg * 0.5)

    # ------------------------------------------------------------------
    # Convenience: get camera frustum corner rays in ECEF for plotting.
    # ------------------------------------------------------------------
    def frustum_rays_ecef(self, range_m: float) -> np.ndarray:
        """Return 4 ECEF points reached by the frustum edge rays at *range_m*.

        Order: [top‑left, top‑right, bottom‑right, bottom‑left]."""
        cam_ecef = latlonalt_to_ecef(self.lat, self.lon, self.alt)
        fwd, right, up_c = self._camera_basis_in_enu()
        # Express basis in ECEF directly: fwd, right, up_c are in ECEF already.
        # Build direction vectors for each corner.
        h = math.tan(deg2rad(self.fov_h_deg * 0.5))
        v = math.tan(deg2rad(self.fov_v_deg * 0.5))
        # directions in camera frame: (1, ±h, ±v) normalized
        dirs_cam = np.array([
            [1.0, -h,  v],  # TL (image coordinates x right, y up) -> choose consistent
            [1.0,  h,  v],  # TR
            [1.0,  h, -v],  # BR
            [1.0, -h, -v],  # BL
        ])
        # Build camera matrix columns = fwd, right, up_c
        C = np.column_stack([fwd, right, up_c])  # ECEFx3
        pts = []
        for d in dirs_cam:
            d_norm = d / np.linalg.norm(d)
            # Map to ECEF: d_world = C @ [fwd, right, up] coords
            d_world = C @ d_norm
            pts.append(cam_ecef + d_world * range_m)
        return np.array(pts)
