# test_camera_visibility.py
import math
from camera_visibility import CameraVisibility      # ← import your class

# ────────────────────────────────────────────────────────────────────────────────
# Helper: ~0.001° of lon ≃ 111 m at the equator – plenty close for this test.
DLON = 0.001

def test_front_vs_back():
    cam = CameraVisibility(
        lat=0.0, lon=0.0, alt=0.0,          # camera at the Equator, sea‑level
        yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
        fov_h_deg=60.0, fov_v_deg=40.0)

    # ❶  ~111 m due‑east, exactly on the optical axis (yaw = 0° faces east in the class)
    assert cam.can_see(0.0,  DLON, 0.0), "Point on optical axis should be visible"

    # ❷  Same distance due‑west – behind the sensor, must be rejected
    assert not cam.can_see(0.0, -DLON, 0.0), "Point behind camera should be invisible"

test_front_vs_back()