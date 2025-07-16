import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# Re‑import from same file if running in single script mode
# (When split into individual files, remove these local definitions.)
from camera_visibility import *

R_EARTH = 6378137.0

class CameraTargetDemo:
    def __init__(self):
        # State
        self.cam = CameraVisibility(
            lat=0.0, lon=0.0, alt=0.0,
            yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
            fov_h_deg=60.0, fov_v_deg=40.0,
        )
        self.tgt = dict(lat=0.0, lon=1.0, alt=0.0)  # 1° east
        self.selected = 'Camera'  # which sliders edit
        self.frustum_range_m = 1_000_000.0  # 1000 km just for display scale

        # Figure / axes
        self.fig = plt.figure(figsize=(9, 7))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect([1, 1, 1])
        self._init_globe()
        self._init_widgets()
        self._connect_events()
        self._redraw()

    # ------------------------------------------------------------------
    def _init_globe(self, n=32):
        """Draw wireframe Earth."""
        u = np.linspace(0, 2 * np.pi, n)
        v = np.linspace(-np.pi / 2, np.pi / 2, n // 2)
        xs = R_EARTH * np.outer(np.cos(v), np.cos(u))
        ys = R_EARTH * np.outer(np.cos(v), np.sin(u))
        zs = R_EARTH * np.outer(np.sin(v), np.ones_like(u))
        self.ax.plot_wireframe(xs, ys, zs, color='k', linewidth=0.2, alpha=0.4)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')

    # ------------------------------------------------------------------
    def _init_widgets(self):
        # Layout: reserve bottom area for sliders
        plt.subplots_adjust(left=0.25, bottom=0.40)

        # Radio buttons: select Camera / Target
        ax_radio = plt.axes([0.025, 0.80, 0.15, 0.15])
        self.radio_entity = RadioButtons(ax_radio, ('Camera', 'Target'))
        self.radio_entity.on_clicked(self._on_entity_change)

        # Camera orientation sliders (always active)
        self.sl_yaw = self._new_slider(0.25, 0.30, 'Yaw°', -180, 180, self.cam.yaw_deg, self._on_pose_change)
        self.sl_pitch = self._new_slider(0.25, 0.25, 'Pitch°', -90, 90, self.cam.pitch_deg, self._on_pose_change)
        self.sl_roll = self._new_slider(0.25, 0.20, 'Roll°', -180, 180, self.cam.roll_deg, self._on_pose_change)
        self.sl_fovh = self._new_slider(0.25, 0.15, 'FOVh°', 1, 180, self.cam.fov_h_deg, self._on_pose_change)
        self.sl_fovv = self._new_slider(0.25, 0.10, 'FOVv°', 1, 180, self.cam.fov_v_deg, self._on_pose_change)

        # Position sliders (edit selected entity)
        self.sl_lat = self._new_slider(0.25, 0.05, 'Lat°', -90, 90, self.cam.lat, self._on_latlonalt_change)
        self.sl_lon = self._new_slider(0.25, 0.00, 'Lon°', -180, 180, self.cam.lon, self._on_latlonalt_change)
        self.sl_alt = self._new_slider(0.25, -0.05, 'Alt m', -10_000, 1_000_000, self.cam.alt, self._on_latlonalt_change)

    # Helper to create slider
    def _new_slider(self, x, y, label, vmin, vmax, valinit, cb):
        ax = plt.axes([x, y, 0.65, 0.03])
        sl = Slider(ax, label, vmin, vmax, valinit=valinit)
        sl.on_changed(cb)
        return sl

    # ------------------------------------------------------------------
    def _connect_events(self):
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    # ------------------------------------------------------------------
    def _on_entity_change(self, label):
        self.selected = label
        # Update sliders to show current selection values
        if label == 'Camera':
            self.sl_lat.set_val(self.cam.lat)
            self.sl_lon.set_val(self.cam.lon)
            self.sl_alt.set_val(self.cam.alt)
        else:
            self.sl_lat.set_val(self.tgt['lat'])
            self.sl_lon.set_val(self.tgt['lon'])
            self.sl_alt.set_val(self.tgt['alt'])

    # ------------------------------------------------------------------
    def _on_pose_change(self, _val):
        self.cam.yaw_deg = self.sl_yaw.val
        self.cam.pitch_deg = self.sl_pitch.val
        self.cam.roll_deg = self.sl_roll.val
        self.cam.fov_h_deg = max(1e-3, self.sl_fovh.val)
        self.cam.fov_v_deg = max(1e-3, self.sl_fovv.val)
        self._redraw()

    # ------------------------------------------------------------------
    def _on_latlonalt_change(self, _val):
        if self.selected == 'Camera':
            self.cam.lat = self.sl_lat.val
            self.cam.lon = self.sl_lon.val
            self.cam.alt = self.sl_alt.val
        else:
            self.tgt['lat'] = self.sl_lat.val
            self.tgt['lon'] = self.sl_lon.val
            self.tgt['alt'] = self.sl_alt.val
        self._redraw()

    # ------------------------------------------------------------------
    def _on_key(self, event):
        if event.key in ('q', 'escape'):
            plt.close(self.fig)
        elif event.key == 'r':
            # Reset
            self.cam.lat = 0.0; self.cam.lon = 0.0; self.cam.alt = 0.0
            self.cam.yaw_deg = 0.0; self.cam.pitch_deg = 0.0; self.cam.roll_deg = 0.0
            self.cam.fov_h_deg = 60.0; self.cam.fov_v_deg = 40.0
            self.tgt = dict(lat=0.0, lon=1.0, alt=0.0)
            self._on_entity_change(self.selected)  # refresh sliders
            self.sl_yaw.set_val(self.cam.yaw_deg)
            self.sl_pitch.set_val(self.cam.pitch_deg)
            self.sl_roll.set_val(self.cam.roll_deg)
            self.sl_fovh.set_val(self.cam.fov_h_deg)
            self.sl_fovv.set_val(self.cam.fov_v_deg)
            print("Reset.")
            self._redraw()

    # ------------------------------------------------------------------
    def _redraw(self):
        self.ax.cla()
        self._init_globe()

        # Points
        cam_ecef = latlonalt_to_ecef(self.cam.lat, self.cam.lon, self.cam.alt)
        tgt_ecef = latlonalt_to_ecef(self.tgt['lat'], self.tgt['lon'], self.tgt['alt'])

        self.ax.scatter(*cam_ecef, s=60, marker='^', label='Camera')
        self.ax.scatter(*tgt_ecef, s=60, marker='o', label='Target')

        # Frustum edges
        fr = self.cam.frustum_rays_ecef(self.frustum_range_m)
        # Draw pyramid from cam to each corner + base loop
        for p in fr:
            self.ax.plot([cam_ecef[0], p[0]], [cam_ecef[1], p[1]], [cam_ecef[2], p[2]], color='gray', linewidth=0.5)
        # base loop
        fr_loop = np.vstack([fr, fr[0]])
        self.ax.plot(fr_loop[:,0], fr_loop[:,1], fr_loop[:,2], color='gray', linewidth=0.5)

        # Visibility line
        visible = self.cam.can_see(self.tgt['lat'], self.tgt['lon'], self.tgt['alt'])
        col = 'g' if visible else 'r'
        self.ax.plot([cam_ecef[0], tgt_ecef[0]], [cam_ecef[1], tgt_ecef[1]], [cam_ecef[2], tgt_ecef[2]], color=col, linewidth=2.0)
        self.ax.legend(loc='upper left')

        # Autoscale view to see both points; choose limit ~Earth radius + alt range
        max_r = R_EARTH + max(abs(self.cam.alt), abs(self.tgt['alt']), self.frustum_range_m * 0.1)
        lim = max_r * 1.1
        for setlim in (self.ax.set_xlim, self.ax.set_ylim, self.ax.set_zlim):
            setlim(-lim, lim)
        self.ax.set_title(f"Target is {'VISIBLE' if visible else 'NOT visible'}")
        self.fig.canvas.draw_idle()
        print(f"Visible = {visible}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    demo = CameraTargetDemo()
    plt.show()