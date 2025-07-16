"""Local-Scale (≤3 km) Camera Visibility 3‑D Demo
===================================================

**What changed from the globe version?**

- Dropped full-Earth rendering; now a clean local Cartesian ENU patch (meters).
- Axes: X=East, Y=North, Z=Up.
- Fixed view box ±3,000 m so your points are easy to see.
- Simple ground grid at Z=0 every 250 m (tick labels in meters).
- Separate sliders for Camera (X,Y,Z) and Target (X,Y,Z) – no need to fuss with lat/lon.
- Camera yaw (deg, 0°=+X/East, CCW=turn toward +Y/North), pitch (+up), roll (+right‑wing‑down).
- Horizontal/Vertical FOV sliders.
- Visibility line colors: green visible, red not visible.
- Frustum wedge drawn to MAX_RANGE (default 3 km).

Because we operate entirely in a *local* Euclidean frame, Earth curvature is ignored.
At 3 km max range the curvature error (< ~0.0007 m drop) is irrelevant for visualization.

Run:
    pip install numpy matplotlib
    python demo_local3k.py

"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import math

MAX_RANGE = 3000.0  # meters (scene half-size)
GRID_SPACING = 250.0  # meters

# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def deg2rad(d):
    return np.deg2rad(d)


def rotmat_yaw_pitch_roll(yaw_deg, pitch_deg, roll_deg):
    """Return 3x3 direction cosine matrix from camera->world.

    Camera body axes:
        fwd = +X_cam  (optical axis)
        right = +Y_cam
        up = +Z_cam

    World axes (local ENU):
        +X East, +Y North, +Z Up.

    Convention:
        yaw about world Z (+CCW from +X toward +Y)
        pitch about camera right (+nose up)
        roll about camera fwd (+right-wing-down)
    """
    cy = math.cos(deg2rad(yaw_deg));  sy = math.sin(deg2rad(yaw_deg))
    # start: camera aligned with world (fwd->+X, right->+Y, up->+Z)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])  # yaw
    # apply pitch in camera-right after yaw: easier build incremental
    # We'll build using successive rotations of basis vectors rather than full matrix multiply for clarity
    # Compose matrices: R = Rz @ Rx_pitch @ Rfwd_roll  (where Rx_pitch rotates about camera-right AFTER yaw)
    # To get camera-right after yaw we need transform; easier do stepwise using vectors.
    # Simpler: create basis then rotate via Rodrigues.
    def rodrigues(v, axis, ang):
        axis = axis/np.linalg.norm(axis)
        c = math.cos(ang); s = math.sin(ang)
        return v*c + np.cross(axis,v)*s + axis*np.dot(axis,v)*(1-c)

    # Start basis
    fwd = np.array([1.0,0.0,0.0])
    right = np.array([0.0,1.0,0.0])
    up = np.array([0.0,0.0,1.0])

    # Yaw about world Up
    ang = deg2rad(yaw_deg)
    fwd = rodrigues(fwd, np.array([0,0,1]), ang)
    right = rodrigues(right, np.array([0,0,1]), ang)
    up = rodrigues(up, np.array([0,0,1]), ang)

    # Pitch about *right*
    ang = deg2rad(pitch_deg)
    fwd = rodrigues(fwd, right, ang)
    up = rodrigues(up, right, ang)

    # Roll about *fwd*
    ang = deg2rad(roll_deg)
    right = rodrigues(right, fwd, ang)
    up = rodrigues(up, fwd, ang)

    # Orthonormalize
    fwd = fwd/np.linalg.norm(fwd)
    right = right - np.dot(right,fwd)*fwd; right/=np.linalg.norm(right)
    up = np.cross(fwd,right); up/=np.linalg.norm(up)
    # Columns are world coords of camera axes
    return np.column_stack((fwd,right,up))


# ---------------------------------------------------------------------------
# Camera visibility in local Cartesian frame
# ---------------------------------------------------------------------------
class LocalCameraVisibility:
    def __init__(self, x=0.0,y=0.0,z=0.0, yaw_deg=0.0,pitch_deg=0.0,roll_deg=0.0, fov_h_deg=60.0,fov_v_deg=40.0):
        self.x=x; self.y=y; self.z=z
        self.yaw_deg=yaw_deg; self.pitch_deg=pitch_deg; self.roll_deg=roll_deg
        self.fov_h_deg=fov_h_deg; self.fov_v_deg=fov_v_deg

    @property
    def pos(self):
        return np.array([self.x,self.y,self.z],dtype=float)

    def dcm(self):
        return rotmat_yaw_pitch_roll(self.yaw_deg,self.pitch_deg,self.roll_deg)

    def can_see(self, tx,ty,tz):
        vec = np.array([tx-self.x, ty-self.y, tz-self.z],dtype=float)
        # transform world->cam (columns of R are world vectors of cam axes)
        R = self.dcm()
        vec_cam = R.T @ vec
        if vec_cam[0] <= 0:  # behind
            return False
        horiz_ang = math.degrees(math.atan2(vec_cam[1], vec_cam[0]))
        vert_ang  = math.degrees(math.atan2(vec_cam[2], math.hypot(vec_cam[0],vec_cam[1])))
        return (abs(horiz_ang) <= self.fov_h_deg*0.5) and (abs(vert_ang) <= self.fov_v_deg*0.5)

    def frustum_rays(self, rng=MAX_RANGE):
        h = math.tan(math.radians(self.fov_h_deg*0.5))
        v = math.tan(math.radians(self.fov_v_deg*0.5))
        dirs_cam = np.array([
            [1.0,-h, v],  # TL
            [1.0, h, v],  # TR
            [1.0, h,-v],  # BR
            [1.0,-h,-v],  # BL
        ])
        # normalize
        dirs_cam /= np.linalg.norm(dirs_cam,axis=1,keepdims=True)
        R = self.dcm()  # world axes of cam
        pts=[]
        origin=self.pos
        for d in dirs_cam:
            d_w = R @ d
            pts.append(origin + d_w*rng)
        return np.array(pts)


# ---------------------------------------------------------------------------
# Interactive Matplotlib UI
# ---------------------------------------------------------------------------
class DemoLocal3K:
    def __init__(self):
        self.cam = LocalCameraVisibility()
        self.tgt = np.array([500.0,0.0,0.0])  # 500 m east
        self._build_ui()
        self._redraw()

    def _build_ui(self):
        self.fig = plt.figure(figsize=(9,7))
        self.ax = self.fig.add_subplot(111,projection='3d')
        self.ax.set_box_aspect([1,1,0.5])
        plt.subplots_adjust(left=0.25,bottom=0.45)

        # Camera sliders
        self.sl_camx = self._new_slider(0.25,0.35,'Cam X (m)',-MAX_RANGE,MAX_RANGE,self.cam.x,self._on_cam_change)
        self.sl_camy = self._new_slider(0.25,0.30,'Cam Y (m)',-MAX_RANGE,MAX_RANGE,self.cam.y,self._on_cam_change)
        self.sl_camz = self._new_slider(0.25,0.25,'Cam Z (m)',0,MAX_RANGE,self.cam.z,self._on_cam_change)
        self.sl_yaw  = self._new_slider(0.25,0.20,'Yaw°',-180,180,self.cam.yaw_deg,self._on_pose_change)
        self.sl_pitch= self._new_slider(0.25,0.15,'Pitch°',-90,90,self.cam.pitch_deg,self._on_pose_change)
        self.sl_roll = self._new_slider(0.25,0.10,'Roll°',-180,180,self.cam.roll_deg,self._on_pose_change)
        self.sl_fovh = self._new_slider(0.25,0.05,'FOVh°',1,180,self.cam.fov_h_deg,self._on_pose_change)
        self.sl_fovv = self._new_slider(0.25,0.00,'FOVv°',1,180,self.cam.fov_v_deg,self._on_pose_change)

        # Target sliders
        self.sl_tgtx = self._new_slider(0.25,-0.05,'Tgt X (m)',-MAX_RANGE,MAX_RANGE,self.tgt[0],self._on_tgt_change)
        self.sl_tgty = self._new_slider(0.25,-0.10,'Tgt Y (m)',-MAX_RANGE,MAX_RANGE,self.tgt[1],self._on_tgt_change)
        self.sl_tgtz = self._new_slider(0.25,-0.15,'Tgt Z (m)',0,MAX_RANGE,self.tgt[2],self._on_tgt_change)

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _new_slider(self,x,y,label,vmin,vmax,valinit,cb):
        ax = plt.axes([x,y,0.65,0.03])
        sl = Slider(ax,label,vmin,vmax,valinit=valinit)
        sl.on_changed(cb)
        return sl

    def _draw_grid(self):
        # Ground plane grid at Z=0
        xs = np.arange(-MAX_RANGE,MAX_RANGE+1e-6,GRID_SPACING)
        ys = np.arange(-MAX_RANGE,MAX_RANGE+1e-6,GRID_SPACING)
        for x in xs:
            self.ax.plot([x,x],[ys[0],ys[-1]],[0,0],color='lightgray',linewidth=0.5,alpha=0.7)
        for y in ys:
            self.ax.plot([xs[0],xs[-1]],[y,y],[0,0],color='lightgray',linewidth=0.5,alpha=0.7)
        # outline box
        self.ax.plot([-MAX_RANGE,MAX_RANGE],[ -MAX_RANGE,-MAX_RANGE],[0,0],color='gray',linewidth=1)
        self.ax.plot([-MAX_RANGE,MAX_RANGE],[  MAX_RANGE, MAX_RANGE],[0,0],color='gray',linewidth=1)
        self.ax.plot([-MAX_RANGE,-MAX_RANGE],[-MAX_RANGE,MAX_RANGE],[0,0],color='gray',linewidth=1)
        self.ax.plot([ MAX_RANGE, MAX_RANGE],[-MAX_RANGE,MAX_RANGE],[0,0],color='gray',linewidth=1)

    def _on_cam_change(self,_):
        self.cam.x = self.sl_camx.val
        self.cam.y = self.sl_camy.val
        self.cam.z = self.sl_camz.val
        self._redraw()

    def _on_pose_change(self,_):
        self.cam.yaw_deg = self.sl_yaw.val
        self.cam.pitch_deg = self.sl_pitch.val
        self.cam.roll_deg = self.sl_roll.val
        self.cam.fov_h_deg = self.sl_fovh.val
        self.cam.fov_v_deg = self.sl_fovv.val
        self._redraw()

    def _on_tgt_change(self,_):
        self.tgt[0] = self.sl_tgtx.val
        self.tgt[1] = self.sl_tgty.val
        self.tgt[2] = self.sl_tgtz.val
        self._redraw()

    def _on_key(self,event):
        if event.key in ('q','escape'):
            plt.close(self.fig)
        elif event.key=='r':
            self.sl_camx.set_val(0.0); self.sl_camy.set_val(0.0); self.sl_camz.set_val(0.0)
            self.sl_yaw.set_val(0.0); self.sl_pitch.set_val(0.0); self.sl_roll.set_val(0.0)
            self.sl_fovh.set_val(60.0); self.sl_fovv.set_val(40.0)
            self.sl_tgtx.set_val(500.0); self.sl_tgty.set_val(0.0); self.sl_tgtz.set_val(0.0)
            print('Reset.')

    def _redraw(self):
        self.ax.cla()
        self._draw_grid()

        # Camera + Target markers
        cam_p = self.cam.pos
        tgt_p = self.tgt
        self.ax.scatter(*cam_p,s=80,marker='^',color='blue',label='Camera')
        self.ax.scatter(*tgt_p,s=80,marker='o',color='black',label='Target')

        # Frustum
        fr = self.cam.frustum_rays(MAX_RANGE)
        for p in fr:
            self.ax.plot([cam_p[0],p[0]],[cam_p[1],p[1]],[cam_p[2],p[2]],color='gray',linewidth=0.5)
        fr_loop = np.vstack((fr,fr[0]))
        self.ax.plot(fr_loop[:,0],fr_loop[:,1],fr_loop[:,2],color='gray',linewidth=0.5)

        vis = self.cam.can_see(*tgt_p)
        col = 'g' if vis else 'r'
        self.ax.plot([cam_p[0],tgt_p[0]],[cam_p[1],tgt_p[1]],[cam_p[2],tgt_p[2]],color=col,linewidth=2)

        self.ax.set_xlim(-MAX_RANGE,MAX_RANGE)
        self.ax.set_ylim(-MAX_RANGE,MAX_RANGE)
        self.ax.set_zlim(0,MAX_RANGE)
        self.ax.set_xlabel('East (m)')
        self.ax.set_ylabel('North (m)')
        self.ax.set_zlabel('Up (m)')
        self.ax.set_title(f"Target is {'VISIBLE' if vis else 'NOT visible'}")
        self.ax.legend(loc='upper left')
        self.fig.canvas.draw_idle()
        # console
        if vis:
            print('VISIBLE')
        else:
            print('NOT visible')


def main():
    demo = DemoLocal3K()
    plt.show()


if __name__ == '__main__':
    main()
