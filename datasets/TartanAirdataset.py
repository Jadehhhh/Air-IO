"""
TartanAir V2 dataset loader for Air-IO.

Expected directory structure for each trajectory:
    <data_root>/<data_name>/
        imu/
            imu.npy          # shape (N, 7): [timestamp_ns, wx, wy, wz, ax, ay, az] @ ~100 Hz
        pose.txt             # one pose per image frame: "tx ty tz qx qy qz qw" in NED frame

The pose file records the IMU/body pose in the world (NED) frame at the image frame rate
(~10 Hz for V2). IMU data is at ~100 Hz. Ground-truth velocity is computed by differencing
consecutive positions and smoothing with a moving-average filter.

Coordinate convention (NED):
    x – forward, y – right, z – down
    gravity vector in body frame (when level) ≈ [0, 0, +9.81]

Reference: https://tartanair.org
"""

import os

import numpy as np
import pypose as pp
import torch
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d

from utils import qinterp
from .dataset import Sequence
import pickle


class TartanAir(Sequence):
    """Loader for the TartanAir V2 dataset trajectories."""

    def __init__(
        self,
        data_root,
        data_name,
        coordinate=None,
        mode=None,
        rot_path=None,
        rot_type=None,
        gravity=9.81007,
        remove_g=False,
        **kwargs,
    ):
        super(TartanAir, self).__init__()
        (
            self.data_root,
            self.data_name,
            self.data,
            self.ts,
            self.targets,
            self.orientations,
            self.gt_pos,
            self.gt_ori,
        ) = (data_root, data_name, dict(), None, None, None, None, None)

        # In the NED frame the gravity vector points downward along +z.
        self.g_vector = torch.tensor([0, 0, gravity], dtype=torch.double)

        data_path = os.path.join(data_root, data_name)

        # 1. Load raw sensor/GT data
        self.load_imu(data_path)
        self.load_gt(data_path)

        # 2. Trim to overlapping time interval
        t_start = max(self.data["gt_time"][0], self.data["time"][0])
        t_end = min(self.data["gt_time"][-1], self.data["time"][-1])

        idx_start_imu = np.searchsorted(self.data["time"], t_start)
        idx_end_imu = np.searchsorted(self.data["time"], t_end, "right")

        idx_start_gt = np.searchsorted(self.data["gt_time"], t_start)
        idx_end_gt = np.searchsorted(self.data["gt_time"], t_end, "right")

        for k in ["gt_time", "pos", "quat", "velocity"]:
            self.data[k] = self.data[k][idx_start_gt:idx_end_gt]
        for k in ["time", "acc", "gyro"]:
            self.data[k] = self.data[k][idx_start_imu:idx_end_imu]

        # Remove duplicate GT timestamps (required by spherical interpolation)
        _, unique_gt_idx = np.unique(self.data["gt_time"], return_index=True)
        if len(unique_gt_idx) < len(self.data["gt_time"]):
            for k in ["gt_time", "pos", "quat", "velocity"]:
                self.data[k] = self.data[k][unique_gt_idx]

        # 3. Interpolate GT pose & velocity to IMU timestamps
        self.data["gt_orientation"] = self.interp_rot(
            self.data["time"], self.data["gt_time"], self.data["quat"]
        )
        self.data["gt_translation"] = self.interp_xyz(
            self.data["time"], self.data["gt_time"], self.data["pos"]
        )
        self.data["velocity"] = self.interp_xyz(
            self.data["time"], self.data["gt_time"], self.data["velocity"]
        )

        # 4. Convert to torch tensors
        self.data["time"] = torch.tensor(self.data["time"])
        self.data["gt_time"] = torch.tensor(self.data["gt_time"])
        self.data["dt"] = (self.data["time"][1:] - self.data["time"][:-1])[:, None]
        self.data["mask"] = torch.ones(self.data["time"].shape[0], dtype=torch.bool)
        self.data["gyro"] = torch.tensor(self.data["gyro"])
        self.data["acc"] = torch.tensor(self.data["acc"])

        # 5. Optional: override orientation from AirIMU or pre-integration
        self.set_orientation(rot_path, data_name, rot_type)

        # 6. Rotate to requested coordinate frame
        self.update_coordinate(coordinate, mode)

        # 7. Optionally subtract gravity from accelerometer readings
        self.remove_gravity(remove_g)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_imu(self, folder):
        """Load IMU measurements from ``imu/imu.npy``.

        The file stores rows of ``[timestamp_ns, wx, wy, wz, ax, ay, az]``.
        """
        imu_path = os.path.join(folder, "imu", "imu.npy")
        imu_data = np.load(imu_path)          # (N, 7)
        self.data["time"] = imu_data[:, 0] / 1e9   # ns → s
        self.data["gyro"] = imu_data[:, 1:4]        # rad/s
        self.data["acc"] = imu_data[:, 4:7]         # m/s²

    def load_gt(self, folder):
        """Load ground-truth poses from ``pose.txt`` and compute velocity.

        Each line of ``pose.txt`` contains ``tx ty tz qx qy qz qw`` in the
        NED world frame, sampled at the image frame rate (~10 Hz).
        """
        pose_path = os.path.join(folder, "pose.txt")
        poses = np.loadtxt(pose_path, dtype=float)   # (M, 7)

        # TartanAir V2 pose.txt does NOT include timestamps; frames are
        # equally spaced in time.  We recover timestamps from the IMU file
        # (already loaded) by using the image frame rate.
        # Fall back to 10 Hz if we cannot determine it from an adjacent file.
        gt_time = self._load_gt_timestamps(folder, n_frames=poses.shape[0])

        pos = poses[:, :3]                           # (M, 3) tx ty tz
        quat_xyzw = poses[:, 3:7]                   # (M, 4) qx qy qz qw

        # Compute velocity by finite-difference, smoothed with a 5-tap MA filter
        velocity = self._compute_velocity(pos, gt_time)

        self.data["gt_time"] = gt_time
        self.data["pos"] = pos
        self.data["quat"] = quat_xyzw               # stored as xyzw
        self.data["velocity"] = velocity

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _load_gt_timestamps(self, folder, n_frames):
        """Return an array of GT timestamps (seconds) for ``n_frames`` image frames.

        Tries (in order):
        1. ``timestamps.npy`` – absolute timestamps (ns) saved alongside pose.txt
        2. ``imu_time.npy``   – same format, some V2 releases use this name
        3. Falls back to uniform 10 Hz spacing anchored at the first IMU time.
        """
        for fname in ("timestamps.npy", "imu_time.npy"):
            ts_path = os.path.join(folder, fname)
            if os.path.isfile(ts_path):
                ts = np.load(ts_path)
                # Heuristic: if max value exceeds 1e6 seconds (~11 days) the
                # timestamps are almost certainly in nanoseconds.
                if ts.max() > 1e6:
                    ts = ts / 1e9
                return ts[:n_frames]

        # Fallback: uniform 10 Hz, start from IMU t0
        imu_t0 = self.data["time"][0]
        return imu_t0 + np.arange(n_frames) * 0.1

    @staticmethod
    def _compute_velocity(pos, times):
        """Finite-difference velocity with 5-tap moving-average smoothing."""
        dt = np.diff(times)
        # Guard against duplicate timestamps (degenerate data)
        if np.any(dt == 0):
            dt = np.where(dt == 0, np.finfo(float).eps, dt)

        v_start = ((pos[1] - pos[0]) / (times[1] - times[0]) if times[1] != times[0]
                   else np.zeros(3)).reshape(1, 3)
        vel_raw = np.concatenate(
            [v_start, (pos[1:] - pos[:-1]) / dt[:, None]],
            axis=0,
        )
        kernel = np.ones(5) / 5
        vel = np.stack(
            [
                np.convolve(vel_raw[:, i], kernel, mode="same")
                for i in range(3)
            ],
            axis=1,
        )
        return vel

    def interp_rot(self, time, opt_time, quat_xyzw):
        """Spherically interpolate orientations from GT timestamps to IMU timestamps.

        Args:
            time:      IMU timestamps (seconds), shape (N,)
            opt_time:  GT timestamps (seconds),  shape (M,)
            quat_xyzw: GT quaternions [qx, qy, qz, qw], shape (M, 4)

        Returns:
            pp.SO3 tensor of shape (N,)
        """
        # Convert xyzw → wxyz for qinterp
        quat_wxyz = np.zeros_like(quat_xyzw)
        quat_wxyz[:, 0] = quat_xyzw[:, 3]   # w
        quat_wxyz[:, 1:] = quat_xyzw[:, :3] # xyz

        imu_dt = torch.Tensor(time - opt_time[0])
        gt_dt = torch.Tensor(opt_time - opt_time[0])
        quat_wxyz_t = torch.tensor(quat_wxyz)
        quat_interp = qinterp(quat_wxyz_t, gt_dt, imu_dt).double()

        # Store for downstream use (e.g. set_orientation)
        self.data["rot_wxyz"] = quat_interp

        # Convert back to xyzw for pp.SO3
        rot_xyzw = torch.zeros_like(quat_interp)
        rot_xyzw[:, 3] = quat_interp[:, 0]   # w
        rot_xyzw[:, :3] = quat_interp[:, 1:] # xyz
        return pp.SO3(rot_xyzw)

    def interp_xyz(self, time, opt_time, xyz):
        """Linearly interpolate 3-D signals from GT timestamps to IMU timestamps."""
        out = np.stack(
            [np.interp(time, xp=opt_time, fp=xyz[:, i]) for i in range(3)],
            axis=1,
        )
        return torch.tensor(out)

    def get_length(self):
        return self.data["time"].shape[0]

    # ------------------------------------------------------------------
    # Coordinate frame and calibration helpers (mirrors EuRoC / BlackBird)
    # ------------------------------------------------------------------

    def update_coordinate(self, coordinate, mode):
        """Rotate IMU measurements and velocity into the requested frame.

        Args:
            coordinate: ``'glob_coord'`` or ``'body_coord'``
            mode:       dataset mode string (affects velocity rotation)
        """
        if coordinate is None:
            print("No coordinate system provided. Skipping update.")
            return
        try:
            if coordinate == "glob_coord":
                self.data["gyro"] = self.data["gt_orientation"] @ self.data["gyro"]
                self.data["acc"] = self.data["gt_orientation"] @ self.data["acc"]
            elif coordinate == "body_coord":
                self.g_vector = self.data["gt_orientation"].Inv() @ self.g_vector
                if mode not in ("infevaluate", "inference"):
                    self.data["velocity"] = (
                        self.data["gt_orientation"].Inv() @ self.data["velocity"]
                    )
            else:
                raise ValueError(f"Unsupported coordinate system: {coordinate}")
        except Exception as e:
            print("An error occurred while updating coordinates:", e)
            raise

    def set_orientation(self, exp_path, data_name, rotation_type):
        """Optionally override GT orientation with AirIMU or pre-integration result.

        Args:
            exp_path:      path to the pickle file produced by AirIMU inference
            data_name:     key in the pickle file for this sequence
            rotation_type: ``'airimu'``, ``'integration'``, or ``'gtrot'`` / ``None``
        """
        if rotation_type is None or rotation_type == "None" or rotation_type.lower() == "gtrot":
            return
        try:
            with open(exp_path, "rb") as f:
                loaded_data = pickle.load(f)
            state = loaded_data[data_name]
            if rotation_type.lower() == "airimu":
                self.data["gt_orientation"] = state["airimu_rot"]
            elif rotation_type.lower() == "integration":
                self.data["gt_orientation"] = state["inte_rot"]
            else:
                raise ValueError(f"Unsupported rotation type: {rotation_type}")
        except FileNotFoundError:
            print(f"The file {exp_path} was not found.")
            raise

    def remove_gravity(self, remove_g):
        if remove_g is True:
            print("gravity has been removed")
            self.data["acc"] -= self.g_vector
