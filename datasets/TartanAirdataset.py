"""
TartanAir V2 dataset loader adapted to the user's Data_easy structure.

Expected directory structure for each trajectory:
    <data_root>/<data_name>/
        pose_lcam_front.txt          # (M, 7): tx ty tz qx qy qz qw at camera rate
        imu/
            imu_time.npy             # (N,) IMU timestamps in seconds, typically 100 Hz
            cam_time.npy             # (M,) camera/pose timestamps in seconds, typically 10 Hz
            gyro.npy                 # (N, 3) gyroscope measurements in rad/s
            acc.npy                  # (N, 3) accelerometer measurements in m/s^2
            pos_global.npy           # (N, 3) GT position on IMU timeline
            vel_global.npy           # (N, 3) GT velocity in world/global frame on IMU timeline
            vel_body.npy             # (N, 3) GT velocity in body frame on IMU timeline
            parameter.yaml           # contains img_fps / imu_fps

Design choices compared with the previous version:
1. Use the IMU-rate ground truth already provided by the dataset whenever possible.
   - position: pos_global.npy
   - velocity: vel_global.npy or vel_body.npy
2. Only interpolate orientation, because quaternion GT is provided at camera rate
   through pose_lcam_front.txt.
3. Keep the same output keys as the BlackBird loader where possible:
   time, dt, gyro, acc, gt_translation, gt_orientation, velocity, mask.
"""

import os
import pickle
from typing import Optional

import numpy as np
import pypose as pp
import torch

from utils import qinterp
from .dataset import Sequence


class TartanAir(Sequence):
    """Loader for TartanAir V2 trajectories stored in the user's Data_easy format."""

    def __init__(
        self,
        data_root,
        data_name,
        coordinate: Optional[str] = None,
        mode: Optional[str] = None,
        rot_path: Optional[str] = None,
        rot_type: Optional[str] = None,
        gravity: float = 9.81007,
        remove_g: bool = False,
        velocity_frame: str = "global",
        interpolate_orientation: bool = True,
        use_precomputed_position: bool = True,
        use_precomputed_velocity: bool = True,
        acc_source: str = "acc",
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

        self.velocity_frame = velocity_frame.lower()
        if self.velocity_frame not in ("global", "body"):
            raise ValueError("velocity_frame must be 'global' or 'body'.")

        self.interpolate_orientation = interpolate_orientation
        self.use_precomputed_position = use_precomputed_position
        self.use_precomputed_velocity = use_precomputed_velocity
        self.acc_source = acc_source

        # In NED, gravity points along +z.
        self.g_vector = torch.tensor([0, 0, gravity], dtype=torch.double)

        data_path = os.path.join(data_root, data_name)

        # Load sensor data and ground truth.
        self.load_imu(data_path)
        self.load_gt(data_path)

        # Build IMU-timeline tensors.
        self.prepare_aligned_sequences()

        # Optional: override orientation from AirIMU / integration result.
        self.set_orientation(rot_path, data_name, rot_type)

        # Optional: convert into requested coordinate frame.
        self.update_coordinate(coordinate, mode)

        # Optional: subtract gravity from raw acceleration.
        self.remove_gravity(remove_g)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_imu(self, folder):
        """Load IMU measurements already sampled on the IMU timeline."""
        imu_folder = os.path.join(folder, "imu")

        self.data["time"] = np.load(os.path.join(imu_folder, "imu_time.npy")).astype(np.float64)
        self.data["gyro"] = np.load(os.path.join(imu_folder, "gyro.npy")).astype(np.float64)

        acc_map = {
            "acc": "acc.npy",
            "acc_nograv": "acc_nograv.npy",
            "acc_nograv_body": "acc_nograv_body.npy",
        }
        if self.acc_source not in acc_map:
            raise ValueError(
                "acc_source must be one of: 'acc', 'acc_nograv', 'acc_nograv_body'."
            )
        self.data["acc"] = np.load(os.path.join(imu_folder, acc_map[self.acc_source])).astype(np.float64)

    def load_gt(self, folder):
        """Load ground truth from the files already provided in Data_easy."""
        imu_folder = os.path.join(folder, "imu")

        # Camera-rate pose ground truth.
        pose_path = os.path.join(folder, "pose_lcam_front.txt")
        poses = np.loadtxt(pose_path, dtype=np.float64)
        self.data["pose_time"] = np.load(os.path.join(imu_folder, "cam_time.npy")).astype(np.float64)
        self.data["pos_cam"] = poses[:, :3]
        self.data["quat_cam"] = poses[:, 3:7]  # xyzw

        # IMU-rate position GT, already aligned to imu_time.
        if self.use_precomputed_position:
            pos_path = os.path.join(imu_folder, "pos_global.npy")
            if os.path.isfile(pos_path):
                self.data["pos_imu"] = np.load(pos_path).astype(np.float64)
            else:
                self.data["pos_imu"] = self.interp_xyz(
                    self.data["time"], self.data["pose_time"], self.data["pos_cam"]
                ).numpy()
        else:
            self.data["pos_imu"] = self.interp_xyz(
                self.data["time"], self.data["pose_time"], self.data["pos_cam"]
            ).numpy()

        # IMU-rate velocity GT. Prefer provided files over re-differencing pose.
        if self.use_precomputed_velocity:
            if self.velocity_frame == "global":
                vel_file = "vel_global.npy"
            else:
                vel_file = "vel_body.npy"
            vel_path = os.path.join(imu_folder, vel_file)
            if os.path.isfile(vel_path):
                self.data["velocity_imu"] = np.load(vel_path).astype(np.float64)
            else:
                vel_cam = self.compute_velocity(self.data["pos_cam"], self.data["pose_time"])
                self.data["velocity_imu"] = self.interp_xyz(
                    self.data["time"], self.data["pose_time"], vel_cam
                ).numpy()
        else:
            vel_cam = self.compute_velocity(self.data["pos_cam"], self.data["pose_time"])
            self.data["velocity_imu"] = self.interp_xyz(
                self.data["time"], self.data["pose_time"], vel_cam
            ).numpy()

    # ------------------------------------------------------------------
    # Alignment / interpolation helpers
    # ------------------------------------------------------------------

    def prepare_aligned_sequences(self):
        """Create the BlackBird-style aligned output fields."""
        time_np = self.data["time"]

        # Position and velocity are already on IMU timeline (or have been interpolated to it).
        self.data["gt_translation"] = torch.tensor(self.data["pos_imu"], dtype=torch.double)
        self.data["velocity"] = torch.tensor(self.data["velocity_imu"], dtype=torch.double)

        # Orientation is only at camera rate in pose_lcam_front.txt, so interpolate only this part.
        if self.interpolate_orientation:
            self.data["gt_orientation"] = self.interp_rot(
                time_np, self.data["pose_time"], self.data["quat_cam"]
            )
            self.data["gt_time"] = torch.tensor(time_np, dtype=torch.double)
        else:
            # Keep camera-rate orientation if the caller explicitly wants no interpolation.
            # In this branch, gt_time is on the camera timeline and gt_orientation length differs
            # from the IMU sequence length, so coordinate transforms that rely on per-IMU orientation
            # should not be used.
            quat_xyzw = torch.tensor(self.data["quat_cam"], dtype=torch.double)
            self.data["gt_orientation"] = pp.SO3(quat_xyzw)
            self.data["gt_time"] = torch.tensor(self.data["pose_time"], dtype=torch.double)

        self.data["time"] = torch.tensor(time_np, dtype=torch.double)
        self.data["dt"] = (self.data["time"][1:] - self.data["time"][:-1])[:, None]
        self.data["mask"] = torch.ones(self.data["time"].shape[0], dtype=torch.bool)
        self.data["gyro"] = torch.tensor(self.data["gyro"], dtype=torch.double)
        self.data["acc"] = torch.tensor(self.data["acc"], dtype=torch.double)

    @staticmethod
    def compute_velocity(pos, times):
        """Finite-difference velocity with a 5-tap moving-average smoothing."""
        dt = np.diff(times)
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
            [np.convolve(vel_raw[:, i], kernel, mode="same") for i in range(3)],
            axis=1,
        )
        return vel

    def interp_rot(self, time, opt_time, quat_xyzw):
        """Interpolate quaternions from camera timestamps to IMU timestamps."""
        quat_wxyz = np.zeros_like(quat_xyzw)
        quat_wxyz[:, 0] = quat_xyzw[:, 3]
        quat_wxyz[:, 1:] = quat_xyzw[:, :3]

        imu_dt = torch.tensor(time - opt_time[0], dtype=torch.double)
        gt_dt = torch.tensor(opt_time - opt_time[0], dtype=torch.double)
        quat_wxyz_t = torch.tensor(quat_wxyz, dtype=torch.double)
        quat_interp = qinterp(quat_wxyz_t, gt_dt, imu_dt).double()

        self.data["rot_wxyz"] = quat_interp

        rot_xyzw = torch.zeros_like(quat_interp)
        rot_xyzw[:, 3] = quat_interp[:, 0]
        rot_xyzw[:, :3] = quat_interp[:, 1:]
        return pp.SO3(rot_xyzw)

    def interp_xyz(self, time, opt_time, xyz):
        """Linearly interpolate 3-D vectors from one timeline to another."""
        out = np.stack(
            [np.interp(time, xp=opt_time, fp=xyz[:, i]) for i in range(3)],
            axis=1,
        )
        return torch.tensor(out, dtype=torch.double)

    def get_length(self):
        return self.data["time"].shape[0]

    # ------------------------------------------------------------------
    # Coordinate frame and calibration helpers
    # ------------------------------------------------------------------

    def update_coordinate(self, coordinate, mode):
        """Rotate IMU measurements and velocity into the requested frame.

        Notes:
        - Raw gyro/acc are body/IMU-frame measurements.
        - gt_orientation maps body vectors into the global frame.
        - velocity may already be loaded either in global or body frame.
        """
        if coordinate is None:
            print("No coordinate system provided. Skipping update.")
            return

        if self.data["gt_orientation"].shape[0] != self.data["time"].shape[0]:
            raise ValueError(
                "gt_orientation is not on the IMU timeline. "
                "Set interpolate_orientation=True before using coordinate transforms."
            )

        try:
            if coordinate == "glob_coord":
                self.data["gyro"] = self.data["gt_orientation"] @ self.data["gyro"]
                self.data["acc"] = self.data["gt_orientation"] @ self.data["acc"]
                if self.velocity_frame == "body":
                    self.data["velocity"] = self.data["gt_orientation"] @ self.data["velocity"]
                    self.velocity_frame = "global"
            elif coordinate == "body_coord":
                self.g_vector = self.data["gt_orientation"].Inv() @ self.g_vector
                if mode not in ("infevaluate", "inference") and self.velocity_frame == "global":
                    self.data["velocity"] = self.data["gt_orientation"].Inv() @ self.data["velocity"]
                    self.velocity_frame = "body"
            else:
                raise ValueError(f"Unsupported coordinate system: {coordinate}")
        except Exception as e:
            print("An error occurred while updating coordinates:", e)
            raise

    def set_orientation(self, exp_path, data_name, rotation_type):
        """Optionally override GT orientation with AirIMU or pre-integration result."""
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
