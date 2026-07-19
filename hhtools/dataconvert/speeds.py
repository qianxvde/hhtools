"""Root-velocity summary for a converted motion NPZ (robot-agnostic).

Ported from my_mjlab's ``evaluate_t1_motion_speeds.py`` but without the
T1-specific AMP motion-group routing -- it reports planar speed and yaw rate in
the body frame, useful for sanity-checking a clip before training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class SpeedSummary:
    frames: int
    fps: float
    duration_s: float
    mean_vx_b: float
    mean_vy_b: float
    mean_abs_vy_b: float
    mean_planar_speed: float
    p95_planar_speed: float
    max_planar_speed: float
    mean_abs_yaw_rate: float
    p95_abs_yaw_rate: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _root_quat_xyzw(data: dict[str, np.ndarray]) -> np.ndarray:
    quat = np.asarray(data["root_quaternion"], dtype=np.float64)
    order = str(np.asarray(data.get("root_quaternion_order", "xyzw")).item())
    if order == "wxyz":
        quat = quat[:, [1, 2, 3, 0]]
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    return quat / np.maximum(norms, 1e-12)


def summarize(data: dict[str, np.ndarray]) -> SpeedSummary:
    fps = float(np.asarray(data["fps"]).item())
    if fps <= 0:
        raise ValueError(f"bad fps {fps}")
    root_pos = np.asarray(data["root_position"], dtype=np.float64)
    quat_xyzw = _root_quat_xyzw(data)
    n = int(root_pos.shape[0])
    if n < 2:
        raise ValueError("need >= 2 frames")

    dt = 1.0 / fps
    vel_w = np.diff(root_pos, axis=0) / dt  # (T-1, 3) world
    rot = Rotation.from_quat(quat_xyzw[:-1])
    vel_b = rot.inv().apply(vel_w)  # body frame
    planar = np.linalg.norm(vel_b[:, :2], axis=1)

    # yaw rate from quaternion finite difference
    euler = Rotation.from_quat(quat_xyzw).as_euler("xyz")
    yaw = np.unwrap(euler[:, 2])
    yaw_rate = np.diff(yaw) / dt

    return SpeedSummary(
        frames=n,
        fps=fps,
        duration_s=n / fps,
        mean_vx_b=float(np.mean(vel_b[:, 0])),
        mean_vy_b=float(np.mean(vel_b[:, 1])),
        mean_abs_vy_b=float(np.mean(np.abs(vel_b[:, 1]))),
        mean_planar_speed=float(np.mean(planar)),
        p95_planar_speed=float(np.percentile(planar, 95)),
        max_planar_speed=float(np.max(planar)),
        mean_abs_yaw_rate=float(np.mean(np.abs(yaw_rate))),
        p95_abs_yaw_rate=float(np.percentile(np.abs(yaw_rate), 95)),
    )


def summarize_file(path: str | Path) -> SpeedSummary:
    with np.load(Path(path), allow_pickle=True) as archive:
        data = {k: archive[k] for k in archive.files}
    return summarize(data)
