"""Import a ``full_state`` motion clip into the unified trajectory.

Source (``full_state`` JSON) frame layout, ``6 + 2*ndof + 6`` columns::

    [ root_pos(3) | root_euler_XYZ(3) | dof_pos(ndof) | root_lin_vel(3)
      | root_ang_vel(3) | dof_vel(ndof) ]

produced the Euler angles with ``R.from_quat(xyzw).as_euler("XYZ")``,
so we invert with ``R.from_euler("XYZ", euler).as_quat()`` to recover the xyzw
root quaternion. The result is a :class:`TrajectorySource`, so the standard
:func:`hhtools.dataconvert.convert.convert_trajectory` then produces the NPZ
(with MuJoCo-FK body states) against any MJCF.

This replaces my_mjlab's ``convert_amp_motion.py`` /
``convert_tracking_motion.py`` import path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from hhtools.dataconvert.csv_io import TrajectorySource


def _load_frames(path: Path) -> tuple[np.ndarray, float]:
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)
    frames = np.asarray(data["Frames"], dtype=np.float64)
    frame_duration = float(data["FrameDuration"])
    if frame_duration <= 0:
        raise ValueError(f"{path.name}: non-positive FrameDuration {frame_duration}")
    return frames, 1.0 / frame_duration


def load_fullstate(path: str | Path, source_joints: tuple[str, ...]) -> TrajectorySource:
    """Parse a full_state JSON clip into a :class:`TrajectorySource`.

    ``source_joints`` is the joint order the full_state ``dof_pos`` block uses
    (e.g. the robot's full DOF order). Names must match the MJCF you later
    convert against.
    """
    path = Path(path)
    ndof = len(source_joints)
    expected_cols = 6 + 2 * ndof + 6
    frames, fps = _load_frames(path)
    if frames.shape[1] != expected_cols:
        raise ValueError(
            f"{path.name}: expected {expected_cols} columns for {ndof} DoF, "
            f"got {frames.shape[1]}. Is this a full_state clip?"
        )
    root_pos = frames[:, 0:3]
    euler_xyz = frames[:, 3:6]
    dof_pos = frames[:, 6 : 6 + ndof]
    root_quat_xyzw = Rotation.from_euler("XYZ", euler_xyz, degrees=False).as_quat()
    return TrajectorySource(
        root_pos=root_pos,
        root_quat_xyzw=root_quat_xyzw,
        joint_pos=dof_pos,
        joint_names=tuple(source_joints),
        fps=fps,
        meta={"source": "humanoid_full_state"},
        source_path=str(path),
    )
