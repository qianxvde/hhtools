"""Export a unified trajectory into a Humanoid/TienKung-style AMP ``.txt`` clip.

This is the dual of :mod:`hhtools.dataconvert.fullstate` (which *imports* Humanoid
full-state clips). ``isaaclab_amp`` now consumes a TienKung-style AMP loader
whose per-frame observation is::

    joint_pos (N) | joint_vel (N) | end_effector_pos_b (E*3)

where the end-effector positions are expressed **relative to the floating base,
rotated into the base frame** -- exactly what ``HumanoidEnv.compute_amp_observations``
produces at runtime (``rel_pos_b = quat_apply_inverse(root_quat_w, ee_pos_w -
root_pos_w)``). The frames are wrapped in the JSON document the loader reads::

    {
      "LoopMode": "Wrap",
      "FrameDuration": 1 / fps,
      "EnableCycleOffsetPosition": true,
      "EnableCycleOffsetRotation": true,
      "MotionWeight": 1.0,
      "Frames": [[...], ...]
    }

The end-effector positions come from the *same* MuJoCo forward kinematics the
NPZ exporter uses, so a single FK pass feeds both my_mjlab (NPZ) and
isaaclab_amp (TXT) targets. No IK / direct MJCF retargeting is involved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hhtools.dataconvert.csv_io import TrajectorySource, load_trajectory
from hhtools.dataconvert.mjcf_model import MjcfRobot, quat_wxyz_to_mat

# The AMP discriminator observation carries no root pose; these two hands + two
# feet, in this order, are the T1/K1 humanoid default (12 values = 4 * 3).
DEFAULT_END_EFFECTOR_BODIES: tuple[str, ...] = (
    "left_hand_link",
    "right_hand_link",
    "left_foot_link",
    "right_foot_link",
)


@dataclass
class IsaacLabTxtOptions:
    """Knobs for :func:`build_amp_frames` / :func:`write_amp_txt`."""

    joint_order: tuple[str, ...]
    end_effector_bodies: tuple[str, ...] = DEFAULT_END_EFFECTOR_BODIES
    loop_mode: str = "Wrap"
    motion_weight: float = 1.0
    enable_cycle_offset_position: bool = True
    enable_cycle_offset_rotation: bool = True
    sign: dict[str, float] = field(default_factory=dict)
    offset: dict[str, float] = field(default_factory=dict)
    frame_float_format: str = "%.6f"

    def observation_dim(self) -> int:
        return 2 * len(self.joint_order) + 3 * len(self.end_effector_bodies)


def build_amp_frames(
    src: TrajectorySource, robot: MjcfRobot, options: IsaacLabTxtOptions
) -> tuple[np.ndarray, float]:
    """Return ``(frames (T, 2N + E*3), fps)`` matching the env AMP observation."""
    if not options.joint_order:
        raise ValueError("IsaacLabTxtOptions.joint_order must be non-empty.")
    if not options.end_effector_bodies:
        raise ValueError("IsaacLabTxtOptions.end_effector_bodies must be non-empty.")

    joint_pos = _reorder_dofs(
        src.joint_pos,
        source_order=src.joint_names,
        output_order=options.joint_order,
        sign=options.sign,
        offset=options.offset,
    )
    fps = float(src.fps)
    joint_vel = _finite_difference(joint_pos, fps)

    root_pos = np.asarray(src.root_pos, dtype=np.float64)
    root_quat_xyzw = _normalize_quat(np.asarray(src.root_quat_xyzw, dtype=np.float64))
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]

    body_pos_w, _body_quat_w = robot.fk_body_states(
        root_pos, root_quat_wxyz, joint_pos.astype(np.float64), options.joint_order
    )

    ee_idx = _end_effector_indices(robot, options.end_effector_bodies)
    ee_pos_w = body_pos_w[:, ee_idx, :].astype(np.float64)  # (T, E, 3)
    rel_w = ee_pos_w - root_pos[:, None, :]
    # rel_b = R(root)^T @ rel_w  (world -> base frame), matches quat_apply_inverse.
    root_mat = quat_wxyz_to_mat(root_quat_wxyz)  # (T, 3, 3)
    rel_b = np.einsum("tji,tej->tei", root_mat, rel_w)
    ee_flat = rel_b.reshape(rel_b.shape[0], -1)  # (T, E*3), body-major

    frames = np.concatenate(
        (joint_pos.astype(np.float64), joint_vel.astype(np.float64), ee_flat), axis=1
    )
    _sanity(frames, src.source_path)
    if frames.shape[1] != options.observation_dim():
        raise ValueError(
            f"{src.source_path}: built frame width {frames.shape[1]} != expected "
            f"{options.observation_dim()} (2*{len(options.joint_order)} joints + "
            f"3*{len(options.end_effector_bodies)} end-effectors)."
        )
    return frames, fps


def amp_txt_document(
    frames: np.ndarray, fps: float, options: IsaacLabTxtOptions
) -> dict[str, Any]:
    """JSON-serialisable AMP document (kept as a dict for tests / previews)."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return {
        "LoopMode": options.loop_mode,
        "FrameDuration": 1.0 / float(fps),
        "EnableCycleOffsetPosition": bool(options.enable_cycle_offset_position),
        "EnableCycleOffsetRotation": bool(options.enable_cycle_offset_rotation),
        "MotionWeight": float(options.motion_weight),
        "Frames": frames.tolist(),
    }


def write_amp_txt(
    output_path: str | Path,
    frames: np.ndarray,
    fps: float,
    options: IsaacLabTxtOptions,
) -> Path:
    """Write the AMP JSON ``.txt`` with one frame array per line for readability."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = options.frame_float_format
    lines = [
        "{",
        f'"LoopMode": "{options.loop_mode}",',
        f'"FrameDuration": {1.0 / float(fps):.6f},',
        f'"EnableCycleOffsetPosition": {str(options.enable_cycle_offset_position).lower()},',
        f'"EnableCycleOffsetRotation": {str(options.enable_cycle_offset_rotation).lower()},',
        f'"MotionWeight": {float(options.motion_weight)},',
        "",
        '"Frames":',
        "[",
    ]
    n = frames.shape[0]
    for i in range(n):
        row = ", ".join(fmt % v for v in frames[i])
        sep = "" if i == n - 1 else ","
        lines.append(f"  [{row}]{sep}")
    lines.append("]")
    lines.append("}")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def convert_file_to_amp_txt(
    input_path: str | Path,
    mjcf_path: str | Path,
    output_path: str | Path,
    *,
    options: IsaacLabTxtOptions,
    fps_override: float | None = None,
) -> dict[str, Any]:
    """Load a CSV/PKL trajectory, build AMP frames against an MJCF, write ``.txt``."""
    src = load_trajectory(input_path, fps_override=fps_override)
    robot = MjcfRobot.from_path(mjcf_path)
    frames, fps = build_amp_frames(src, robot, options)
    write_amp_txt(output_path, frames, fps, options)
    return amp_txt_summary(frames, fps, options)


def amp_txt_summary(
    frames: np.ndarray, fps: float, options: IsaacLabTxtOptions
) -> dict[str, Any]:
    """Small JSON-friendly summary for the web UI / CLI."""
    n = int(frames.shape[0])
    return {
        "frames": n,
        "fps": float(fps),
        "duration_s": n / fps if fps else 0.0,
        "observation_dim": int(frames.shape[1]),
        "num_joints": len(options.joint_order),
        "num_end_effectors": len(options.end_effector_bodies),
        "end_effector_bodies": list(options.end_effector_bodies),
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _end_effector_indices(robot: MjcfRobot, ee_bodies: tuple[str, ...]) -> list[int]:
    missing = [b for b in ee_bodies if b not in robot.body_names]
    if missing:
        raise ValueError(
            f"MJCF {robot.path.name} is missing end-effector bodies {missing}. "
            f"Available bodies include: {list(robot.body_names)}"
        )
    return [robot.body_names.index(b) for b in ee_bodies]


def _reorder_dofs(
    dof: np.ndarray,
    *,
    source_order: tuple[str, ...],
    output_order: tuple[str, ...],
    sign: dict[str, float],
    offset: dict[str, float],
) -> np.ndarray:
    src_index = {name: i for i, name in enumerate(source_order)}
    missing = [name for name in output_order if name not in src_index]
    if missing:
        raise ValueError(
            f"joint_order references joints not present in the trajectory: {missing}. "
            "The retarget dof_* names must match the target robot joint names."
        )
    out = np.zeros((dof.shape[0], len(output_order)), dtype=np.float64)
    for j, name in enumerate(output_order):
        col = src_index[name]
        out[:, j] = sign.get(name, 1.0) * dof[:, col] + offset.get(name, 0.0)
    return out


def _finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    if values.shape[0] < 2:
        raise ValueError("need at least 2 frames to compute velocity")
    dt = 1.0 / float(fps)
    vel = np.diff(values, axis=0) / dt
    return np.vstack((vel, vel[-1:]))


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    return quat / np.maximum(norms, 1e-12)


def _sanity(frames: np.ndarray, name: str) -> None:
    if not np.isfinite(frames).all():
        raise ValueError(f"{name}: AMP frames contain NaN/Inf")


__all__ = [
    "DEFAULT_END_EFFECTOR_BODIES",
    "IsaacLabTxtOptions",
    "build_amp_frames",
    "amp_txt_document",
    "write_amp_txt",
    "convert_file_to_amp_txt",
    "amp_txt_summary",
]
