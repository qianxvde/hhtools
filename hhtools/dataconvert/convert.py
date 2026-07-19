"""Convert an hhtools retarget export into a canonical mjlab motion NPZ.

This is the single-source replacement for the old
``scripts/convert_hhtools_csv_to_mjlab_npz.py`` (CSV -> NPZ + FK) plus the
my_mjlab ``add_body_states_to_npz.py`` / ``convert_t1_motion_to_body21.py``
scripts. The output contract matches mjlab AMP / motion tracking::

    format_version    scalar str
    fps               scalar float32
    joints_list       (N,)   str          # MJCF joint order
    joint_names       (N,)   str
    joint_pos         (T, N) float32       # MJCF qpos order
    joint_vel         (T, N) float32       # finite-difference of joint_pos
    joint_positions   (T, N) float32       # alias used by some loaders
    root_position     (T, 3) float32
    root_quaternion   (T, 4) float32       # xyzw by default (configurable)
    body_pos_w        (T, B, 3) float32    # MuJoCo FK (optional)
    body_quat_w       (T, B, 4) float32    # wxyz (optional)
    body_names        (B,)   str           # optional

Joint matching is by **name** against the MJCF -- no per-robot remapping config
is required. ``ConvertOptions`` only carries optional sign/offset/axis-flip
overrides for the rare case a source axis convention differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hhtools.dataconvert.csv_io import TrajectorySource, load_trajectory
from hhtools.dataconvert.mjcf_model import MjcfRobot, quat_to_wxyz


@dataclass
class ConvertOptions:
    """Knobs for :func:`convert_trajectory`. All optional; defaults = name-match."""

    output_format: str = "mjlab_motion_v1"
    output_joint_order: tuple[str, ...] | None = None  # default: MJCF joints ∩ clip
    sign: dict[str, float] = field(default_factory=dict)
    offset: dict[str, float] = field(default_factory=dict)
    root_axis_flip: dict[str, int] | None = None
    root_quat_output_order: str = "xyzw"
    compute_body_states: bool = True
    snap_to_ground: bool = False


def convert_trajectory(
    src: TrajectorySource, robot: MjcfRobot, options: ConvertOptions | None = None
) -> dict[str, np.ndarray]:
    """Convert a loaded trajectory + MJCF robot into the canonical payload dict."""
    options = options or ConvertOptions()

    output_order = options.output_joint_order or _default_output_order(robot, src)
    if not output_order:
        raise ValueError(
            "No joints in common between the CSV dof_* names and the MJCF joints. "
            "Check that the MJCF joint names match the retarget export."
        )

    joint_pos = _reorder_dofs(
        src.joint_pos,
        source_order=src.joint_names,
        output_order=output_order,
        sign=options.sign,
        offset=options.offset,
    )

    root_pos = np.asarray(src.root_pos, dtype=np.float64)
    root_quat_xyzw = np.asarray(src.root_quat_xyzw, dtype=np.float64)
    if options.root_axis_flip:
        root_pos, root_quat_xyzw = _apply_axis_flip(root_pos, root_quat_xyzw, options.root_axis_flip)
    root_quat_xyzw = _normalize_quat(root_quat_xyzw)

    fps = float(src.fps)
    joint_vel = _finite_difference(joint_pos, fps)

    # Body-state FK uses MuJoCo's wxyz free-joint quaternion.
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
    body_pos_w: np.ndarray | None = None
    body_quat_w: np.ndarray | None = None
    body_names: list[str] | None = None
    if options.compute_body_states:
        body_pos_w, body_quat_w = robot.fk_body_states(
            root_pos, root_quat_wxyz, joint_pos.astype(np.float64), output_order
        )
        body_names = list(robot.body_names)
        if options.snap_to_ground:
            correction = robot.ground_height_correction(body_pos_w, body_quat_w, body_names)
            if correction is not None:
                root_pos = root_pos.copy()
                root_pos[:, 2] += correction
                root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
                body_pos_w, body_quat_w = robot.fk_body_states(
                    root_pos, root_quat_wxyz, joint_pos.astype(np.float64), output_order
                )

    root_quat_out = (
        root_quat_xyzw if options.root_quat_output_order == "xyzw" else root_quat_wxyz
    )

    _sanity(joint_pos, joint_vel, root_pos, root_quat_out, src.source_path)

    payload: dict[str, np.ndarray] = {
        "format_version": np.asarray(options.output_format),
        "fps": np.asarray(fps, dtype=np.float32),
        "joints_list": np.asarray(output_order),
        "joint_names": np.asarray(output_order),
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
        "joint_positions": joint_pos.astype(np.float32),
        "root_position": root_pos.astype(np.float32),
        "root_quaternion": root_quat_out.astype(np.float32),
        "root_quaternion_order": np.asarray(options.root_quat_output_order),
        "quaternion_order": np.asarray(options.root_quat_output_order),
        "source_file": np.asarray(str(src.source_path)),
        "source_mjcf": np.asarray(str(robot.path)),
        "source_joint_order": np.asarray(tuple(src.joint_names)),
    }
    if body_pos_w is not None:
        payload["body_pos_w"] = body_pos_w
        payload["body_quat_w"] = body_quat_w
        payload["body_names"] = np.asarray(body_names)
        payload["body_quat_order"] = np.asarray("wxyz")
        payload["quat_order"] = np.asarray("wxyz")
    return payload


def convert_file(
    input_path: str | Path,
    mjcf_path: str | Path,
    output_path: str | Path,
    *,
    options: ConvertOptions | None = None,
    fps_override: float | None = None,
) -> dict[str, np.ndarray]:
    """Load a CSV/PKL trajectory, convert against an MJCF, and write the NPZ."""
    src = load_trajectory(input_path, fps_override=fps_override)
    robot = MjcfRobot.from_path(mjcf_path)
    payload = convert_trajectory(src, robot, options)
    save_npz(output_path, payload)
    return payload


def save_npz(output_path: str | Path, payload: dict[str, np.ndarray]) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return output_path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _default_output_order(robot: MjcfRobot, src: TrajectorySource) -> tuple[str, ...]:
    clip = set(src.joint_names)
    return tuple(name for name in robot.joint_names if name in clip)


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
            f"output_joint_order references joints not present in the CSV: {missing}"
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


def _apply_axis_flip(
    root_pos: np.ndarray, root_quat_xyzw: np.ndarray, axis_flip: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation

    signs = np.array(
        [int(axis_flip.get("x", 1)), int(axis_flip.get("y", 1)), int(axis_flip.get("z", 1))],
        dtype=np.float64,
    )
    for s in signs:
        if s not in (-1, 1):
            raise ValueError(f"root_axis_flip values must be +1 or -1, got {s}")
    root_pos = root_pos * signs
    s_mat = np.diag(signs)
    mats = Rotation.from_quat(root_quat_xyzw).as_matrix()
    flipped = s_mat @ mats @ s_mat.T
    return root_pos, Rotation.from_matrix(flipped).as_quat()


def _sanity(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    name: str,
) -> None:
    if not (np.isfinite(joint_pos).all() and np.isfinite(joint_vel).all()):
        raise ValueError(f"{name}: joint_pos/joint_vel contains NaN/Inf")
    if not (np.isfinite(root_pos).all() and np.isfinite(root_quat).all()):
        raise ValueError(f"{name}: root pose contains NaN/Inf")
    norms = np.linalg.norm(root_quat, axis=1)
    if np.max(np.abs(norms - 1.0)) > 1e-3:
        raise ValueError(f"{name}: root quaternions are not unit length")


def npz_payload_summary(payload: dict[str, np.ndarray]) -> dict[str, Any]:
    """Small JSON-friendly summary for the web UI / CLI."""
    jp = payload["joint_pos"]
    fps = float(np.asarray(payload["fps"]).item())
    n = int(jp.shape[0])
    out: dict[str, Any] = {
        "frames": n,
        "fps": fps,
        "duration_s": n / fps if fps else 0.0,
        "num_joints": int(jp.shape[1]),
        "joints_list": [str(x) for x in payload["joints_list"]],
        "has_body_states": "body_pos_w" in payload,
    }
    if "body_pos_w" in payload:
        out["num_bodies"] = int(payload["body_pos_w"].shape[1])
        out["body_names"] = [str(x) for x in payload["body_names"]]
    return out
