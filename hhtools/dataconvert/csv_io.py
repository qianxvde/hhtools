"""Load an hhtools retarget export (CSV or PKL) into a uniform trajectory.

The retarget / R2R panels export a robot trajectory as either:

* **CSV** -- ``time, root_x, root_y, root_z, root_qx, root_qy, root_qz,
  root_qw, dof_<joint>...`` (root quaternion is **xyzw**). Header ``# k: v``
  comment lines are optional; headerless CSVs are also accepted.
* **PKL** -- a bundle dict whose ``robot`` blob (or the blob itself) holds
  ``joint_q`` ``(F, 7 + N)`` with a **wxyz** root quaternion plus ``dof_names``
  and ``sample_rate``.

Both collapse to :class:`TrajectorySource` so the converter only deals with one
shape: root position, **xyzw** root quaternion, joint positions in clip order.
"""

from __future__ import annotations

import csv as _csv
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TrajectorySource:
    """Robot trajectory normalised for the converter."""

    root_pos: np.ndarray  # (T, 3) float64
    root_quat_xyzw: np.ndarray  # (T, 4) float64
    joint_pos: np.ndarray  # (T, N) float64
    joint_names: tuple[str, ...]
    fps: float
    meta: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])


def load_trajectory(path: str | Path, *, fps_override: float | None = None) -> TrajectorySource:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        src = _load_csv(path)
    elif suffix in (".pkl", ".pickle"):
        src = _load_pkl(path)
    else:
        raise ValueError(f"Unsupported trajectory file {path.name!r} (need .csv or .pkl)")
    if fps_override is not None:
        if fps_override <= 0:
            raise ValueError(f"fps override must be positive, got {fps_override}")
        src.fps = float(fps_override)
    _validate(src)
    return src


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def _load_csv(path: Path) -> TrajectorySource:
    header: list[str] | None = None
    meta: dict[str, str] = {}
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8") as fp:
        reader = _csv.reader(fp)
        for raw in reader:
            if not raw:
                continue
            if raw[0].startswith("#"):
                body = ",".join(raw).lstrip("#").strip()
                if ":" in body:
                    k, _, v = body.partition(":")
                    meta[k.strip()] = v.strip()
                continue
            # A header row begins with the literal "time"; anything else that
            # is non-numeric we also treat as a header. Pure numbers => data.
            if header is None and not _looks_numeric(raw[0]):
                header = list(raw)
                continue
            rows.append(list(raw))

    if not rows:
        raise ValueError(f"{path}: no data rows")
    arr = np.asarray(rows, dtype=np.float64)

    if header is not None:
        if header[0] != "time":
            raise ValueError(f"{path}: expected first column 'time', got {header[0]!r}")
        dof_names = tuple(c[len("dof_"):] for c in header[8:] if c.startswith("dof_"))
        if len(dof_names) != len(header) - 8:
            raise ValueError(f"{path}: non-`dof_` columns after the root block: {header[8:]}")
    else:
        # Headerless: time + 7 root + N dofs, names unknown -> synthesise.
        ndof = arr.shape[1] - 8
        if ndof < 0:
            raise ValueError(f"{path}: headerless CSV needs >= 8 columns, got {arr.shape[1]}")
        dof_names = tuple(f"dof_{i}" for i in range(ndof))

    times = arr[:, 0]
    root_pos = arr[:, 1:4]
    root_quat_xyzw = arr[:, 4:8]
    joint_pos = arr[:, 8:]
    if joint_pos.shape[1] != len(dof_names):
        raise ValueError(
            f"{path}: {joint_pos.shape[1]} dof columns but {len(dof_names)} dof names"
        )

    fps = _fps_from_meta_or_times(meta.get("sample_rate"), times)
    return TrajectorySource(
        root_pos=root_pos,
        root_quat_xyzw=root_quat_xyzw,
        joint_pos=joint_pos,
        joint_names=dof_names,
        fps=fps,
        meta=dict(meta),
        source_path=str(path),
    )


def _looks_numeric(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _fps_from_meta_or_times(sample_rate: str | None, times: np.ndarray) -> float:
    if sample_rate:
        return float(sample_rate)
    if times.shape[0] > 1:
        dt = float(times[1] - times[0])
        if dt > 1e-9:
            return 1.0 / dt
    return 30.0


# ---------------------------------------------------------------------------
# PKL
# ---------------------------------------------------------------------------


def _load_pkl(path: Path) -> TrajectorySource:
    with path.open("rb") as fp:
        blob = pickle.load(fp)
    robot = _find_robot_blob(blob)
    joint_q = np.asarray(robot["joint_q"], dtype=np.float64)
    if joint_q.ndim != 2 or joint_q.shape[1] < 7:
        raise ValueError(f"{path}: robot joint_q has unexpected shape {joint_q.shape}")
    dof_names = tuple(str(n) for n in robot.get("dof_names", ()))
    ndof = joint_q.shape[1] - 7
    if not dof_names:
        dof_names = tuple(f"dof_{i}" for i in range(ndof))
    elif len(dof_names) != ndof:
        # Trim leading free-joint placeholders if present.
        dof_names = dof_names[-ndof:]

    root_pos = joint_q[:, 0:3]
    # PKL stores the root quaternion as wxyz; convert to xyzw.
    quat_wxyz = joint_q[:, 3:7]
    root_quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    joint_pos = joint_q[:, 7:]

    fps = float(robot.get("sample_rate", blob.get("sample_rate", 30.0)) if isinstance(blob, dict) else 30.0)
    meta = {k: v for k, v in (robot.items() if isinstance(robot, dict) else []) if k not in ("joint_q",)}
    return TrajectorySource(
        root_pos=root_pos,
        root_quat_xyzw=root_quat_xyzw,
        joint_pos=joint_pos,
        joint_names=dof_names,
        fps=fps,
        meta={k: v for k, v in meta.items() if isinstance(v, (str, int, float))},
        source_path=str(path),
    )


def _find_robot_blob(blob: Any) -> dict[str, Any]:
    if isinstance(blob, dict):
        if "joint_q" in blob:
            return blob
        if isinstance(blob.get("robot"), dict) and "joint_q" in blob["robot"]:
            return blob["robot"]
    raise ValueError("PKL has no robot blob with a 'joint_q' array")


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def _validate(src: TrajectorySource) -> None:
    if src.num_frames < 2:
        raise ValueError("trajectory needs at least 2 frames")
    if src.fps <= 0:
        raise ValueError(f"non-positive fps {src.fps}")
    if not np.isfinite(src.root_pos).all() or not np.isfinite(src.root_quat_xyzw).all():
        raise ValueError("root pose contains NaN/Inf")
    if not np.isfinite(src.joint_pos).all():
        raise ValueError("joint positions contain NaN/Inf")
