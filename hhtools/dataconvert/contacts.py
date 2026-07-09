"""Per-frame self-collision / ground-penetration / contact-force audit.

Generalises the T1/K1-specific ``check_penetration`` + ``_contact_pairs`` from
``scripts/preview_npz.py`` to any MJCF. Two entry points:

* :func:`analyze` -- per-frame contact points + world contact-force vectors +
  an issue classification, used by the 数据转换 panel to draw markers/arrows and
  a frame-by-frame issue strip in the browser.
* :func:`audit` -- a headless text report (replaces ``preview_npz
  --check-penetration`` / ``replay-motion --check``).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from hhtools.dataconvert.mjcf_model import MjcfRobot

_FOOT_HINTS = ("foot", "sole", "toe", "ankle")
_DEFAULT_THRESHOLD = 0.001  # 1 mm
_FOOT_GROUND_PENETRATION = -0.02  # only flag feet below floor by > 2 cm


def _is_ground_geom(model: mujoco.MjModel, geom_id: int) -> bool:
    if geom_id < 0:
        return False
    if int(model.geom_bodyid[geom_id]) == 0:
        return True
    return model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE


def _geom_name(model: mujoco.MjModel, gid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"<geom#{gid}>"


def _body_name(model: mujoco.MjModel, bid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"<body#{bid}>"


def _is_foot_body(name: str) -> bool:
    low = name.lower()
    return any(h in low for h in _FOOT_HINTS)


def _contact_world_force(model: mujoco.MjModel, data: mujoco.MjData, i: int) -> np.ndarray:
    """World-frame contact force (3,) for contact ``i``."""
    force6 = np.zeros(6, dtype=np.float64)
    mujoco.mj_contactForce(model, data, i, force6)
    frame = np.asarray(data.contact[i].frame, dtype=np.float64).reshape(3, 3)
    # rows of `frame` are the contact axes (normal, tangent1, tangent2) in world.
    return frame.T @ force6[:3]


def _frame_contacts(
    model: mujoco.MjModel, data: mujoco.MjData, *, threshold: float
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(data.ncon):
        c = data.contact[i]
        dist = float(c.dist)
        if dist >= threshold:
            continue
        g1, g2 = int(c.geom1), int(c.geom2)
        b1 = int(model.geom_bodyid[g1]) if g1 >= 0 else -1
        b2 = int(model.geom_bodyid[g2]) if g2 >= 0 else -1
        ground1 = _is_ground_geom(model, g1)
        ground2 = _is_ground_geom(model, g2)
        if ground1 and not ground2:
            kind, body, geom = "ground", _body_name(model, b2), _geom_name(model, g2)
        elif ground2 and not ground1:
            kind, body, geom = "ground", _body_name(model, b1), _geom_name(model, g1)
        elif ground1 and ground2:
            kind, body, geom = "ground-ground", "", ""
        else:
            kind = "self"
            body = f"{_body_name(model, b1)} <-> {_body_name(model, b2)}"
            geom = f"{_geom_name(model, g1)} <-> {_geom_name(model, g2)}"
        force = _contact_world_force(model, data, i)
        out.append(
            {
                "kind": kind,
                "body": body,
                "geom": geom,
                "pos": np.asarray(c.pos, dtype=np.float64).copy(),
                "force": force,
                "dist": dist,
            }
        )
    return out


def _set_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    joint_pos: np.ndarray,
    qpos_adr: list[tuple[int, int]],
) -> None:
    data.qpos[:] = 0.0
    data.qpos[0:3] = root_pos
    data.qpos[3:7] = root_quat_wxyz
    for adr, col in qpos_adr:
        data.qpos[adr] = joint_pos[col]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


@dataclass
class _FrameState:
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    joint_pos: np.ndarray
    order: tuple[str, ...]


def _state_from_payload(payload: dict[str, np.ndarray]) -> _FrameState:
    joint_pos = np.asarray(payload["joint_pos"], dtype=np.float64)
    order = tuple(map(str, payload["joints_list"])) if "joints_list" in payload else tuple(
        map(str, payload["joint_names"])
    )
    root_pos = np.asarray(payload["root_position"], dtype=np.float64)
    quat = np.asarray(payload["root_quaternion"], dtype=np.float64)
    quat_order = str(np.asarray(payload.get("root_quaternion_order", "xyzw")).item())
    root_quat_wxyz = quat if quat_order == "wxyz" else quat[:, [3, 0, 1, 2]]
    return _FrameState(root_pos, root_quat_wxyz, joint_pos, order)


def analyze(
    robot: MjcfRobot,
    payload: dict[str, np.ndarray],
    *,
    threshold: float = _DEFAULT_THRESHOLD,
    max_frames: int = 0,
) -> dict[str, Any]:
    """Per-frame contact overlay + issue classification for the browser.

    Returns ``{"frame_indices", "frames": [{contacts, issues}], "summary"}``.
    """
    robot.require_free_base()
    model = robot.model
    data = mujoco.MjData(model)
    st = _state_from_payload(payload)
    qpos_adr = robot.qpos_map(st.order)
    n = int(st.joint_pos.shape[0])
    indices = _downsample_indices(n, max_frames)

    frames: list[dict[str, Any]] = []
    self_total = 0
    ground_pen_total = 0
    nonfoot_total = 0
    for f in indices:
        _set_frame(model, data, st.root_pos[f], st.root_quat_wxyz[f], st.joint_pos[f], qpos_adr)
        contacts = _frame_contacts(model, data, threshold=threshold)
        pts: list[dict[str, Any]] = []
        has_self = has_pen = has_nonfoot = False
        for c in contacts:
            if c["kind"] == "self":
                has_self = True
            elif c["kind"] == "ground":
                if _is_foot_body(c["body"]):
                    if c["dist"] < _FOOT_GROUND_PENETRATION:
                        has_pen = True
                else:
                    has_nonfoot = True
            pts.append(
                {
                    "pos": [round(float(x), 5) for x in c["pos"]],
                    "force": [round(float(x), 4) for x in c["force"]],
                    "kind": c["kind"],
                    "body": c["body"],
                }
            )
        self_total += int(has_self)
        ground_pen_total += int(has_pen)
        nonfoot_total += int(has_nonfoot)
        frames.append(
            {
                "contacts": pts,
                "issues": {
                    "self_collision": has_self,
                    "ground_penetration": has_pen,
                    "non_foot_ground": has_nonfoot,
                },
            }
        )

    return {
        "frame_indices": indices.tolist(),
        "num_frames_total": n,
        "frames": frames,
        "summary": {
            "frames_with_self_collision": self_total,
            "frames_with_ground_penetration": ground_pen_total,
            "frames_with_non_foot_ground": nonfoot_total,
            "clean": self_total == 0 and ground_pen_total == 0 and nonfoot_total == 0,
        },
    }


def audit(
    robot: MjcfRobot, payload: dict[str, np.ndarray], *, threshold: float = _DEFAULT_THRESHOLD
) -> tuple[str, bool]:
    """Headless report. Returns ``(text_report, has_issues)``."""
    robot.require_free_base()
    model = robot.model
    data = mujoco.MjData(model)
    st = _state_from_payload(payload)
    qpos_adr = robot.qpos_map(st.order)
    n = int(st.joint_pos.shape[0])

    self_coll: dict[str, list[int]] = defaultdict(list)
    ground_pen: dict[str, list[int]] = defaultdict(list)
    non_foot: dict[str, list[int]] = defaultdict(list)
    for f in range(n):
        _set_frame(model, data, st.root_pos[f], st.root_quat_wxyz[f], st.joint_pos[f], qpos_adr)
        for c in _frame_contacts(model, data, threshold=threshold):
            if c["kind"] == "self":
                self_coll[c["body"]].append(f)
            elif c["kind"] == "ground":
                if _is_foot_body(c["body"]):
                    if c["dist"] < _FOOT_GROUND_PENETRATION:
                        ground_pen[f"{c['body']} (dist={c['dist']:.3f})"].append(f)
                else:
                    non_foot[f"{c['body']} (dist={c['dist']:.3f})"].append(f)

    lines = ["=" * 70]
    _section(lines, "SELF-COLLISION (unexpected body-body contacts)", self_coll)
    _section(lines, "NON-FOOT BODY TOUCHING GROUND (suspicious)", non_foot)
    _section(lines, "FOOT GROUND PENETRATION (foot below floor > 2cm)", ground_pen)
    lines.append("=" * 70)
    has_issues = bool(self_coll or ground_pen or non_foot)
    lines.append("[FAIL] penetration issues found." if has_issues else "[OK] no penetration issues found.")
    return "\n".join(lines), has_issues


def _section(lines: list[str], title: str, items: dict[str, list[int]]) -> None:
    if items:
        lines.append(f"{title}:")
        for key, frames in items.items():
            lines.append(f"  {key}: {len(frames)} frames (first: {frames[:5]})")
    else:
        lines.append(f"{title}: none")


def _downsample_indices(num_frames: int, max_frames: int) -> np.ndarray:
    if max_frames <= 0 or num_frames <= max_frames:
        return np.arange(num_frames, dtype=np.int64)
    return np.unique(np.linspace(0, num_frames - 1, max_frames).round().astype(np.int64))
