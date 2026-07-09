"""Load an arbitrary MJCF/xml robot for motion-data conversion + audit.

Everything the conversion / preview pipeline needs is derived from the MJCF
itself: the actuated joint order (so a retarget CSV's ``dof_*`` columns can be
matched by *name*), the body tree (for forward-kinematics body world poses),
and the foot collision geoms (for the optional ground snap). No per-robot YAML
remapping is required.

This generalises the T1/K1-hardcoded helpers that previously lived in
``scripts/preview_npz.py`` and ``scripts/convert_hhtools_csv_to_mjlab_npz.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

_FOOT_GEOM_HINTS = ("foot", "sole", "toe", "heel", "ankle")
_GROUND_CLEARANCE = 0.005


# ---------------------------------------------------------------------------
# Quaternion / rotation helpers (shared across dataconvert modules)
# ---------------------------------------------------------------------------


def quat_to_wxyz(quat: np.ndarray, order: str) -> np.ndarray:
    """Normalise an ``(T, 4)`` quaternion stream to MuJoCo's wxyz convention."""
    q = np.asarray(quat, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quaternion shape must be (T, 4), got {q.shape}")
    if order == "wxyz":
        return q
    if order == "xyzw":
        return q[:, [3, 0, 1, 2]]
    raise ValueError(f"Unsupported quaternion order {order!r}")


def quat_wxyz_to_mat(quat: np.ndarray) -> np.ndarray:
    """Convert unit quaternion(s) wxyz to rotation matrix/matrices."""
    q = np.asarray(quat, dtype=np.float64)
    single = q.ndim == 1
    if single:
        q = q[None, :]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    mat = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    mat[:, 0, 0] = ww + xx - yy - zz
    mat[:, 0, 1] = 2.0 * (xy - wz)
    mat[:, 0, 2] = 2.0 * (xz + wy)
    mat[:, 1, 0] = 2.0 * (xy + wz)
    mat[:, 1, 1] = ww - xx + yy - zz
    mat[:, 1, 2] = 2.0 * (yz - wx)
    mat[:, 2, 0] = 2.0 * (xz - wy)
    mat[:, 2, 1] = 2.0 * (yz + wx)
    mat[:, 2, 2] = ww - xx - yy + zz
    return mat[0] if single else mat


def compose_mat4(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Row-major 4x4 from a 3-vector and a 3x3 rotation matrix."""
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    out[:3, 3] = pos
    return out


# ---------------------------------------------------------------------------
# model loading (floating-base guarantee)
# ---------------------------------------------------------------------------


def _first_joint_is_free(model: mujoco.MjModel) -> bool:
    return int(model.njnt) > 0 and int(model.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE)


def load_free_base_model(path: str | Path) -> mujoco.MjModel:
    """Compile an MJCF/URDF, guaranteeing a floating base at ``qpos[0:7]``.

    The data-convert / FK / contact pipeline replays clips as a free-flying
    robot (root pose in ``qpos[0:7]``). URDFs and fixed-base MJCFs declare no
    free joint, so we add one to the base body via :class:`mujoco.MjSpec` and
    recompile. Models that already float are returned unchanged.
    """
    path = Path(path)
    direct: mujoco.MjModel | None = None
    try:
        direct = mujoco.MjModel.from_xml_path(str(path))
    except Exception:  # noqa: BLE001 - fall through to the MjSpec path
        direct = None
    if direct is not None and _first_joint_is_free(direct):
        return direct

    try:
        spec = mujoco.MjSpec.from_file(str(path))
        has_free = any(
            int(j.type) == int(mujoco.mjtJoint.mjJNT_FREE) for j in spec.joints
        )
        if not has_free:
            top_bodies = list(spec.worldbody.bodies)
            if top_bodies:
                top_bodies[0].add_freejoint()
        return spec.compile()
    except Exception:  # noqa: BLE001
        if direct is not None:
            return direct
        raise


# ---------------------------------------------------------------------------
# MJCF robot wrapper
# ---------------------------------------------------------------------------


@dataclass
class MjcfRobot:
    """A compiled MuJoCo model plus the metadata the converter needs."""

    model: mujoco.MjModel
    path: Path
    joint_names: tuple[str, ...]  # named hinge/slide joints, MJCF order
    body_names: tuple[str, ...]  # all bodies incl. ``world`` at index 0
    has_free_base: bool

    @classmethod
    def from_path(cls, path: str | Path) -> "MjcfRobot":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"MJCF not found: {path}")
        model = load_free_base_model(path)
        return cls.from_model(model, path)

    @classmethod
    def from_model(cls, model: mujoco.MjModel, path: str | Path = "<memory>") -> "MjcfRobot":
        free_types = {int(mujoco.mjtJoint.mjJNT_FREE)}
        joint_names: list[str] = []
        has_free = False
        for ji in range(model.njnt):
            jtype = int(model.jnt_type[ji])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, ji)
            if jtype in free_types:
                has_free = True
                continue
            if name:
                joint_names.append(name)
        body_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
            for i in range(model.nbody)
        )
        return cls(
            model=model,
            path=Path(path),
            joint_names=tuple(joint_names),
            body_names=body_names,
            has_free_base=has_free,
        )

    # -- joint mapping -------------------------------------------------------

    def qpos_map(self, order: tuple[str, ...]) -> list[tuple[int, int]]:
        """Map clip joints -> MJCF qpos addresses.

        Returns ``(qpos_addr, clip_col)`` pairs for every clip joint that the
        MJCF actually has. Raises if the clip references a joint the MJCF lacks.
        """
        model = self.model
        src_index = {name: i for i, name in enumerate(order)}
        model_joint_names: set[str] = set()
        qpos_adr: list[tuple[int, int]] = []
        for ji in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, ji)
            if not name:
                continue
            model_joint_names.add(name)
            if name in src_index:
                qpos_adr.append((int(model.jnt_qposadr[ji]), src_index[name]))
        missing = [n for n in order if n not in model_joint_names]
        if missing:
            raise ValueError(
                f"MJCF {self.path.name} is missing joints present in the clip: "
                f"{missing}. The CSV dof_* names must match MJCF joint names."
            )
        return qpos_adr

    def require_free_base(self) -> None:
        model = self.model
        if model.njnt == 0 or int(model.jnt_type[0]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError(
                f"{self.path.name}: expected the first joint to be a free joint "
                "(floating base) at qpos[0:7]. MJCF must declare a floating base."
            )

    # -- foot geoms (for optional ground snap) -------------------------------

    def foot_geom_specs(self) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Foot collision-box offsets keyed by body name.

        Heuristic: prefer box geoms whose name or body name mentions a foot,
        falling back to any box geom on a foot-named body.
        """
        model = self.model
        specs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for geom_id in range(model.ngeom):
            if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            gname = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").lower()
            body_id = int(model.geom_bodyid[geom_id])
            bname = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "").lower()
            if not any(h in gname or h in bname for h in _FOOT_GEOM_HINTS):
                continue
            body_name = self.body_names[body_id]
            if body_name in specs:
                continue
            specs[body_name] = (
                np.asarray(model.geom_pos[geom_id], dtype=np.float64).copy(),
                np.asarray(model.geom_quat[geom_id], dtype=np.float64).copy(),
                np.asarray(model.geom_size[geom_id], dtype=np.float64).copy(),
            )
        return specs

    def ground_height_correction(
        self, body_pos_w: np.ndarray, body_quat_w: np.ndarray, body_names: list[str]
    ) -> np.ndarray | None:
        """Per-frame root-z offset that puts the lowest foot box at clearance.

        Returns ``None`` when no foot collision boxes are found.
        """
        foot_specs = self.foot_geom_specs()
        if not foot_specs:
            return None
        body_pos = np.asarray(body_pos_w, dtype=np.float64)
        body_quat = np.asarray(body_quat_w, dtype=np.float64)
        bottoms = []
        for body_name, (geom_pos_b, geom_quat_wxyz, geom_size) in foot_specs.items():
            if body_name not in body_names:
                continue
            body_id = body_names.index(body_name)
            body_mat = quat_wxyz_to_mat(body_quat[:, body_id])
            geom_mat = quat_wxyz_to_mat(geom_quat_wxyz)
            center_w = body_pos[:, body_id] + np.einsum("tij,j->ti", body_mat, geom_pos_b)
            geom_rot_w = body_mat @ geom_mat
            extent_z = np.abs(geom_rot_w[:, 2, :]) @ geom_size
            bottoms.append(center_w[:, 2] - extent_z)
        if not bottoms:
            return None
        lowest_bottom = np.min(np.stack(bottoms, axis=1), axis=1)
        return (_GROUND_CLEARANCE - lowest_bottom).astype(np.float64)

    # -- forward kinematics --------------------------------------------------

    def fk_body_states(
        self,
        root_pos: np.ndarray,
        root_quat_wxyz: np.ndarray,
        joint_pos: np.ndarray,
        order: tuple[str, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Replay a clip and return per-body world ``(pos, quat_wxyz)`` arrays.

        ``body_pos_w``: ``(T, nbody, 3)`` float32.
        ``body_quat_w``: ``(T, nbody, 4)`` float32, wxyz.
        """
        self.require_free_base()
        model = self.model
        data = mujoco.MjData(model)
        qpos_adr = self.qpos_map(order)
        t = int(root_pos.shape[0])
        body_pos_w = np.zeros((t, model.nbody, 3), dtype=np.float32)
        body_quat_w = np.zeros((t, model.nbody, 4), dtype=np.float32)
        rp = np.asarray(root_pos, dtype=np.float64)
        rq = np.asarray(root_quat_wxyz, dtype=np.float64)
        jp = np.asarray(joint_pos, dtype=np.float64)
        for frame in range(t):
            data.qpos[:] = 0.0
            data.qpos[0:3] = rp[frame]
            data.qpos[3:7] = rq[frame]
            for adr, col in qpos_adr:
                data.qpos[adr] = jp[frame, col]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            body_pos_w[frame] = data.xpos
            body_quat_w[frame] = data.xquat
        return body_pos_w, body_quat_w
