"""Serialise an MJCF robot + converted NPZ into browser (three.js) payloads.

The 数据转换 panel reuses the existing ``RobotView`` animator in ``app.js``, whose
contract is:

* ``links``                -- body names.
* ``link_transforms_zero`` -- ``{body: row-major 4x4}`` at the rest pose
  (floating base at identity).
* ``mesh_to_link``         -- ``{glb_node: body}``.
* ``glb_base64``           -- a GLB whose mesh nodes sit at each geom's *world*
  transform for that same rest pose.
* per-frame ``frames[i] = {root: [x,y,z, qx,qy,qz,qw], links: {body: 4x4},
  mesh_z_lift}`` -- body transforms are **relative to the floating base** (root
  identity); the root is applied separately by the animator.

Posing math in the animator: ``mesh_world = root · (T_frame(body) ·
T_zero(body)⁻¹ · baked)`` which equals ``root · T_frame_rel(body) · geom_local``
-- the correct rigid placement.
"""

from __future__ import annotations

import base64
from typing import Any

import mujoco
import numpy as np

from hhtools.dataconvert.mjcf_model import MjcfRobot, compose_mat4, quat_wxyz_to_mat

_MAX_PLAYBACK_FRAMES = 600


# ---------------------------------------------------------------------------
# rest-pose model / GLB
# ---------------------------------------------------------------------------


def _rest_data(robot: MjcfRobot) -> mujoco.MjData:
    """MjData at a sane rest pose (keyframe if any) with root at identity."""
    model = robot.model
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    if robot.has_free_base:
        data.qpos[0:3] = 0.0
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return data


def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


def _geom_trimesh(model: mujoco.MjModel, gid: int):
    """Build a trimesh in the geom-local frame, or ``None`` if unsupported."""
    import trimesh

    gtype = int(model.geom_type[gid])
    size = np.asarray(model.geom_size[gid], dtype=np.float64)
    mesh = None
    if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[gid])
        if mesh_id >= 0:
            vadr = int(model.mesh_vertadr[mesh_id])
            vnum = int(model.mesh_vertnum[mesh_id])
            fadr = int(model.mesh_faceadr[mesh_id])
            fnum = int(model.mesh_facenum[mesh_id])
            verts = np.asarray(model.mesh_vert[vadr : vadr + vnum], dtype=np.float64).reshape(-1, 3)
            faces = np.asarray(model.mesh_face[fadr : fadr + fnum], dtype=np.int64).reshape(-1, 3)
            # MuJoCo's compiler recenters/reorients mesh assets and folds
            # ``mesh_pos`` / ``mesh_quat`` into the geom pose.  ``mesh_vert`` is
            # already in the compiled geom-local frame, so applying mesh_pos again
            # makes visual nodes drift away from their bodies.
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        mesh = trimesh.creation.box(extents=2.0 * size[:3])
    elif gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
    elif gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        mesh = trimesh.creation.capsule(height=2.0 * float(size[1]), radius=float(size[0]))
    elif gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        mesh = trimesh.creation.cylinder(radius=float(size[0]), height=2.0 * float(size[1]))
    elif gtype == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        sph = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        sph.apply_scale(size[:3])
        mesh = sph
    return mesh


def _is_visual_geom(model: mujoco.MjModel, gid: int) -> bool:
    """Render geoms that are not the ground/collision-only set."""
    if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE:
        return False
    if int(model.geom_bodyid[gid]) == 0:  # world body (ground props)
        return False
    rgba = np.asarray(model.geom_rgba[gid], dtype=np.float64)
    # Group 3 is the common "collision" group convention; skip it when other
    # geoms exist on the same body.
    return rgba[3] > 0.0


def serialize_mjcf_robot(robot: MjcfRobot, *, name: str) -> dict[str, Any]:
    model = robot.model
    data = _rest_data(robot)

    links: list[str] = [robot.body_names[i] for i in range(1, model.nbody)]
    link_transforms_zero: dict[str, list[float]] = {}
    for i in range(1, model.nbody):
        T = compose_mat4(
            np.asarray(data.xpos[i], dtype=np.float64),
            quat_wxyz_to_mat(np.asarray(data.xquat[i], dtype=np.float64)),
        )
        link_transforms_zero[robot.body_names[i]] = T.flatten().tolist()

    glb_b64, mesh_to_link, ground_offset_z = _build_glb(robot, data)

    return {
        "name": name,
        "display_name": name,
        "base_link": robot.body_names[1] if model.nbody > 1 else "base",
        "links": links,
        "actuated_joints": list(robot.joint_names),
        "num_dof": len(robot.joint_names),
        "link_transforms_zero": link_transforms_zero,
        "mesh_to_link": mesh_to_link,
        "glb_base64": glb_b64,
        "ground_offset_z": ground_offset_z,
        "joint_names": list(robot.joint_names),
        "body_names": list(robot.body_names),
    }


def _build_glb(robot: MjcfRobot, data: mujoco.MjData):
    try:
        import trimesh
    except Exception:
        return None, {}, 0.0
    model = robot.model
    scene = trimesh.Scene()
    mesh_to_link: dict[str, str] = {}
    min_z: float | None = None
    counts: dict[str, int] = {}
    for gid in range(model.ngeom):
        if not _is_visual_geom(model, gid):
            continue
        try:
            mesh = _geom_trimesh(model, gid)
        except Exception:
            mesh = None
        if mesh is None or mesh.is_empty:
            continue
        bid = int(model.geom_bodyid[gid])
        body = robot.body_names[bid]
        node_transform = compose_mat4(
            np.asarray(data.geom_xpos[gid], dtype=np.float64),
            np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3),
        )
        idx = counts.get(body, 0)
        counts[body] = idx + 1
        node_name = f"{_sanitize(body)}__{idx}"
        rgba = np.asarray(model.geom_rgba[gid], dtype=np.float64)
        try:
            mesh.visual.face_colors = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
        except Exception:
            pass
        scene.add_geometry(mesh, node_name=node_name, geom_name=node_name, transform=node_transform)
        mesh_to_link[node_name] = body
        # track lowest world vertex for the static ground offset
        world_v = mesh.vertices @ node_transform[:3, :3].T + node_transform[:3, 3]
        z = float(world_v[:, 2].min())
        min_z = z if min_z is None else min(min_z, z)

    if not mesh_to_link:
        return None, {}, 0.0
    try:
        glb_bytes = scene.export(file_type="glb")
        glb_b64 = base64.b64encode(glb_bytes).decode("ascii")
    except Exception:
        glb_b64 = None
    ground_offset_z = max(0.0, -min_z) if min_z is not None else 0.0
    return glb_b64, mesh_to_link, ground_offset_z


# ---------------------------------------------------------------------------
# per-frame trajectory
# ---------------------------------------------------------------------------


def serialize_trajectory(
    robot: MjcfRobot,
    payload: dict[str, np.ndarray],
    *,
    max_frames: int = _MAX_PLAYBACK_FRAMES,
) -> dict[str, Any]:
    """Per-frame body transforms (root identity) + the NPZ root, for playback."""
    model = robot.model
    joint_pos = np.asarray(payload["joint_pos"], dtype=np.float64)
    order = tuple(map(str, payload["joints_list"]))
    root_pos = np.asarray(payload["root_position"], dtype=np.float64)
    quat = np.asarray(payload["root_quaternion"], dtype=np.float64)
    quat_order = str(np.asarray(payload.get("root_quaternion_order", "xyzw")).item())
    root_quat_xyzw = quat if quat_order == "xyzw" else quat[:, [1, 2, 3, 0]]
    fps = float(np.asarray(payload["fps"]).item())

    n = int(joint_pos.shape[0])
    indices = _downsample_indices(n, max_frames)

    # Body transforms relative to the floating base: FK with root identity.
    zeros = np.zeros((n, 3), dtype=np.float64)
    ident = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    body_pos_rel, body_quat_rel = robot.fk_body_states(zeros, ident, joint_pos, order)

    links = [robot.body_names[i] for i in range(1, model.nbody)]
    frames: list[dict[str, Any]] = []
    for f in indices:
        link_T: dict[str, list[float]] = {}
        for i in range(1, model.nbody):
            T = compose_mat4(
                body_pos_rel[f, i].astype(np.float64),
                quat_wxyz_to_mat(body_quat_rel[f, i].astype(np.float64)),
            )
            link_T[robot.body_names[i]] = np.round(T, 6).flatten().tolist()
        root = root_pos[f]
        q = root_quat_xyzw[f]
        frames.append(
            {
                "root": [
                    round(float(root[0]), 5),
                    round(float(root[1]), 5),
                    round(float(root[2]), 5),
                    round(float(q[0]), 6),
                    round(float(q[1]), 6),
                    round(float(q[2]), 6),
                    round(float(q[3]), 6),
                ],
                "links": link_T,
                "mesh_z_lift": 0.0,
            }
        )

    return {
        "links": links,
        "frames": frames,
        "frame_indices": indices.tolist(),
        "framerate": fps,
        "sample_rate": fps,
        "num_frames_total": n,
        "playback_frames": len(frames),
        "duration": n / fps if fps else 0.0,
    }


def _downsample_indices(num_frames: int, max_frames: int) -> np.ndarray:
    if max_frames <= 0 or num_frames <= max_frames:
        return np.arange(num_frames, dtype=np.int64)
    return np.unique(np.linspace(0, num_frames - 1, max_frames).round().astype(np.int64))
