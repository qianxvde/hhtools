from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import mujoco
import numpy as np

from hhtools.dataconvert.mjcf_model import MjcfRobot
from hhtools.dataconvert.serialize import serialize_mjcf_robot

_MESH_OFFSET_MJCF = """\
<mujoco model="mesh_offset">
  <asset>
    <mesh
      name="tet"
      vertex="1 0 0  2 0 0  1 1 0  1 0 1"
      face="0 1 2  0 1 3  0 2 3  1 2 3"
    />
  </asset>
  <worldbody>
    <body name="base">
      <freejoint/>
      <geom name="visual" type="mesh" mesh="tet" pos="0.3 0.4 0.5"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_mjcf_glb_mesh_nodes_align_with_mujoco_geom_pose(tmp_path: Path) -> None:
    import trimesh

    xml_path = tmp_path / "robot.xml"
    xml_path.write_text(_MESH_OFFSET_MJCF, encoding="utf-8")

    robot = MjcfRobot.from_path(xml_path)
    payload = serialize_mjcf_robot(robot, name="robot")

    scene = trimesh.load(
        BytesIO(base64.b64decode(payload["glb_base64"])),
        file_type="glb",
        force="scene",
        process=False,
    )
    node_name = next(iter(scene.graph.nodes_geometry))
    mat, geom_name = scene.graph[node_name]
    geom = scene.geometry[geom_name]
    world_vertices = np.asarray(geom.vertices) @ mat[:3, :3].T + mat[:3, 3]

    data = mujoco.MjData(robot.model)
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(robot.model, data)

    np.testing.assert_allclose(world_vertices.mean(axis=0), data.geom_xpos[0], atol=1e-6)
