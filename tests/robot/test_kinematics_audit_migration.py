from pathlib import Path

from hhtools.robot.kinematics import (
    infer_ik_map_from_kinematics,
    prepare_ik_map,
    validate_ik_map,
)

_SINGLE_TRUNK_URDF = """\
<robot name="single_trunk">
  <link name="Trunk"/>
  <link name="left_hip_roll_link"/>
  <link name="left_knee_link"/>
  <link name="left_ankle_link"/>
  <link name="right_hip_roll_link"/>
  <link name="right_knee_link"/>
  <link name="right_ankle_link"/>
  <link name="left_shoulder_link"/>
  <link name="left_elbow_link"/>
  <link name="left_wrist_link"/>
  <link name="right_shoulder_link"/>
  <link name="right_elbow_link"/>
  <link name="right_wrist_link"/>
  <joint name="left_hip_roll_joint" type="revolute">
    <parent link="Trunk"/>
    <child link="left_hip_roll_link"/>
  </joint>
  <joint name="left_knee_joint" type="revolute">
    <parent link="left_hip_roll_link"/>
    <child link="left_knee_link"/>
  </joint>
  <joint name="left_ankle_joint" type="revolute">
    <parent link="left_knee_link"/>
    <child link="left_ankle_link"/>
  </joint>
  <joint name="right_hip_roll_joint" type="revolute">
    <parent link="Trunk"/>
    <child link="right_hip_roll_link"/>
  </joint>
  <joint name="right_knee_joint" type="revolute">
    <parent link="right_hip_roll_link"/>
    <child link="right_knee_link"/>
  </joint>
  <joint name="right_ankle_joint" type="revolute">
    <parent link="right_knee_link"/>
    <child link="right_ankle_link"/>
  </joint>
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="Trunk"/>
    <child link="left_shoulder_link"/>
  </joint>
  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_shoulder_link"/>
    <child link="left_elbow_link"/>
  </joint>
  <joint name="left_wrist_joint" type="revolute">
    <parent link="left_elbow_link"/>
    <child link="left_wrist_link"/>
  </joint>
  <joint name="right_shoulder_joint" type="revolute">
    <parent link="Trunk"/>
    <child link="right_shoulder_link"/>
  </joint>
  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_shoulder_link"/>
    <child link="right_elbow_link"/>
  </joint>
  <joint name="right_wrist_joint" type="revolute">
    <parent link="right_elbow_link"/>
    <child link="right_wrist_link"/>
  </joint>
</robot>
"""


def _write_urdf(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_single_rigid_trunk_drops_duplicate_chest(tmp_path: Path) -> None:
    urdf = _write_urdf(tmp_path / "single_trunk.urdf", _SINGLE_TRUNK_URDF)

    inferred = infer_ik_map_from_kinematics(urdf)

    assert inferred["hips"] == "Trunk"
    assert "chest" not in inferred


def test_prepare_ik_map_removes_chest_duplicate_of_hips(tmp_path: Path) -> None:
    urdf = _write_urdf(tmp_path / "single_trunk.urdf", _SINGLE_TRUNK_URDF)

    prepared, changes = prepare_ik_map(urdf, {"hips": "Trunk", "chest": "Trunk"})

    assert prepared["hips"] == "Trunk"
    assert "chest" not in prepared
    assert "chest: removed duplicate of hips ('Trunk')" in changes
    assert not validate_ik_map(urdf, prepared)


def test_batch_import_copies_nested_mesh_tree_before_repair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts import batch_robot_audit

    configs = tmp_path / "configs"
    monkeypatch.setattr(batch_robot_audit, "CONFIGS", configs)

    vendor = tmp_path / "vendor"
    urdf_dir = vendor / "urdf"
    mesh = vendor / "meshes" / "chunk" / "part.stl"
    urdf_dir.mkdir(parents=True)
    mesh.parent.mkdir(parents=True)
    mesh.write_text("solid part\nendsolid part\n", encoding="utf-8")
    urdf = _write_urdf(
        urdf_dir / "bot.urdf",
        """\
<robot name="bot">
      <link name="base">
        <visual>
          <geometry>
            <mesh filename="meshes/chunk/part.stl"/>
          </geometry>
        </visual>
      </link>
</robot>
""",
    )

    dest = configs / "bot"
    dest.mkdir(parents=True)
    (dest / "robot.yaml").write_text("name: bot\n", encoding="utf-8")

    batch_robot_audit._import_robot("bot", urdf, link_meshes=False)

    assert (dest / "meshes" / "chunk" / "part.stl").is_file()
    repaired = (dest / "bot.urdf").read_text(encoding="utf-8")
    assert 'filename="meshes/chunk/part.stl"' in repaired
