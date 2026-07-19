"""Tests for the isaaclab_amp AMP .txt exporter and profile registry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hhtools.dataconvert import profiles as _profiles
from hhtools.dataconvert.csv_io import TrajectorySource
from hhtools.dataconvert.isaaclab_txt import (
    IsaacLabTxtOptions,
    amp_txt_document,
    build_amp_frames,
    write_amp_txt,
)
from hhtools.dataconvert.mjcf_model import MjcfRobot

_MINI_MJCF = """\
<mujoco model="mini">
  <worldbody>
    <body name="Trunk" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="left_hand_link" pos="0.2 0 0">
        <joint name="j_lh" type="hinge" axis="0 0 1"/>
        <geom type="box" size="0.05 0.05 0.05"/>
      </body>
      <body name="right_hand_link" pos="-0.2 0 0">
        <joint name="j_rh" type="hinge" axis="0 0 1"/>
        <geom type="box" size="0.05 0.05 0.05"/>
      </body>
      <body name="left_foot_link" pos="0.1 0 -0.5">
        <joint name="j_lf" type="hinge" axis="0 0 1"/>
        <geom type="box" size="0.05 0.05 0.05"/>
      </body>
      <body name="right_foot_link" pos="-0.1 0 -0.5">
        <joint name="j_rf" type="hinge" axis="0 0 1"/>
        <geom type="box" size="0.05 0.05 0.05"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_JOINT_ORDER = ("j_lh", "j_rh", "j_lf", "j_rf")
_EE_BODIES = ("left_hand_link", "right_hand_link", "left_foot_link", "right_foot_link")


def _mini_robot(tmp_path: Path) -> MjcfRobot:
    path = tmp_path / "mini.xml"
    path.write_text(_MINI_MJCF, encoding="utf-8")
    return MjcfRobot.from_path(path)


def _mini_source(frames: int = 4) -> TrajectorySource:
    t = np.arange(frames)
    root_pos = np.zeros((frames, 3), dtype=np.float64)
    root_pos[:, 2] = 1.0
    root_quat_xyzw = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (frames, 1))
    joint_pos = np.zeros((frames, len(_JOINT_ORDER)), dtype=np.float64)
    return TrajectorySource(
        root_pos=root_pos,
        root_quat_xyzw=root_quat_xyzw,
        joint_pos=joint_pos,
        joint_names=_JOINT_ORDER,
        fps=30.0,
        source_path="<mini>",
    )


def test_frame_layout_and_dim(tmp_path: Path) -> None:
    robot = _mini_robot(tmp_path)
    src = _mini_source()
    options = IsaacLabTxtOptions(joint_order=_JOINT_ORDER, end_effector_bodies=_EE_BODIES)

    frames, fps = build_amp_frames(src, robot, options)

    assert fps == 30.0
    # 2*N joints + 3*E end-effectors
    assert frames.shape == (4, 2 * 4 + 3 * 4)
    assert frames.shape[1] == options.observation_dim()


def test_end_effector_positions_are_base_relative(tmp_path: Path) -> None:
    robot = _mini_robot(tmp_path)
    src = _mini_source()
    options = IsaacLabTxtOptions(joint_order=_JOINT_ORDER, end_effector_bodies=_EE_BODIES)

    frames, _ = build_amp_frames(src, robot, options)

    ee_block = frames[0, 2 * 4 :].reshape(4, 3)
    # identity root rotation => rel_b == world offset from Trunk.
    np.testing.assert_allclose(ee_block[0], [0.2, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(ee_block[1], [-0.2, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(ee_block[2], [0.1, 0.0, -0.5], atol=1e-6)
    np.testing.assert_allclose(ee_block[3], [-0.1, 0.0, -0.5], atol=1e-6)


def test_written_txt_is_valid_amp_json(tmp_path: Path) -> None:
    robot = _mini_robot(tmp_path)
    src = _mini_source(frames=5)
    options = IsaacLabTxtOptions(joint_order=_JOINT_ORDER, end_effector_bodies=_EE_BODIES)

    frames, fps = build_amp_frames(src, robot, options)
    out = write_amp_txt(tmp_path / "clip.txt", frames, fps, options)

    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["LoopMode"] == "Wrap"
    assert doc["FrameDuration"] == pytest.approx(1.0 / 30.0, rel=1e-3)
    assert len(doc["Frames"]) == 5
    assert len(doc["Frames"][0]) == 20
    # document helper matches the on-disk frame count
    assert len(amp_txt_document(frames, fps, options)["Frames"]) == 5


def test_joint_order_mismatch_raises(tmp_path: Path) -> None:
    robot = _mini_robot(tmp_path)
    src = _mini_source()
    options = IsaacLabTxtOptions(joint_order=("does_not_exist",), end_effector_bodies=_EE_BODIES)
    with pytest.raises(ValueError):
        build_amp_frames(src, robot, options)


def test_profiles_registry_contract() -> None:
    # Robot-agnostic: exactly two targets (framework/format), no T1/K1 split.
    ids = {p.id for p in _profiles.list_profiles()}
    assert ids == {"my_mjlab.body_npz", "isaaclab_amp.amp_txt"}
    with pytest.raises(KeyError):
        _profiles.get_profile("nope")

    txt = _profiles.get_profile("isaaclab_amp.amp_txt")
    # joint order is derived from the robot at export time, so the profile
    # itself pins only the 4 AMP end-effectors and no joints.
    assert txt.joint_order == ()
    assert len(txt.end_effector_bodies) == 4
    npz = _profiles.get_profile("my_mjlab.body_npz")
    assert npz.joint_order == ()
    assert npz.anchor_body == "Trunk"


def test_export_with_profile_derives_joint_order_from_robot(tmp_path: Path) -> None:
    robot = _mini_robot(tmp_path)
    src = _mini_source()
    # Registry profile has no joint_order; it must fall back to the robot's
    # actuated joints (j_lh, j_rh, j_lf, j_rf) => 2*4 + 3*4 = 20 obs dims.
    profile = _profiles.get_profile("isaaclab_amp.amp_txt")
    out = tmp_path / "mini.txt"
    summary = _profiles.export_with_profile(src, robot, profile, out)

    assert out.is_file()
    assert summary["observation_dim"] == 20
    assert summary["num_end_effectors"] == 4
    assert summary["profile"] == "isaaclab_amp.amp_txt"
