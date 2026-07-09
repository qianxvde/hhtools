"""Training-export targets: one retarget, two training frameworks.

The user only picks a **framework/format** — ``booster_mjlab`` (body NPZ) or
``booster_isaaclab`` (AMP TXT). Everything robot-specific (joint order, DOF
count) is **derived from the loaded robot asset at export time**, so there is no
per-robot (T1 vs K1) choice to make: the same target works for any Booster
humanoid because

* the AMP end-effector bodies (``left/right_hand_link`` + ``left/right_foot_link``)
  are named identically across T1 and K1, and
* the AMP joint order is just the robot's own actuated-joint order.

``T1_JOINT_NAMES`` / ``K1_JOINT_NAMES`` are kept only as a reference contract
(mirrors ``booster_isaaclab/legged_lab/envs/booster/booster_cfg.py``) for tests
and validation; they are no longer required to export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hhtools.dataconvert.csv_io import TrajectorySource
from hhtools.dataconvert.mjcf_model import MjcfRobot

FORMAT_NPZ = "body_npz"
FORMAT_AMP_TXT = "amp_txt"

# --- robot joint / body contracts (mirror booster_isaaclab) ----------------

K1_JOINT_NAMES: tuple[str, ...] = (
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch",
    "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch",
    "Right_Ankle_Pitch", "Right_Ankle_Roll",
)

T1_JOINT_NAMES: tuple[str, ...] = (
    "AAHead_yaw", "Head_pitch",
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Waist",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch",
    "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch",
    "Right_Ankle_Pitch", "Right_Ankle_Roll",
)

END_EFFECTOR_BODIES: tuple[str, ...] = (
    "left_hand_link", "right_hand_link", "left_foot_link", "right_foot_link",
)

# Bodies the mjlab AMP discriminator tracks (anchor = Trunk). None => all bodies.
_MJLAB_ANCHOR = "Trunk"


@dataclass(frozen=True)
class TrainingExportProfile:
    """A training target. Robot-specific details are derived at export time.

    ``joint_order`` empty => derive from the loaded robot's actuated joints.
    ``end_effector_bodies`` empty (NPZ) => not used; (TXT) => Booster default.
    """

    id: str
    framework: str
    fmt: str
    label: str
    joint_order: tuple[str, ...] = ()
    end_effector_bodies: tuple[str, ...] = ()
    anchor_body: str = ""
    output_subdir: str = ""
    file_ext: str = ""
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "framework": self.framework,
            "format": self.fmt,
            "label": self.label,
            "end_effector_bodies": list(self.end_effector_bodies),
            "anchor_body": self.anchor_body,
            "output_subdir": self.output_subdir,
            "file_ext": self.default_ext(),
            "notes": self.notes,
        }

    def default_ext(self) -> str:
        return self.file_ext or (".txt" if self.fmt == FORMAT_AMP_TXT else ".npz")


_PROFILES: dict[str, TrainingExportProfile] = {}


def _register(profile: TrainingExportProfile) -> None:
    _PROFILES[profile.id] = profile


_register(TrainingExportProfile(
    id="booster_mjlab.body_npz",
    framework="booster_mjlab",
    fmt=FORMAT_NPZ,
    label="booster_mjlab · NPZ",
    anchor_body=_MJLAB_ANCHOR,
    output_subdir="src/assets/motions",
    file_ext=".npz",
    notes="Body-keypoint NPZ with MuJoCo-FK body_pos_w/body_quat_w (anchor=Trunk).",
))
_register(TrainingExportProfile(
    id="booster_isaaclab.amp_txt",
    framework="booster_isaaclab",
    fmt=FORMAT_AMP_TXT,
    label="booster_isaaclab · AMP TXT",
    end_effector_bodies=END_EFFECTOR_BODIES,
    output_subdir="legged_lab/envs/booster/datasets",
    file_ext=".txt",
    notes="TienKung-style AMP frame: joint_pos|joint_vel|ee_pos_b; joint order from the robot.",
))


def get_profile(profile_id: str) -> TrainingExportProfile:
    try:
        return _PROFILES[profile_id]
    except KeyError:
        raise KeyError(
            f"unknown export profile {profile_id!r}; known: {sorted(_PROFILES)}"
        ) from None


def list_profiles() -> list[TrainingExportProfile]:
    return [_PROFILES[k] for k in sorted(_PROFILES)]


def export_with_profile(
    src: TrajectorySource,
    robot: MjcfRobot,
    profile: TrainingExportProfile,
    output_path: str | Path,
    *,
    compute_body_states: bool = True,
    snap_to_ground: bool = False,
) -> dict[str, Any]:
    """Serialise ``src`` (against ``robot``) into ``profile``'s training format.

    Returns a JSON-friendly summary. NPZ profiles use the canonical mjlab
    converter; AMP-txt profiles use the TienKung-style exporter. Both share the
    single MuJoCo FK pass semantics (no direct MJCF retargeting).
    """
    output_path = Path(output_path)
    if profile.fmt == FORMAT_AMP_TXT:
        from hhtools.dataconvert.isaaclab_txt import (
            DEFAULT_END_EFFECTOR_BODIES,
            IsaacLabTxtOptions,
            amp_txt_summary,
            build_amp_frames,
            write_amp_txt,
        )

        joint_order = profile.joint_order or tuple(robot.joint_names)
        options = IsaacLabTxtOptions(
            joint_order=joint_order,
            end_effector_bodies=profile.end_effector_bodies or DEFAULT_END_EFFECTOR_BODIES,
        )
        frames, fps = build_amp_frames(src, robot, options)
        write_amp_txt(output_path, frames, fps, options)
        summary = amp_txt_summary(frames, fps, options)
        summary["profile"] = profile.id
        summary["path"] = str(output_path)
        return summary

    if profile.fmt == FORMAT_NPZ:
        from hhtools.dataconvert.convert import (
            ConvertOptions,
            convert_trajectory,
            npz_payload_summary,
            save_npz,
        )

        clip = set(src.joint_names)
        output_order = tuple(j for j in profile.joint_order if j in clip) or None
        options = ConvertOptions(
            output_joint_order=output_order,
            compute_body_states=compute_body_states,
            snap_to_ground=snap_to_ground,
        )
        payload = convert_trajectory(src, robot, options)
        save_npz(output_path, payload)
        summary = npz_payload_summary(payload)
        summary["profile"] = profile.id
        summary["path"] = str(output_path)
        return summary

    raise ValueError(f"profile {profile.id!r} has unsupported format {profile.fmt!r}")


__all__ = [
    "FORMAT_AMP_TXT",
    "FORMAT_NPZ",
    "END_EFFECTOR_BODIES",
    "K1_JOINT_NAMES",
    "T1_JOINT_NAMES",
    "TrainingExportProfile",
    "export_with_profile",
    "get_profile",
    "list_profiles",
]
