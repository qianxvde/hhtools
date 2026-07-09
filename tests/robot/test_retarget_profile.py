# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Retarget profile defaults."""

from __future__ import annotations

from types import SimpleNamespace

from hhtools.robot.retarget_profile import build_pipeline_config_for_preset


def _preset(**meta):
    return SimpleNamespace(
        meta=meta,
        ik_map={},
        has_urdf=False,
        urdf_path=None,
    )


def test_lafan_bvh_disables_non_pelvis_rotation_objectives() -> None:
    cfg = build_pipeline_config_for_preset(
        _preset(), "lafan_bvh", ik_iterations=24,
    )

    assert cfg.disable_rotation_objectives is True
    assert cfg.pelvis_yaw_only_rotation_target is True


def test_robot_yaml_can_override_lafan_rotation_defaults() -> None:
    cfg = build_pipeline_config_for_preset(
        _preset(
            retarget={
                "references": {
                    "lafan_bvh": {
                        "disable_rotation_objectives": False,
                        "pelvis_yaw_only_rotation_target": False,
                    },
                },
            },
        ),
        "lafan_bvh",
        ik_iterations=24,
    )

    assert cfg.disable_rotation_objectives is False
    assert cfg.pelvis_yaw_only_rotation_target is False
