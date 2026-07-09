"""Motion-data conversion + audit (single source of truth).

This package consolidates the motion-data processing that used to be split
across ``human-humanoid-tools/scripts`` and the ``booster_mjlab`` repo:

* :mod:`hhtools.dataconvert.mjcf_model` -- load an *arbitrary* MJCF/xml robot
  (joint order, body tree, foot collision geoms, forward kinematics) without
  any per-robot remapping config.
* :mod:`hhtools.dataconvert.csv_io` -- parse the hhtools retarget CSV/PKL export.
* :mod:`hhtools.dataconvert.convert` -- CSV/PKL/NPZ -> canonical mjlab NPZ
  (joint + root state, finite-difference velocities, MuJoCo-FK body world poses).
* :mod:`hhtools.dataconvert.isaaclab_txt` -- CSV/PKL -> booster_isaaclab
  TienKung-style AMP ``.txt`` (joint_pos|joint_vel|end_effector_pos_b), reusing
  the same MuJoCo FK pass.
* :mod:`hhtools.dataconvert.profiles` -- training-export profile registry so a
  single retarget can be emitted for booster_mjlab (NPZ) and booster_isaaclab
  (AMP TXT) without hand-typing joint / body / anchor names.
* :mod:`hhtools.dataconvert.contacts` -- per-frame self-collision / ground
  penetration / contact-force audit (drives the 数据转换 panel overlays).
* :mod:`hhtools.dataconvert.serialize` -- turn an MJCF + NPZ into browser
  payloads (robot GLB + per-frame link transforms + contact markers).
* :mod:`hhtools.dataconvert.speeds` -- root-velocity summary.
* :mod:`hhtools.dataconvert.fullstate` -- Booster ``full_state`` TXT/JSON import.

The robot asset is always resolved to one MuJoCo (MJCF) model that drives FK,
contact detection and both export formats; there is no direct MJCF/URDF
retargeting in this package. ``booster_mjlab`` consumes the NPZ and
``booster_isaaclab`` consumes the AMP TXT this package produces.
"""

from __future__ import annotations

__all__ = [
    "convert",
    "contacts",
    "csv_io",
    "isaaclab_txt",
    "mjcf_model",
    "profiles",
    "serialize",
    "speeds",
]
