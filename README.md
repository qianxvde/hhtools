# hhtools

[中文说明](README_cn.md)

This repository is a project-specific fork of the open-source tooling from
[jaggerShen/human-humanoid-tools.git](https://github.com/jaggerShen/human-humanoid-tools.git).
It keeps the core humanoid motion retargeting workflow and adds adapters for our
own training stack.

## What Changed

- Added motion clipping utilities for selecting useful segments before training.
- Added data conversion utilities for exporting training-ready MJLab/robot
  motion files from CSV, PKL, BVH, and related motion sources.
- Added Humanoid-oriented robot adapters and schemas used by our training
  framework.
- Kept the web and CLI workflow for motion preview, robot retargeting, and batch
  dataset inspection.

## Install

```bash
uv sync --extra all
```

If `uv` is not installed, see <https://docs.astral.sh/uv/>.

## Common Commands

```bash
uv run hhtools web
uv run hhtools convert --help
uv run hhtools robot --help
```

The web app runs at `http://127.0.0.1:8009` by default.

## Notes

- SMPL/SMPL-X body model weights are not redistributed. Place your licensed
  model files under `configs/body_models/` when needed.
- Full third-party motion datasets are not redistributed. Use this project to
  convert or inspect locally obtained data.
- Converted MJLab NPZ files can be consumed by the companion my_mjlab
  training repository.

## License

Code is released under the repository license. Please also follow the licenses
of upstream datasets, body models, and the original
human-humanoid-tools project.
