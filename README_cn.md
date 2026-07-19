# hhtools

[English README](README.md)

本仓库是在开源工具
[jaggerShen/human-humanoid-tools.git](https://github.com/jaggerShen/human-humanoid-tools.git)
基础上做的项目适配版。核心仍然是人类动作到人形机器人的重映射、预览和批处理流程，
同时加入了适配我们训练框架所需的数据整理能力。

## 主要改动

- 加入动作剪辑工具，用于在训练前截取有效动作片段。
- 加入数据转换工具，可将 CSV、PKL、BVH 等动作来源导出为 MJLab/机器人训练可用格式。
- 加入 相关机器人适配、关节 schema 和训练侧数据接口。
- 保留 Web/CLI 的动作预览、机器人重映射和批量数据检查流程。

## 安装

```bash
uv sync --extra all
```

如果尚未安装 `uv`，可参考 <https://docs.astral.sh/uv/>。

## 常用命令

```bash
uv run hhtools web
uv run hhtools convert --help
uv run hhtools robot --help
```

Web 默认地址为 `http://127.0.0.1:8009`。

## 注意

- SMPL/SMPL-X 等人体模型权重不随仓库分发，需要按许可证自行放到
  `configs/body_models/`。
- 完整第三方动作数据集不随仓库分发，请自行获取后用本工具转换或检查。
- 转换后的 MJLab NPZ 文件可供配套的 my_mjlab 训练仓库使用。

## 许可证

代码按本仓库许可证发布。使用时也请遵守上游数据集、人体模型以及原
human-humanoid-tools 项目的许可证。
