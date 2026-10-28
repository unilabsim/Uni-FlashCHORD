# CHORD_UniLab

![CHORD_UniLab demo](docs/media/tissue_demo.gif)

CHORD_UniLab 是基于 **UniLab + UniSim Newton** 的 FlashCHORD 灵巧手 reference
retargeting 复现项目。它将 ManoSharpaData 参考序列重定向到悬浮双 Sharpa 手与物体场景，
使用 FlashSAC 进行参考跟踪训练，并提供物理 replay 和 checkpoint 可视化。

**[安装](#安装) → [示例数据](#示例数据) → [Replay](#replay) → [训练](#训练) → [查看策略](#查看策略) · [配置](#配置)**

主要能力：

- 直接读取 parquet，自动解析物体、支持面、碰撞资产并执行 CoACD；
- 4096 个 Newton 并行环境，20 Hz 控制 / 100 Hz 物理；
- 异步 Torch FlashSAC、n-step replay、VOC 课程和 checkpoint 恢复；
- kinematic、VOC=0、VOC=1 三种 replay 模式；
- 四环境 checkpoint 展示、单环境切换和自动循环播放。

当前实现已验证真实训练、replay 和 checkpoint 可视化链路，但不代表已经完成完整
250M-step 收敛验收，也不宣称与参考实现逐值一致。

## 安装

要求：

- Linux 和支持 Newton 的 NVIDIA CUDA 环境；
- Python 3.11–3.13；
- [uv](https://docs.astral.sh/uv/) 和 Git LFS；
- 正式训练建议使用 48 GiB 级 GPU。

```bash
git clone https://github.com/unilabsim/CHORD_UniLab.git
cd CHORD_UniLab
git lfs pull
uv sync
```

Git LFS 用于下载纸巾盒 parquet、Sharpa mesh 和其他资产。UniSim 使用正式发布版本；
UniLab 暂通过完整上游 commit pin 获取，均由 uv 自动处理，无需手动克隆其他仓库。

可视化入口默认使用 `http://localhost:8080`。远程训练时可通过 SSH 转发该端口；
关闭浏览器标签页不会停止服务，请在终端按 **Ctrl+C** 退出后再启动下一个 viewer。

## 示例数据

仓库自带完整的 `tissue_box_simple` 纸巾盒参考序列和配套资产。在仓库根目录设置：

```bash
PARQUET="$PWD/data/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave"
```

后续 replay、训练和 checkpoint 查看命令都复用该变量。

使用自己的数据时，将 `PARQUET` 改为包含目标 parquet 的目录：

```bash
PARQUET="/absolute/path/to/robot_name=sharpa_wave"
```

本仓库直接消费 **ManoSharpaData parquet 及配套 URDF、mesh 和支持面资产**，不需要复制
大数据集处理脚本。资产转换和 CoACD 缓存会在运行时自动完成。

如果只有原始视频、动捕或重建数据，请先通过外部 `video_to_data/robotic_grounding`
管线生成 parquet 和资产。数据生产由上游管线负责，CHORD_UniLab 负责训练、replay 和评估。

## Replay

```bash
uv run python scripts/replay.py "task.parquet='$PARQUET'" mode=voc1
```

打开 `http://localhost:8080`，点击播放即可查看双手、物体和支持面。

可选模式：

- `kinematic`：逐帧显示参考状态，不进行物理积分；
- `voc0`：启用碰撞，关闭物体 VOC wrench 和 articulation drive；
- `voc1`：启用完整 VOC。

Viser 提供播放、重启、进度和倍速控制。追加 `record.output=/tmp/replay.npz` 可记录
公开状态与控制量。

## 训练

```bash
uv run python -m chord_unilab.scripts.train_flashsac \
  task=sharpa_chord/newton \
  "env.scene.reference_path='$PARQUET'" \
  training.no_play=true
```

这是完整的 **250M-environment-step** 配方，默认使用 4096 个并行环境，每次训练一个
参考序列。切换数据只需修改 `PARQUET`，无需增加数据集专用 YAML 或 Hydra searchpath。

checkpoint 每 2500 iterations 保存一次，默认位于：

```text
logs/flash_sac/SharpaChordFlashSAC/<run>/model_<iteration>.pt
```

## 查看策略

```bash
CHECKPOINT="/absolute/path/to/model.pt"

uv run python scripts/evaluate.py \
  --parquet "$PARQUET" \
  --checkpoint "$CHECKPOINT"
```

默认启动 4 个环境并以 `all` 同时显示，也可切换 `single` 逐个查看。策略从参考帧 0
开始，VOC=0 且无 settling；到达参考序列末尾后自动 reset 并循环播放。

- 使用 Viser 暂停或单步控制检查行为；
- 使用 `--port` 修改端口、`--num-envs` 修改环境数；
- 按 **Ctrl+C** 退出；checkpoint 缺失时会直接报错，不会回退到零动作。

该入口用于确定性策略可视化，不启动训练，也不计算 MPPE、ADD-AUC 等聚合指标。

## 配置

任务使用 Hydra 配置，主要入口如下：

| 内容 | 位置 |
| --- | --- |
| 场景、观测、动作、奖励、reset 和 termination | `src/chord_unilab/conf/flashsac/task/sharpa_chord/base.yaml` |
| Newton、4096 环境和 FlashSAC 配方 | `src/chord_unilab/conf/flashsac/task/sharpa_chord/newton.yaml` |
| 训练、日志和 Viser 默认值 | `src/chord_unilab/conf/flashsac/config.yaml` |
| 数据格式和重采样语义 | `src/chord_unilab/data/README.md` |
| 资产解析、转换和缓存 | `src/chord_unilab/assets/README.md` |

覆盖配置时直接在训练命令后追加 `key=value`，例如：

```bash
algo.seed=123 training.wandb_name=tissue_seed_123
```

## 开发

```bash
make check      # Ruff、format、mypy、pyright
make test       # 完整测试，要求 Git LFS 资产
make test-fast  # 跳过真实 Newton 集成测试
```

真实 Newton 测试默认使用仓库内纸巾盒数据，也可用 `CHORD_TISSUE_PARQUET` 覆盖。

## License

Apache License 2.0。第三方代码来源见 `THIRD_PARTY_NOTICES`。
