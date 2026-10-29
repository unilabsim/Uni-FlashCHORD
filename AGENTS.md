# Uni-FlashCHORD Agent Guide

本文件记录长期仓库事实与维护约束。保持简短，不记录当前分支、SHA、PR 状态或
临时验收结果；这些信息由专门文档维护。

## 工作约定

- 方案决策、依赖变更、文件写入和代码改动先讨论并获得明确确认。
- 讨论时给出选项与取舍，由用户决定；例行读取、测试和格式检查除外。
- 保留无关工作，不覆盖、重置或删除他人未提交改动。

## 项目定位与边界

- 目标是在 UniLab + UniSim Newton 上复现 CHORD 的 `sharpa_flash_sac` 路径：
  悬浮双 Sharpa 手、物体和参考动作跟踪 RL。
- `video_to_data/robotic_grounding/flash_chord` 的配置、超参与语义是唯一事实源。
  近似或简化必须在相关 README 或模块 docstring 记录理由，不允许静默偏离。
- 本仓库通过官方 entry point 和公共 API 消费 UniLab environment 与 UniSim 后端；
  禁止 monkey-patch、私有后端访问和上游 CHORD 特判。
- 后端能力归 UniSim；UniLab 只提供 environment、Manager/Entity、资产装配和公共
  playback 适配。FlashSAC、异步 Torch collector/learner、replay、training、logging
  与 IPC 归 CHORD；保留该异步 Torch 架构，不改单进程/JAX。
- RL/training 栈是从 UniLab 一次性迁移的 CHORD-owned 代码，来源见
  `THIRD_PARTY_NOTICES`，不是持续 vendored 同步机制。CHORD 训练入口不得 import
  `unilab.rl` 或 `unilab.scripts`，也不得导入其他 `unilab.training` 模块；唯一例外是
  复用 UniLab 长期保留的 `training.tensor_diagnostics.NanGuard`。后端与环境缺口按
  对应仓库 issue → 修复 → PR 处理，迁移台账和一次性对照工具不进入公开仓库。

## 仓库布局

- `src/chord_unilab/tasks/`：Manager-Based 任务 terms。
- `src/chord_unilab/data/`：parquet 加载与参考数据语义。
- `src/chord_unilab/assets/`：资产转换、组合与传感器声明。静态机器人资产以
  Git LFS 入库，物体和支持面由管线解析；其他生成物不进 Git。
- `src/chord_unilab/conf/` + `src/hydra_plugins/`：owner YAML 与自动配置发现。
- `src/chord_unilab/rl/`：FlashSAC/common、contracts、异步 runtime、distributed、
  off-policy、IPC、off-policy logging 和 CHORD custom actor；不含 APPO/SAC/RSL。
- `src/chord_unilab/training/` + `src/chord_unilab/scripts/`：训练生命周期、checkpoint、
  run metadata、DP/CUDA MPS/NanGuard/ONNX 与 Hydra 入口。
- `src/chord_unilab/replay/` + `scripts/replay.py`：与训练共用装配/reset 的 replay。
- `src/chord_unilab/eval/` + `scripts/evaluate.py`：经 CHORD training playback 的
  checkpoint 评估与可视化。

## 开发约束

- owner 配置按 embodiment 管理，parquet 是运行期输入；不建 per-dataset 配置文件。
- 训练热路径保持 device-resident。step/reset 不得无意引入 NumPy 往返、D2H/H2D
  或同步；确需 host transfer 时必须放在明确边界，并说明性能影响。
- 依赖声明、`uv.lock` 和开发期 override 必须同步更新，不把本地路径当作可复现性保证。
- 命名沿用 UniLab：无时间化命名，一个含义一个公开名，跨模块名不带前导下划线，
  模块名带类型限定而类名不重复。
- 公共接口变更必须同步实现、调用方、测试和文档。每个 task term 在 docstring
  标注消费的 SimBackend 能力。

## 验证与文档

- 常规门禁：`make check` 和 `make test`。修改行为时必须运行真实调用路径，
  不能只验证 import 或 compile。
- 测试使用仓库内资产，不依赖其他 checkout。需要真实 Newton/CUDA 的测试不得
  静默跳过；`make test-fast` 才是显式的快速子集。
- 不用“服务已启动”“单次 step 成功”代替语义验收。记录结果时区分 smoke、
  完整序列、性能测量和收敛证据。
- 架构变化同步根 README 以及 assets/data 模块 README。历史实验、迁移工具和上游
  PR 过程记录保存在仓库外，不作为公开项目文档维护。
