# 训练结果

**语言：** [English](training_results.md) | 简体中文 · [返回 README](../README_zh.md)

## 纸巾盒训练对比

以下数据比较两种实现在**同一台 RTX 4090** 上各自完成的一次纸巾盒训练，均使用 seed 42、4096 个环境和约 250M environment steps。Uni-FlashCHORD 使用 CUDA MPS 和延迟接触发布；FlashCHORD 使用上游 JAX 训练路径。这些数字是端到端实现测量，不是针对任一优化项的受控消融实验。

| 指标 | Uni-FlashCHORD | FlashCHORD |
| --- | ---: | ---: |
| Environment steps | 249,999,360 | 250,003,456 |
| 首末标量之间的训练时间 | 1 小时 43 分 51 秒 | 1 小时 50 分 40 秒 |
| 同一时间跨度内的平均吞吐 | 40,039 env steps/s | 37,635 env steps/s |
| 最终记录的 episode return | 493.81 | 460.51 |
| 最终记录的完整 episode 长度 | 420.53 control steps | 390.90 control steps |

训练时间取自 TensorBoard 标量时间戳；平均吞吐为 environment steps 除以同一时间跨度；最终 return 和 episode 长度是最后一条日志值，不是多随机种子平均值。下图仅为可读性使用 25 点滑动平均。

![训练 return、episode 长度、wall time 和吞吐对比](media/training_comparison.png)

这是单次运行对比，不能用于证明统计显著性，也不表示与参考实现达到逐值一致。

## 最终策略可视化

GIF 左右两侧分别使用上述最终 checkpoint，在相同纸巾盒参考序列和相机下执行确定性单环境 Viser rollout。它们仅用于定性展示，不是聚合评估指标。

![Uni-FlashCHORD 与 FlashCHORD 最终策略对比](media/policy_comparison.gif)
