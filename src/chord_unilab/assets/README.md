# 资产管线

资产分为机器人静态资产和随参考数据变化的场景资产。

## 机器人资产

`robots/sharpawave/` 保存双 Sharpa 手的 MJCF、URDF、网格和物理参数。网格通过
Git LFS 管理，安装或克隆后需要执行 `git lfs pull`。左右手使用同侧网格，腕部采用
3 个 slide 关节与 1 个 ball 关节的浮动根结构。

## 场景资产

物体、支持面和地面由参考 parquet 及其相邻资产解析，不随数据集维护独立配置。
通常不需要直接调用转换脚本，任务和 replay 会通过
`assemble_sharpa_scene()` 完成装配：

1. 从 parquet 解析物体 URDF、mesh、articulation 和支持面；
2. 将 URDF 转为 MJCF，使用 CoACD 生成碰撞凸包；
3. 组合左右手、物体、支持面和地面五个实体；
4. 将跨实体接触排除对传入 `SceneCfg.contact_excludes`。

调试资产转换时可运行：

```bash
uv run python -m chord_unilab.assets.convert_object [parquet]
uv run python -m chord_unilab.assets.convert_support
uv run python -m chord_unilab.assets.convert_floor
```

## 缓存

生成物默认写入 `$CHORD_CACHE_DIR`；未设置时使用 `~/.cache/chord_unilab`：

```text
objects/<key>/       物体 MJCF 与清单
support/<key>/       支持面 MJCF 与清单
floor/<key>/         地面 MJCF 与清单
scene/hands/         左右手实体 XML
convex/<key>/        CoACD NPZ 与保序 OBJ
```

缓存键包含输入资产、物理参数和相关数据语义。生成物不进入 Git；仓库内的
`tissue_box_simple` 数据闭包用于 Demo 和测试。

## 物理约定

- 物体碰撞凸包保持 CoACD 输出顺序，visual mesh 不改变原点或缩放语义。
- 手部直接使用原始 STL，由 Newton 后端构造凸包，不经过离线 CoACD。
- 地面用顶面位于 `z=0` 的薄 box 实体表示，而非世界无限平面。
- 运行时位姿由参考帧 reset 提供，不写入静态场景默认状态。
