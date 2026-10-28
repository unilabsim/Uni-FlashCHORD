# 参考数据加载

`ChordReferenceLoader` 直接读取 ManoSharpaData parquet，不需要预先转换成 NPZ。
训练和 replay 将包含 parquet 的目录作为运行期输入：

```python
from chord_unilab.data import load_chord_reference

reference = load_chord_reference(
    "/path/to/robot_name=sharpa_wave",
    control_fps=20.0,
    motion_speed=0.5,
)
```

输入目录可以是 Hive 分区路径；每份 parquet 必须只包含一条参考序列。

## 数据语义

加载结果通过 `ChordReference` 访问：

- 双手腕部位姿、22 维手指关节和命名帧；
- 物体 body 位姿、articulation、mesh 半径和资产路径；
- 每只手 16 个接触槽的位置、法向、part ID 和 active 标志；
- 数据集、序列、机器人、坐标系和源帧率等元数据。

坐标约定为世界系、米制、四元数 `wxyz`，时间轴位于第一维。加载器不额外进行
坐标变换，并校验腕部四元数归一化、接触 part ID 和 parquet 行数。

## 重采样

指定 `control_fps` 和 `motion_speed` 时，位置和关节使用线性插值，四元数使用
SLERP，接触字段使用最近邻采样。`frame_window(start, stop)` 对所有时间轴执行
相同的半开区间切片。

## 资产解析

物体 URDF、mesh 和支持面按 parquet 附近的 `assets` 目录布局解析。资产转换和
缓存由 `chord_unilab.assets` 负责；更换数据集只需更换 parquet 路径，不增加
per-dataset 配置文件。

仓库内 `data/assets/` 提供 `tissue_box_simple` 示例和测试数据闭包。运行完整测试
需要 Git LFS 数据；也可用 `CHORD_TISSUE_PARQUET` 指定另一份纸巾盒测试序列。
