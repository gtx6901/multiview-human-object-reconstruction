# 八视角人—物动态重建

> **2026-10-18 人体管线 v2**：单一 RTMW 检测 + 统一多视角筛选 → 静态参照一次性比例标定 → 统一动作拟合（body→feet→hands→face）。完整 300 帧结果见 [media/final_full300_orbit_008.mp4](media/final_full300_orbit_008.mp4) 与 [方向 B](media/final_full300_orbit_032.mp4)，方法、参数与实测数字见 [docs/PIPELINE_V2.md](docs/PIPELINE_V2.md)。

八台固定相机拍摄同一个人完成“弯腰拿起头盔 → 抬起检视 → 戴到头上调整”的动作（每路 300 帧）。本仓库记录人体（LHM++ 神经高斯）、头盔（自建网格 + UV + 贴图 + 刚体位姿）与八机位联合拟合的实际结果、代码快照和明确限制。

**这是一个研究阶段的归档，不是已验收的重建系统。** 原片、标定数组、模型权重、许可人体资产与私有缓存不在本仓库。

## 直接查看

- [完整 300 帧，两新方向，30fps](media/temporal_body_full300_30fps.mp4) · [10fps 慢放](media/temporal_body_full300_slow10fps.mp4)
- [方向 A：全 300 帧修改前后对照](media/temporal_body_novel0_continuous_before_after.mp4) · [方向 B](media/temporal_body_novel1_continuous_before_after.mp4)
- [方向 A：佩戴段头手近景慢放](media/temporal_body_novel0_worn_hands_slow.mp4) · [方向 B](media/temporal_body_novel1_worn_hands_slow.mp4)
- [20 个跨阶段时刻：新方向 A](media/temporal_body_final_novel0_keyframes.mp4) · [方向 B](media/temporal_body_final_novel1_keyframes.mp4)
- [第 225 帧八真方向头手近景](media/temporal_body_final_head8_0225_generated.jpg)
- [腿足修复前后连续对照](media/grounded_legs_novel_0_before_after.mp4)

以上画面只含生成的新视角，不含原始拍摄。20 时刻审阅视频每个时刻停留一秒，不是连续动作视频。

## 当前数字（同帧输入诊断，不是新视角真值）

| 指标 | 之前 | 之后 |
|---|---:|---:|
| 八机位身体重投影均值（原图像素，1024 宽） | 28.90 | 26.50 |
| 足部重投影 | 12.30 | 11.33 |
| 足部二阶变化 95 分位（模型单位） | 0.00423 | 0.000998 |
| 着地足速 95 分位（模型单位/秒） | 0.0783 | 0.0173 |
| 腋下固定 20,217 像素 sRGB MAE | 0.47619 | 0.45192 |
| 手腕旋转加速度 95 分位 | 145.52 | 26.57 |
| 手指二阶变化 95 分位（rad） | 1.2381 | 0.02437 |
| 佩戴段头盔轮廓同协议 IoU | 0.8115 | 0.8569 |

## 方法

- **人体**：LHM++ 规范高斯 + 固定神经解码器；共享 SMPL-X 体型/尺度 + 逐帧 300 帧动作；腿足、头颈、手部分别加观测与时间约束。
- **头盔**：封闭外壳 + UV + 观测/补全贴图，逐帧刚体位姿；佩戴段由头部动作 + 共享附着变换解释，只保留小幅残差。
- **联合**：共享资产（人体外观/体型、头盔几何/贴图）与逐帧状态分离；全八机位、全时间做可靠观测筛选后交替更新。详见 [设计说明](docs/DESIGN.md)。
- **评估**：输入拟合、跨视角一致性和实际渲染三者分开记录；没有中间视角真值，不把输入误差当新视角精度。

## 已知问题

1. 头盔仍浮在头顶上方，耳侧/后脑头发外露，整体偏浅、有黑斑；内腔与开口是几何先验。
2. 腋下细长暗条仍在；手脸局部遮挡、脸部模型感。
3. 手—头盔接触与非穿透没有进入优化目标。
4. 背包场景的另一位演员只有 A-pose 初始化；现有标定与其对应严重不符。
5. 佩戴段第 179→180 帧存在约 21° 的旋转跳变（连续视频里可见）。

## 仓库内容

```
research/interaction_20261017/   实验脚本快照（含特定片段/目录名，不是通用流水线）
evidence/                        标量证据（当前结果 + 否证记录）
media/                           纯生成的新视角视频与图像
docs/                            设计、决策、复现与第三方说明
tools/verify_release.py          标准库归档完整性校验
environment/                     上游版本与运行环境记录
```

## 复现

从仓库根目录运行 `python tools/verify_release.py`：只核对文件 SHA256、大小、白名单、源码编译和媒体签名，不会下载模型，也不声称重建真人。完整渲染需要本地原始数据、标定、许可资产与 GPU 环境，见 [复现说明](docs/REPRODUCIBILITY.md)。

## 许可

本快照面向非商业研究记录。第三方组件、模型及衍生使用遵循各自上游条款；本仓库不重新许可第三方资产，也未对自有部分额外授予统一开源许可。详见 [第三方说明](docs/THIRD_PARTY_NOTICES.md) 与 [决策记录](docs/DECISIONS.md)。
