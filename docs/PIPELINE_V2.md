# 人体管线 v2：检测 → 筛选 → 比例 → 统一拟合

本文件记录 2026-10-18 重构后的实际链路、参数和已测数字。旧文档在 `docs/history/`。

## 1. 关键点检测（单一模型）

`research/pipeline_v2/detect_keypoints.py`

- 唯一检测器：RTMW wholebody onnx（288×384，onnxruntime CPU）。
- 检测框：SAM2.1 人物掩码外接框 ×1.15，裁到图内。
- 输入帧：`assets/sam2_all300_v1/<serial>/video_frames/*.jpg`（2048×1536，去畸变）。
- 输出：`assets/keypoints_all300_v1/<serial>.npz`（xy/score/in_image/boxes）。
- 实测：8 机位 × 300 帧 = 2400 次推理，271 秒。

**分数注意**：RTMW 的 SimCC 分数不是跨部位同尺度（身体中位 4.65、脸 7.59、脚 0.78），只能靠统一阈值 + 几何门控，不能直接比数值。

## 2. 统一多视角筛选

`research/pipeline_v2/screen_keypoints.py` → `assets/observations_all300_v1/`

- 规则：分数 ≥0.5 + 图内 + 两两 RANSAC 三角化（8 视角 28 对）+ 内点重投影 ≤10 native px + ≥3 视角 + 射线夹角 ≥1.5°。
- 身体/脚/脸/手共用同一函数。
- 实测（53 秒）：

| 组 | 有效率 | 中位支持 | 内点误差 |
|---|---:|---:|---:|
| 身体 | 91.9% | 4 | 4.73 px |
| 脚 | 93.8% | 3 | 4.70 px |
| 脸 | 68.9% | 3 | 4.77 px |
| 左手 / 右手 | 74.3% / 54.0% | 3 | 4.76 / 4.90 px |

## 3. 脸/手高分辨率方案（结论）

- **脸**：MediaPipe Face Mesh + 整身 RTMW 兜底（`face_source` 标记）。LOO 中位 27.06 → 24.52 px（−9.4%），正脸子集 −13.5%；3DDFA 否决（会在头发/后脑 hallucinate）。
- **手**：保持整身 RTMW。紧裁剪 RTMW、MediaPipe Hands、RTMPose-hand 的多视角一致性都更差（LOO 42.8 / 235.5 / 111.2 vs 25.08）。
- 全量脸：`assets/keypoints_face_hand_all300_v1`（300 帧，每机位 MediaPipe 命中约 85 帧，其余 fallback）。

## 4. 静态参照比例标定

- 输入：同一人纯静态视频 `D:/Document/3dgs/data/organized/fcs/1_static`（2026-08-28 11:20，与动作同场同标定）。
- `research/pipeline_v2/prepare_static_reference.py` 抽取 8 机位 × 3 帧去畸变原图。
- `research/pipeline_v2/detect_static_keypoints.py`：bootstrap 框 + 精化，8.4 秒。
- `research/pipeline_v2/fit_proportions.py`：三角化（f60/f240）→ 六项长度 → 拟合 SMPL-X beta/scale/neck + 肩/肘/踝偏移。

实测残差（应用全部偏移后）：肩宽 +2.4%、右上臂 −1.1%、前臂 +3.2/−2.1%、大腿 +0.6/+0.1%、小腿 +0.4/+1.7%、肩中点→耳 −1.2%；**左上臂 −7.8%** 是左右不对称（左 0.1637 vs 右 0.1566），判定为检测/三角化偏置。

## 5. 统一动作拟合

`research/pipeline_v2/fit_motion.py`（body → feet → hands）

- FK 与 avatar 存储的 `joints55_world` max 误差 0.0。
- 参数：root/body/trans（阶段 A/B），hands（阶段 C）；beta/scale 冻结。
- 损失：三角化 3D + 八视角 2D 重投影 + 时间二阶差分 + **trust-region 先验**（防止偏离已认可初始化）。
- 实测：重投影 33.28 → 30.07 px（33 秒）。

`research/pipeline_v2/fit_face_stage.py`（face）

- 脸部 5 点（鼻/眼/耳）的 head-local 偏移从当前动作学习，只优化颈（body_pose[11]）与头（body_pose[14]）。
- 实测：脸部 5 点重投影 27.19 → 22.10 px。

按帧对比（12 个 COCO 关节、八机位）：

| 指标 | 基线 | 统一拟合+face |
|---|---:|---:|
| 300 帧均值 | 32.80 px | **29.61 px** |
| f0 | 73.91 | 69.26 |
| 变差帧数 | — | **0 / 300** |

## 6. 抖动与平滑（结论：暂不启用后处理）

尝试过三种：

- 轴角分量 RTS → 根旋转偏差 0.9 rad（错误参数化）。
- 6D 分量 RTS → p95 23.6°（扭曲姿态）。
- 流形随机游走 ESKF-RTS（δθ 切空间）→ 抖动 −62~76%，但 f0 回归 +20.9 px；锚定首帧后回归转移到 f5（+19 px）。
- 常加速度 ESKF-RTS → 实现不稳定（回写后姿态崩到 90°）。

结论：**后处理平滑与开头的快速弯腰冲突**，数值抖动下降但真实姿态回归。采用未平滑的统一拟合结果（300/300 帧优于基线）。要真正降抖动，应在估计器内部用批量时间模型（样条/BA 或正确的常加速度滤波），而不是事后滤波。

## 7. 未解决

- 左上臂 +7.8% 的检测偏置。
- ESKF 里偶发的单帧 42° 离群。
- 相机外参审计（sub-agent 进行中）。
- 背包场景未动。
