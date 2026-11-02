# 双人交互当前入口

最新进度与活任务统一见 `../dual_track/LATEST.md`。此处只保留当前代码链。

FCS008，8机位、300帧、30fps；FCS与partner各自身份和人体尺度，统一场景坐标。基线FCS v1.0不改。

## 当前唯一重建入口

`pipeline_adapter.py reconstruct` 顺序执行真实三阶段：

```text
run_quaternion_fit.py   body/feet/hand q + image-hand loss
run_quaternion_face.py  neck/head
run_quaternion_map.py   final 58+face + image-hand loss
```

输出固定覆盖现有目录，不建 `_v2/_final`：
`data/quaternion_fit/<identity>`、`face_<identity>`、`map_<identity>`。
`fit.log` 在每个 stage 目录。

`run-motion`、`run-smooth`、`reference/fit_motion.py`/`smooth_6d.py` 是历史 frozen-AA 兼容工具，不是当前推荐路径。

### 当前实际两人输入（相对当前目录）

FCS：

```powershell
python pipeline_adapter.py reconstruct `
  --person data/persons_full/fcs/person.json `
  --avatar ../../../assets/human/avatar/avatar.pt `
  --initialization data/initialization/nlf_fcs_control/motion.pt `
  --hand-observations data/hand_surface/fcs/observations.npz `
  --out data/quaternion_fit `
  --world-up 0.0497958211609707 -0.9978588791639914 -0.04240322474161076 `
  --world-up-source ../reconstruction/refinement/fcs_feedback/world_up_audit.json
```

Partner：

```powershell
python pipeline_adapter.py reconstruct `
  --person data/persons_full/partner/person.json `
  --avatar assets/partner_identity/avatar_proportioned/avatar.pt `
  --initialization data/initialization/nlf_partner/motion.pt `
  --hand-observations data/hand_surface/partner/observations.npz `
  --out data/quaternion_fit `
  --world-up 0.0497958211609707 -0.9978588791639914 -0.04240322474161076 `
  --world-up-source ../reconstruction/refinement/fcs_feedback/world_up_audit.json
```

## 当前生成链

1. 人工首帧身份标记，SAM2 两实例传播；external bbox + tracker/RTMW 得各身份观测。
2. 同身份多视角 RANSAC-DLT；NLF 只作初始化，translation 用真实 screen 双髋，缺失时只作 prior。
3. `instance_inputs.py observe-hands` 产生逐身份/实际时刻/机位 HaMeR 局部15旋转；`hand_condition.py` 做 SO(3) 视图融合和 OAS 切空间似然。只初始化/约束手指，不采用单目 root/beta/depth。
4. `run_quaternion_fit.py` 初始化并拟合身体/脚/手；`run_quaternion_face.py` 只改头颈；`run_quaternion_map.py` 最终 55 q + translation，以已拟合 motion 为 trust 中心。
5. `joint_render.py` 同一次 Gaussian raster/DPT 渲染两身份；8视角共享动作/几何。

## 对照与限制

两人 full300 三阶段与 front/all8×7 渲染已完成。`data/quaternion_fit/temporal_consistent/{fcs,partner}` 是有意义的时序基线，仅作对照，不是另一条代码分支。图像条件手形已改善；手部交接对齐、外观细节和软包仍有限。当前不是带接触/非穿透项的联合物理拟合；不要把已有深度 alpha 互遮挡误说成没有遮挡。上述状态不代表画质验收。
