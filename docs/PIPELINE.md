# 统一重建入口

`python pipeline.py <command> --config <json>` 与安装后的 `hoi` 调用同一实现。场景采用一份 `SceneConfig` 和一个非空、身份名称唯一的 `actors` 列表；单人和多人经过相同阶段。

## 命令与缓存

| 命令 | 实际行为 |
| --- | --- |
| `plan` | 读取配置，列出场景、人物、帧率、输出根和阶段，不执行重建。 |
| `prepare` | 准备相机/帧索引，SAM2 实例分割，RTMW 观测，再构建每人的共享坐标观测包。 |
| `identity` | 优先使用每人的可选 `proportions` 文件，否则从观测估计体型；再以真实可见视角构建 LHM 身份并导出张量 avatar。显式调用会重建身份。 |
| `initialize` | 对请求的实际源帧运行 NLF 初始化，再提取图像手部旋转。已有 `actor.motion` 时将其作为显式源运动先验。 |
| `fit` | 按人物运行身体/手 → 脸 → 最终四元数 MAP，写入新动作并保存更新后的场景配置。 |
| `render` | 渲染配置中 `actor.avatar` / `actor.motion` 指向的资产，可合成可选刚体。 |
| `run` | 串联上述生产阶段；复用已有部分输入、身份和初始化，再拟合并渲染新动作。 |

`run` 只有在任一人的 manifest、observations 或 keypoints 缺失时才执行准备；avatar 文件存在时跳过身份重建，initialization 文件存在时跳过初始化阶段。复用判断主要依据路径是否存在，不能据此断言旧缓存与更改后的输入、配置完全一致。实例观测和手部观测内部另有签名/恢复检查。

阶段之间的修改后配置保存在 `runs/<scene>/config.json`。分阶段继续时使用这份配置，避免回到原场景文件中的旧路径：

```powershell
python pipeline.py prepare --config configs/datasets/fcs008.json
python pipeline.py identity --config runs/fcs008/config.json
python pipeline.py initialize --config runs/fcs008/config.json
python pipeline.py fit --config runs/fcs008/config.json
python pipeline.py render --config runs/fcs008/config.json --serial 21305717 --frames 0
```

这些命令会实际计算。完整串联命令为 `python pipeline.py run --config configs/datasets/fcs008.json`；它复用已提供输入后仍会运行拟合和渲染。

## 配置和输入约定

`configs/datasets/fcs006.json` 指定一人、30 FPS、已归一化相机和头盔资产；`fcs008.json` 指定两人、30 FPS、原始相机外参与各人的既有观测。路径相对项目根解析，差异由配置表达。

主要字段：

- `scene_path`：相机/捕获索引；`manifest_path`：定义共享 world normalization 的 capture manifest。
- `camera_space`：`raw` 或 `normalized`；已归一化的相机平移不能再次施加共享变换。
- `actors`：每人的 id、avatar、motion，以及可复用的 manifest、observations、keypoints、initialization、hand_observations；可选 `proportions` 指定已有体型文件。
- `fps`、`world_up`、`world_up_source`：显式时间和坐标约定。
- `preparation`：身份 seed、视频根、结束帧、身份参考帧，以及准备后保存的 segmentation index。
- `rigid_object`：可选的已有 mesh 和 rigid motion 路径；入口不重建该网格或位姿。
- `rendering`：输出宽高与可选 cameras JSON；命令行尺寸和相机参数可覆盖它。

`configs/reconstruction.json` 当前设置身体阶段 `[400, 200, 200]`、脸部 200 次、最终 MAP 600 次。FCS008两人全部300帧已按这些设置完成拟合和渲染审看；这次复用了已选身份与观测，不等同于全新数据端到端验收。

观测使用 `raw_distorted_pixels` / `undistorted_pixels`，尺寸来自各相机记录。缺失点保留缺失掩码和 NaN，不补成有效观测。各人共享同一世界变换，不能独立重定心或缩放。

体型拟合仍要求某帧包含全部测量关节。动态片段不满足时会停止；可显式提供来自静态参照或已选身份的 `proportions.pt`，其中 beta/scale 沿同一导出算法使用，不放松测量筛选。

## 渲染与相机轨迹

所有人体独立蒙皮后拼接 Gaussian，共用经过一致性检查的 decoder；首个人物提供背景 embedding。刚体使用 expected-depth 合成。渲染帧是所有人体和可选刚体实际 motion frames 的交集。

`--cameras <json>` 支持 `{width, height, views: [{label 或 serial, K, W}]}`。K 为 3×3，静态 W 为 4×4；轨迹 W 为 T×4×4，须同时提供对应的实际 `frames`。自定义相机默认 `camera_space: normalized`，也可显式声明 raw。轨迹按帧 ID 查表，不把帧号当数组下标；缺少选中帧时会报错，不插值。

```powershell
python pipeline.py render --config configs/datasets/fcs006.json --cameras assets/scenes/fcs006/fixed_cameras.json --serial all --frames 0
```

`--video` 取配置 FPS，要求选中源帧连续。稀疏帧可以输出 PNG，不能冒充原速视频。视频编码成功后默认删除本次PNG中间帧；需要保留时使用 `--keep-frames`。

## 产物与外置依赖

```text
runs/<scene>/
  config.json, runtime.json, render_config.json
  capture/, segmentation/, observations/
  <actor>/inputs/, proportions/, identity/, initialization/, hand_observations/
  <actor>/body/, face/, motion/
  render/camera_<serial>/frames/*.png
  render/camera_<serial>/preview.mp4
  render/report.json
```

阶段日志写入相应目录的 `run.log`。身份资产为 `identity/avatar.pt`，最终拟合为 `motion/motion.pt`。分割帧、mask、观测和模型张量是本地数据资产，不作为源码提交。

先在匹配的 PyTorch/CUDA 环境中执行 `python -m pip install -e .`。包声明常用Python依赖；LHM++源码、预训练权重、PyTorch3D、SAM2、NLF与本地CUDA扩展仍需外置资源。可选刚体需要nvdiffrast，视频需要FFmpeg。实际环境见 [DEPENDENCIES.md](DEPENDENCIES.md)。

`configs/runtime.local.json` 从 `runtime.example.json` 首次配置，文件被 Git 忽略。`HOI_RUNTIME_CONFIG` 可指定本机配置文件，场景 `runtime` 字段继续覆盖它。运行时键包括 lhm_root、gsplat_extension、ffmpeg、hamer_root、nlf_model、sam2_root、sam2_checkpoint、rtmw_model。权重和第三方源码放在 `lhmpp/`、`runtime/`；已有 Windows 扩展需匹配所用 Python/PyTorch/CUDA，不能仅凭基础包安装成功就判断环境可运行。
