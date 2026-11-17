# 统一重建入口

`python pipeline.py <command> --config <json>` 与安装后的 `hoi` 调用同一实现。场景采用一份 `SceneConfig` 和一个非空、身份名称唯一的 `actors` 列表；单人和多人经过相同阶段。

## 命令与缓存

| 命令 | 实际行为 |
| --- | --- |
| `plan` | 读取配置，列出场景、人物、帧率、输出根和阶段，不执行重建。 |
| `prepare` | 准备相机/帧索引，SAM2 实例分割，RTMW 观测，再构建每人的共享坐标观测包。 |
| `identity` | 优先使用每人的可选 `proportions` 文件，否则从观测估计体型；再以真实可见视角构建 LHM 身份并导出张量 avatar。显式调用会重建身份。 |
| `initialize` | 对请求的实际源帧运行多视角 NLF 初始化，再提取图像手部旋转。初始世界位置由有效身体关节共同估计，不读取已有输出动作。 |
| `fit` | 按人物运行身体/手 → 脸 → 最终四元数 MAP，写入新动作并保存更新后的场景配置。 |
| `refine` | 可选的多视角面部身份细化；拟合共享三维面部位移、Gaussian尺度和外观特征，动作与DPT保持输入值。输出候选，不自动替换原身份。 |
| `render` | 渲染配置中 `actor.avatar` / `actor.motion` 指向的资产，可合成可选刚体。 |
| `run` | 串联上述生产阶段；复用已有部分输入、身份和初始化，再拟合并渲染新动作。 |
| `editor` | 本地场景编辑器：拖动人物、换场景、选机位/动态镜头、调灯光，预览即真实渲染；见 [SCENE.md](SCENE.md)。 |

`run` 在任一人的 manifest、observations 或 keypoints 缺失时执行准备。显式设置 `preparation.end` 时，每人的三维观测和各相机关键点必须与请求的 `0:end:1` 帧序列逐项同序一致；旧三维观测没有 `frames` 时沿用关键点时间轴，并核对三维数组的时间长度。三帧、乱序或多帧缓存都不会冒充所请求的完整片段。

如果需要重建身份、动作初始化或已配置但丢失的手部观测，还会检查 segmentation index；文件缺失或其帧范围与显式请求不符时先执行准备，相对路径按项目根目录解析。已有 avatar 跳过身份重建，NLF 初始化和手部观测分别复用，只补齐缺失的阶段；所有请求输入完整时不依赖 segmentation index。若已提供初始化且未配置手部观测，仍保留该可选输入为空。实例观测和手部观测内部继续使用各自的输入签名。

阶段之间的修改后配置保存在 `runs/<scene>/config.json`。分阶段继续时使用这份配置，避免回到原场景文件中的旧路径：

`render`和可选`refine`保留已有完整阶段配置；渲染输入另写`render_config.json`，身份细化保存自身输入签名。用精简的回放配置渲染，不会将已有重建输入覆盖为空。

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

`configs/reconstruction.json` 当前设置身体阶段 `[400, 200, 200]`、脸部 200 次、最终 MAP 600 次。FCS008两人全部300帧已按这些设置完成拟合和渲染审看。2026-11-16还完成全八相机原片重新准备、两人完整NLF/图像手观测，以及不借旧动作的初始化和完整拟合；八机位0/149/299与正面300帧30fps视频已审看。该轮复用已选身份，不等同于从新受试者原片重新构建身份；抓握接触、脸手外观仍不足。

观测使用 `raw_distorted_pixels` / `undistorted_pixels`，尺寸来自各相机记录。缺失点保留缺失掩码和 NaN，不补成有效观测。各人共享同一世界变换，不能独立重定心或缩放。

NLF提供多视角融合的关节旋转，体型和尺度来自avatar。固定这些量后，将有效的肩、肘、腕、髋、膝、踝三维观测与同名FK关节配准，只优化世界平移；IRLS沿用后续拟合的径向Huber目标与0.02世界单位尺度。资格仍为观测包的`valid`与有限坐标，不增加筛选。手从avatar姿态初始化，再由真实图像手观测拟合；需要复用既有动作时显式设置`initialization`，输出`motion`是否存在不会改变新初始化。

帧号按顺序解码索引解释，不按媒体标头的平均FPS随机定位。准备阶段复用已有逐帧时间轴；缺少时间轴时顺序读取视频的实际帧数和源PTS，保存为本次capture的NPZ，供分割与人物观测共用。自动生成的时间轴按源视频路径、大小和修改时间复用，视频变化后重新测量；不会用旧清单中的声明帧数或平均FPS合成时间。PTS保留其来源；容器时间一致不能单独证明八相机物理同步。

体型拟合仍要求某帧包含全部测量关节。动态片段不满足时会停止；可显式提供来自静态参照或已选身份的 `proportions.pt`，其中 beta/scale 沿同一导出算法使用，不放松测量筛选。

## 可选身份细化与颜色响应

`refine`在既有身份和动作上，用原机位图像进一步拟合当前人物。它与动作拟合的五点脸部姿态阶段用途不同：后者不改变五官形状或外观。所有人物共同渲染，使用同一个DPT解码器；同一相机的RGB响应跨人物、跨所选源时刻共享。参数集中在`configs/reconstruction.json`的`identity_refinement`中，场景同名字典可覆盖数值。

```powershell
python pipeline.py refine --config runs/example/config.json --frames 0 49 98 --updates 10000
```

输入配置必须提供`preparation.segment_index`，各actor的avatar和motion，以及共享标定/manifest。面部网格顶点法向由当前beta的SMPL-X零姿态表面计算；同一顶点的Gaussian共享法向位移，其外观特征和协方差尺度另行拟合。身体身份、动作和DPT冻结，仅为静态身份估计的分工，不改变动态拟合自由度。原图使用实际源帧ID查找mask，缺任一所请求人物的mask时不将该图当作完整监督。采样与解码保留全图上下文，光度与轮廓约束的可见区域来自初始联合光栅，不能通过缩小可见区域逃离监督。

有效相机RGB响应在sRGB线性域作用，包含传感器饱和，再合成背景；其对数跨相机均值为零以固定颜色尺度。它仍与视角反射、原始外观和照明存在歧义，不是独立曝光标定或可重光照模型。亮部已经饱和的原图无法靠反曝光恢复细节。

`identity_refinement/training.pt`是含参数、Adam、RNG与采样队列的单滚动检查点，输入和实现SHA不一致时不混用。输出每人候选avatar及response/report；`--updates`是总目标次数。10000是本工程新实验预算，不是LHM++论文推荐步数。候选需另用未训练源时刻、新视角和连续视频审看，并检查实际几何；图像变锐或损失下降不代表五官三维形状已恢复。

前馈`identity`另支持显式`preparation.color_response` JSON文件，其`linear_rgb_response`为相机名到RGB三通道正响应的映射，定义`observed_linear_RGB = canonical_linear_RGB * response`。校正在原RGB读入后、alpha合成和编码前执行；不配置时保留原像素。响应必须来自明确的拍摄片段，缺失相机不猜默认值。MJJ000与MJJ001已观测到不同响应，不能静默跨片复用。此选项尚在对照验证，过曝输入校正后可能呈灰色，不应解释为恢复了原始亮部。

## 渲染与相机轨迹

所有人体独立蒙皮后拼接 Gaussian，共用经过一致性检查的 decoder；首个人物提供背景 embedding。刚体使用 expected-depth 合成。渲染帧是所有人体和可选刚体实际 motion frames 的交集。

`--cameras <json>` 支持 `{width, height, views: [{label 或 serial, K, W}]}`。K 为 3×3，静态 W 为 4×4；轨迹 W 为 T×4×4，须同时提供对应的实际 `frames`。自定义相机默认 `camera_space: normalized`，也可显式声明 raw。轨迹按帧 ID 查表，不把帧号当数组下标；缺少选中帧时会报错，不插值。

```powershell
python pipeline.py render --config configs/datasets/fcs006.json --cameras assets/scenes/fcs006/fixed_cameras.json --serial all --frames 0
```

`--video` 取配置 FPS，要求选中源帧连续。稀疏帧可以输出 PNG，不能冒充原速视频。视频编码成功后默认删除本次PNG中间帧；需要保留时使用 `--keep-frames`。

静态旋转视角通过场景 `rendering.pose_frame` 或命令行 `--pose-frame <源帧ID>` 指定固定姿态。此时人物和刚体都读取该真实源帧；相机按输出帧 ID 推进。输出时间轴来自显式 `--frames`，否则取所选动态相机轨迹的帧交集；只有静态相机时必须显式给出输出帧。相机不插值，固定源帧必须同时存在于所有人物和刚体中。

未设置 `pose_frame` 时，输出帧仍与原动作源帧一一对应，动态渲染行为不变。报告分别保存 `frames`（输出帧）和 `source_pose_frames`（姿态源帧）；静态旋转可以只有一个姿态源帧却输出完整相机轨迹，不能通过复制动作张量伪造动态片段。静态模式的视频要求输出帧连续，帧率仍由场景配置指定。

场景 `rendering.orbit` 可直接生成旋转相机，无需另写相机脚本。它从实际动作的世界关节范围计算取景，以 `world_up` 为轴、指定 `reference_serial` 为起始方向，输出到 `runs/<scene>/cameras/orbit.json`。可配置 `turns`、`vertical_fov_degrees`、`elevation_degrees`、`margin`；静态模式另用 `frame_count` 指定相机帧数，动态模式沿用真实动作帧 ID。显式 `--cameras` 或 `rendering.cameras` 优先于自动orbit。该路径只移动相机，不改变人体动作或插值姿态；取景依据人体关节范围，较大附属物需自行配置足够的margin。

## 产物与外置依赖

```text
runs/<scene>/
  config.json, runtime.json, render_config.json
  capture/, segmentation/, observations/
  <actor>/inputs/, proportions/, identity/, initialization/, hand_observations/
  <actor>/body/, face/, motion/
  render/camera_<serial>/frames/*.png
  render/report.json

runs/delivery/
  README.md, manifest.json
  selected/<subject>_<task>_<view>.mp4
  views/<subject>_<task>_<view>.mp4
  review/<scene>_<purpose>_camera_<serial>.mp4
  exploration/<scene>_<method>_<view>.mp4
```

所有人体视频统一写入`runs/delivery/`，不再把`preview.mp4`留在各场景目录。默认输出在`review/`；探索分支的衣物、软物体结果写入`exploration/`。名称由实际运行目录用途和机位生成，同一用途原位更新。关键图、日志、配置和报告仍留原运行目录，报告及统一清单记录视频位置。历史报告中的旧视频路径通过清单的`source`字段追溯，不篡改旧报告的运行事实。

审看后使用`pipeline.py export --video <实际文件> --name <snake_case名称> --collection selected`移动选定视频并更新清单；可用`views`归档补充视角。显式`--replace`替换已有同名成片，不保存轮次副本。此处的export只整理本地媒体，不上传任何内容，也不重新训练。

阶段日志写入相应目录的 `run.log`。身份资产为 `identity/avatar.pt`，最终拟合为 `motion/motion.pt`。分割帧、mask、观测和模型张量是本地数据资产，不作为源码提交。

先在匹配的 PyTorch/CUDA 环境中执行 `python -m pip install -e .`。包声明常用Python依赖；LHM++源码、预训练权重、PyTorch3D、SAM2、NLF与本地CUDA扩展仍需外置资源。可选刚体需要nvdiffrast，视频需要FFmpeg。实际环境见 [DEPENDENCIES.md](DEPENDENCIES.md)。

`configs/runtime.local.json` 从 `runtime.example.json` 首次配置，文件被 Git 忽略。`HOI_RUNTIME_CONFIG` 可指定本机配置文件，场景 `runtime` 字段继续覆盖它。运行时键包括 lhm_root、gsplat_extension、ffmpeg、hamer_root、nlf_model、sam2_root、sam2_checkpoint、rtmw_model。权重和第三方源码放在 `lhmpp/`、`runtime/`；已有 Windows 扩展需匹配所用 Python/PyTorch/CUDA，不能仅凭基础包安装成功就判断环境可运行。

## 输出帧率

render/run可用`--fps 30`指定输出帧率；输入观测和拟合仍采用场景原FPS。共享渲染器对55关节局部旋转做最短弧SLERP、对世界平移做线性插值，再用FK重算关节；相机中心与SO(3)姿态、可选刚体轨迹使用同一时钟。双人共享时间网格，固定姿态转台保持同一源姿态。99帧/9fps输出330帧/30fps，时长仍11秒，末源样本剩余显示区间保持末姿态。指定FPS时`--frames`选择新的输出帧ID，报告保留源时钟和插值映射，不覆盖源motion。
