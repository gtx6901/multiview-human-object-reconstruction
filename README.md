# 多视角人—物动态重建

单人和多人共用 `pipeline.py` / `hoi` 入口，由场景配置指定相机、时间轴、身份与缓存资产。人体链为观测准备 → 身份与体型 → NLF 初始化和图像手部观测 → 四元数身体拟合 → 脸部拟合 → 四元数 MAP → 联合渲染。已有刚体网格和位姿可通过同一渲染器合成。

人体主链与衣物探索分开管理；场景差异放在配置中，版本由Git提交和分支表达。

衣物探索在独立的 [codex/clothing](https://github.com/gtx6901/multiview-human-object-reconstruction/tree/codex/clothing) 分支提供 `pipeline.py clothing`，各阶段共用一份JSON配置；[用法与当前结果](https://github.com/gtx6901/multiview-human-object-reconstruction/blob/codex/clothing/docs/CLOTHING.md) 随该分支维护。

## 安装与本机依赖

在已有 PyTorch/CUDA 环境中安装本项目：

```powershell
python -m pip install -e .
```

这只安装 `pyproject.toml` 声明的基础依赖，不会下载模型或编译 CUDA 扩展。运行重建还需要本地 LHM++ 源码和权重、SMPL-X、SAM2、RTMW、NLF、HaMeR，以及匹配 Python/PyTorch/CUDA 的渲染扩展。头盔网格渲染还需要 nvdiffrast；视频编码需要 ffmpeg。

首次配置可从 `configs/runtime.example.json` 复制一份 `configs/runtime.local.json`，按本机位置填写依赖路径。已有本机配置应直接检查，不覆盖。模型、数据和生成结果均外置，不随源码提交。详细路径见 [管线说明](docs/PIPELINE.md)。

## 使用

从项目根目录运行；安装后也可用 `hoi` 代替 `python pipeline.py`。

```powershell
python pipeline.py plan --config configs/datasets/fcs006.json
python pipeline.py plan --config configs/datasets/fcs008.json
```

`plan` 只读取配置并列出阶段，不执行重建。以下命令回放已有资产，会实际调用 GPU 渲染：

```powershell
python pipeline.py render --config configs/datasets/fcs006.json --serial 21305717 --frames 0
python pipeline.py render --config configs/datasets/fcs008.json --serial 21305717 --frames 0
```

FCS006 配置包含一人和头盔；FCS008 配置包含两人。算法不按这两个场景名称分支。固定新视角或相机轨迹通过 `--cameras <json>` 提供，`--serial all` 使用所选相机文件的全部视角。

已提供以下统一场景配置；首次重建用`run --config configs/datasets/<场景>.json --video`，接续已有阶段用`runs/<场景>/config.json`保留已生成的真实路径。数据、身份提示与标定仍是本地输入。

| 配置 | 原始场景 | 渲染任务 |
| --- | --- | --- |
| fcs003 | 静态，300源帧 | 固定实际源姿态0，独立相机时间轴环绕 |
| fcs005 | 单人动作，300帧/30fps | 原动作与相机同时推进 |
| fcs006 | 单人-头盔，300帧/30fps | 已有人体、刚体联合渲染 |
| fcs008 | 双人，300帧/30fps | 两个独立身份与同步动作 |
| mjj000 | 静态，99源帧 | 从静态原片重建身份，固定源姿态环绕 |
| mjj001 | 单人动作，99帧/9fps | 复用mjj000身份，独立拟合动态 |
| mjj003 | 双人，150帧/9fps | MJJ与灰衣partner分别观测、共用世界坐标 |

相机环绕由`rendering.orbit`配置生成；`rendering.pose_frame`只固定真实源姿态，输出相机帧与源帧含义分别记录，不复制motion。新受试者也走同一套观测、身份、初始化、拟合和渲染实现。

`prepare`、`identity`、`initialize`、`fit`、`render` 可分阶段调用；`run` 串联这些阶段。它会复用配置中已提供的观测、身份和初始化，再执行拟合及渲染。视频编码成功后默认清理本次PNG中间帧，需要保留时加 `--keep-frames`。

## 目录和状态

- `runs/delivery/`：所有结果视频的统一位置，`README.md`列出可播放文件，`manifest.json`记录来源与SHA256。`selected/`放选定成片，`views/`放补充视角，`review/`放尚未采用的必要对照，`exploration/`放衣物等探索结果。
- `src/hoi/`：共用实现；`configs/datasets/`：场景差异；`configs/reconstruction.json`：拟合迭代设置。
- `assets/`：已有观测和选定人体/刚体资产；`data/`：原始数据；`lhmpp/`、`runtime/`：外置第三方依赖与权重。
- `runs/<scene>/`：阶段产物、日志、保存后的配置、关键图和渲染报告；视频直接写入统一结果目录。
- `reports/deck/final/`：保留已交付 PPT 和 GIF，不在源码整理中改名或覆盖。

渲染输出先进入`review/`或`exploration/`，不能凭运行结束自动成为选定结果。实际审看后，通过同一入口移动归档，例如：

```powershell
python pipeline.py export --video runs/delivery/review/fcs005_camera_orbit.mp4 --name fcs_dynamic_orbit --collection selected --replace
```

此命令移动视频并更新清单，不另存一份媒体副本。只有明确要替换同名成片时使用`--replace`；代码历史由Git管理，视频按用途原位更新，不叠加版本号。

双人场景已完成八机位、300个源时刻的完整原片观测准备、NLF和图像手部观测。初始化用有效身体关节共同估计世界位置，不读取旧动作；手部以身份模型的姿态初始化，随后根据图像观测拟合。本轮复用了已选身份；此前也已通过统一入口，从七个真实可见视角重新生成partner身份。

两人全部300帧已完成默认身体、脸和MAP拟合，并实际审看八机位首/中/尾及正面300帧30fps视频，未见明显大幅抽动回归，握手接触和脸手细节仍有不足。单人含头盔和固定新视角也已实际渲染。动作片可通过每人的 `proportions` 输入复用已有体型。

实现与缓存行为见 [PIPELINE](docs/PIPELINE.md)，资产与验证边界见 [BASELINE](docs/BASELINE.md)。文件按用途命名，版本由 Git 提交、分支和标签表达，不另叠加版本后缀。
