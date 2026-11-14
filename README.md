# 多视角人—物动态重建

单人和多人共用 `pipeline.py` / `hoi` 入口，由场景配置指定相机、时间轴、身份与缓存资产。人体链为观测准备 → 身份与体型 → NLF 初始化和图像手部观测 → 四元数身体拟合 → 脸部拟合 → 四元数 MAP → 联合渲染。已有刚体网格和位姿可通过同一渲染器合成。

人体主链与衣物探索分开管理；场景差异放在配置中，版本由Git提交和分支表达。

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

`prepare`、`identity`、`initialize`、`fit`、`render` 可分阶段调用；`run` 串联这些阶段。它会复用配置中已提供的观测、身份和初始化，再执行拟合及渲染。视频编码成功后默认清理本次PNG中间帧，需要保留时加 `--keep-frames`。

## 目录和状态

- `src/hoi/`：共用实现；`configs/datasets/`：场景差异；`configs/reconstruction.json`：拟合迭代设置。
- `assets/`：已有观测和选定人体/刚体资产；`data/`：原始数据；`lhmpp/`、`runtime/`：外置第三方依赖与权重。
- `runs/<scene>/`：阶段产物、日志、保存后的配置和渲染。
- `reports/deck/final/`：保留已交付 PPT 和 GIF，不在源码整理中改名或覆盖。

已实际核对原片三帧观测准备、七个可见视角的新身份编码/融合/解码/导出，以及该新身份的三帧NLF和图像手部初始化。三个身份各自身体、脸、MAP阶段完成300帧一迭代连接检查；既有资产的单人含头盔、双人和固定新视角已实际渲染，双人完整300帧视频已生成。完整序列从新身份重新拟合至最终画质仍未整体验收。动作片可通过每人的 `proportions` 输入复用已有体型，避免从遮挡动作帧强行重估。

实现与缓存行为见 [PIPELINE](docs/PIPELINE.md)，资产与验证边界见 [BASELINE](docs/BASELINE.md)。文件按用途命名，版本由 Git 提交、分支和标签表达，不另叠加版本后缀。
