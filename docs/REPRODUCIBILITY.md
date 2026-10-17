# 复现入口与限制

## 公开内容可以立即验证什么

从仓库根目录运行 `python tools/verify_release.py`，只使用标准库，核对公开文件SHA256、大小、文件白名单、源码编译和媒体PNG/JPEG/MP4签名。该命令验证归档完整性，不会下载模型，也不声称重建真人。

仓库提供本地标准库归档校验，没有配置无数据/无GPU却宣称重建成功的持续集成。

## 实验脚本快照

`research/interaction_20261017`保留实验中的明确片段/锚点/目录名称，不能当通用接口。发布副本只将本机运行时路径外置，算法、实际模型调用和算子不替换；原文件与发布副本摘要映射见`evidence/source_snapshot_mapping.json`。

`prepare_helmet.py`提取去畸变数据与K/W；`detect_body.py`、`run_nlf_observations.py`、`fit_nlf_smplh.py`准备真实观测；`fuse_body_shortclip.py`在固定相机下共同拟合人体。头盔采用`track_pickup_helmet.py`的观测、`build_pickup_hull.py`的几何对照、`fit_pickup_object.py`与`refine_pickup_motion.py`的独立刚体轨迹。`render_dynamic_hybrid.py`读取冻结数据并渲染。

`run_pickup_observations.py`和`run_pickup_object_pipeline.py`只是阶段顺序入口，依赖已准备的本地素材、固定目录与经过查看的锚点；后者保留初始对照，并非会自动生成报告中全部最终变体。平滑、共同时间优化及最终展示分别对应相应脚本，实验结果不能通过空目录运行这些入口伪造。

## 本地渲染重放所需内容

需自行具备合法的`avatar.pt`、人体与物体序列、颜色数据、相机文件及原图QA输入；本仓库不分发这些私有/受许可数据。需要从所固定版本的上游安装真正的SMPL-X、PyTorch3D、gsplat、nvdiffrast、accelerate，以及LHM神经解码器依赖。

将上游LHM源码放到`research/lhmpp_20261016/vendor`；CARI4D、MobileSAM、UniDepth源码及依赖放到对应`vendor`、`deps`位置。`research/lhmpp_20261016/checkpoint_config.json`仅包含运行合同所需子集，不是一个可替代完整官方模型配置的文件。

Windows如使用已有编译扩展，可显式设置：

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:HOI_GSPLAT_EXTENSION = 'C:/your-ascii-runtime/gsplat_cuda.pyd'
$env:HOI_WARP_ASCII_ROOT = 'C:/your-ascii-runtime/warp-runtime'
# HOI_SHARED_SITE可选，仅供明确复用现有依赖；推荐在自己的隔离环境安装依赖。
```

这些路径均由使用者提供，不能指向不匹配Torch/CUDA/Python的二进制。Warp目录中需有实际的同版本warp包，编译缓存也在该英文目录下；不能用空算子绕过失败。

已具备本地实验数据时，最终中性材质诊断重放入口为：

```powershell
python research/interaction_20261017/render_dynamic_hybrid.py replay_new NONE helmet_pose_pickup_temporal_v1 helmet_colors_pickup000_135_v1 --avatar avatar_motion_pickup000_135_v1 --data helmet_pickup_inspect000_135_v1 --cameras avatar_motion_pickup000_135_v1/cameras.json --neutral-helmet
```

`replay_new`必须是不存在的新输出目录。用已有输出目录作为第二个参数，可逐PNG比较独立重放；参考相机、数据与输出版本必须对应。修改`cameras.json`的明确K/W可生成其他相机视角，当前没有新相机方向真值可自动验收。

## 完整重建为何不能由此仓库一键复现

未公开原始拍摄、原标定、静态初始化及许可资产，也没有提供一键准备这些条件的替代品。源数据的身份、标定、坐标尺度、PTS对应、可见性锚点和真实依赖均属于必须核验的前提。本仓库不会把官方模型下载、导入成功或固定姿态动画当成该项目重建复现。

本次原始试验环境为Windows、Python3.11、Torch2.5.1+cu124、RTX4060 Laptop 8GiB。完整CARI4D官方流水线尚未在当前项目成功复现；真实CoCoNet前向采用了报告中明确列出的自定义初始化/颜色适配。
