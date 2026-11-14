# 依赖与本机环境

自有代码位于 `src/hoi`，第三方源码、模型与 CUDA 扩展不计入自有源码，也不随 Git 上传。来源清单见 `third_party/sources.json`。

当前实际运行环境是 Python 3.11、PyTorch 2.5.1+cu124、torchvision 0.20.1+cu124、NumPy 1.26.4、SciPy 1.12.0、OpenCV 4.10.0.84。先安装相匹配的 CUDA 版 PyTorch，再执行 `python -m pip install -e .` 安装 Python 依赖。`pyproject.toml` 不负责安装 CUDA 驱动或从源码编译本地扩展。

本地另需以下资源，路径可在 `configs/runtime.local.json` 中配置：

| 用途 | 本地资源 |
| --- | --- |
| 身份重建 | `lhmpp/vendor`、`checkpoint_config.json`、LHMPP-700M、ArcFace、FLAME 和 SMPL-X 预训练资源 |
| 分割 | SAM2 源码与 `runtime/sam2/checkpoints/sam2.1_hiera_tiny.pt` |
| 身体关键点 | `runtime/wholebody/rtmw.onnx`，rtmlib 0.0.13，ONNX Runtime 1.20.1 |
| 动作初始化 | `runtime/weights/nlf.torchscript` |
| 图像手部观测 | `runtime/hamer` 的源码、配置、HaMeR 与 MANO 权重 |
| 几何和渲染 | PyTorch3D 0.7.8、SMPL-X 0.1.28、gsplat 1.5.3 及匹配 Python/PyTorch/CUDA 的编译扩展 |
| 刚体网格合成 | nvdiffrast 源码及其本地编译扩展 |
| 视频编码 | FFmpeg 可执行文件 |

现有 `lhmpp/deps`、`runtime/deps` 保留了这台 Windows 机器上可用的依赖。共享加载器固定导入优先级，并为所选 LHM 架构延迟导入未使用的上游 CUDA 模块；不伪造计算结果。神经网络权重严格加载，渲染扩展应与安装环境一致。

SAM2 当前未编译可选的连通区域后处理扩展；分割传播可运行，上游会警告跳过这一后处理。HaMeR源码与官方权重提交记录在 `runtime/hamer/source.json` 和第三方来源清单中。身份加载使用safetensors官方pread读取后端，避开本机Windows mmap存储访问异常；仍严格加载同一份权重。当前环境尚未经全新机器安装验证。

项目已有数据集配置中的原片、标定、身份种子及体型输入仍需本地提供。它们不是安装 Python 包时自动下载的资源。SMPL-X、MANO 和模型权重应分别遵守其使用许可。
