# 第三方组件

主链使用 LHM++ 的身份编码与高斯解码、SAM2 的实例分割、RTMW 的全身关键点、NLF 的动作初始化、HaMeR 的手部观测，以及 SMPL-X、PyTorch3D、gsplat 和 nvdiffrast 的几何与渲染实现。自有代码负责统一数据约定、多视角观测、身份关联、动作拟合和联合渲染。

第三方源码及其原始许可证随本地依赖目录保留；本仓库不重新授予这些组件或权重的许可。LHM 相关已留存许可证位于 `third_party_licenses/`，源码与权重许可分别适用。来源及已知版本见 `third_party/sources.json`，环境要求见 `DEPENDENCIES.md`。

旧实验中使用过的 CARI4D、UniDepth、MobileSAM 等不属于当前运行链。旧实验代码与结果可从 Git 历史查询，不应作为当前依赖重新安装。
