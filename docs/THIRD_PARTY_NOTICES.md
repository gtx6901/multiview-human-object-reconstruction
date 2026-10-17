# 来源与许可说明

本归档没有重新分发上游权重、SMPL/SMPL-H/SMPL-X模型或vendor源码树。实验快照通过名称链接并调用实际上游算子；少量运行时兼容入口会在本地读取和执行原始上游方法，不能脱离上游文件使用。相应来源版本见`environment/upstream_sources.json`，权重来源与校验边界见`environment/weight_provenance.json`。

| 组件 | 来源 | 本实验用途 |
| --- | --- | --- |
| LHM++ | https://github.com/aigc3d/LHM-plusplus | 冻结八视角神经高斯外观、蒙皮动画合同、神经解码器 |
| CARI4D / CoCoNet | https://github.com/NVlabs/CARI4D | SMPL-H接口、辅助资产、真实模型前向及输入诊断 |
| NLF | https://github.com/isarandi/nlf | 每图人体姿态/形状预测 |
| UniDepth | https://github.com/lpiccinelli-eth/UniDepth | 已知相机条件的单目深度 |
| MobileSAM | https://github.com/ChaoningZhang/MobileSAM | 真实图像中受提示的可见轮廓 |
| Hunyuan3D-2 | https://github.com/Tencent/Hunyuan3D-2 | 早期失败形状候选；当前最终形状是轮廓几何对照 |
| nvdiffrast | https://github.com/NVlabs/nvdiffrast | 实际网格光栅化 |
| gsplat | https://github.com/nerfstudio-project/gsplat | 实际高斯特征/深度光栅化 |
| SMPL family | https://smpl.is.tue.mpg.de/ | 许可人体几何先验，文件不随仓库分发 |

`third_party_licenses/`保存本地已核对的CARI4D、LHM源码/权重/部分NVIDIA组件、UniDepth和MobileSAM许可文本，作为来源说明；它们不代表替其他上游组件统一授权。CARI4D、LHM权重及UniDepth含非商业限制，适用范围需按各自原文处理。未授予所有文件一个统一MIT/Apache许可，也没有声称这些输出可不受条件商用。

实验贡献是多机位观测整理与共同坐标处理、实际运行兼容、观测拒绝与诊断、共享人体/独立物体轨迹、冻结动画接口、混合渲染、失败证据及归档；不是重新发明或替代上游模型。私有数据和受许可资产应由使用者自行合法取得。
