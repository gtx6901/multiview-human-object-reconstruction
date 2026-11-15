# 衣物探索

此分支在共享人体主链上接入GaussianWardrobe，保持独立的身体与衣物层。使用同一入口和一份场景JSON，YAML保留作者网络、损失及优化参数。

```powershell
python pipeline.py clothing --config configs/clothing/mjj001.json --stage render --frames 0 49 98 --video
python pipeline.py clothing --config configs/clothing/mjj001.json --stage train --updates 2000
```

`--updates` 是累计目标更新数，不是本次追加量。已有1800次更新的检查点传1000不会再训练；传2000会继续至2000。每个输出目录仅保留一个 `checkpoint.pt`，包含模型、优化器、游标和随机状态。

| 阶段 | 用途 |
| --- | --- |
| bind | 从配置的binding_source绑定已有同步图像、掩码、相机、SMPL参数；新数据需提供这些真实输入 |
| parse | 对实例图像做SCHP语义观测 |
| layers | 汇聚多视角语义，形成身体与衣物模板；需要原模板、权重volume和原canonical定义 |
| canonicalize | 将已有分层模板转换至作者使用的canonical |
| position_maps | 生成身体、衣物位置图 |
| training_masks | 生成canonical手部/身体/衣物监督及邻接数据 |
| pretrain | 作者模板预训练 |
| train | 从现有检查点恢复图像监督训练 |
| render | 从实际图像训练检查点输出相机帧和视频 |

当前MJJ输入已绑定，不应重新bind或覆盖语义/模板来恢复训练；这些操作会改变输入签名。JSON中的source_template、source_weight_volume、source_canonical_config用于重新准备分层输入。该配置没有binding_source，已有绑定数据无需重复绑定。

第三方源码分别位于 `third_party/gaussian_wardrobe` 和 `third_party/human_parsing`，保持其原始许可证。Wardrobe固定上游提交为 `3e62461b724d546f368f72a6d5ef57a21c6e3635`；本地五处修改保存为 `third_party/gaussian_wardrobe.patch`，在对应提交上应用。Human Parsing提交为 `eb84c432cc697f494d99662a05f2335eb2f26095`。

本地编译依赖位于 `runtime/clothing`；唯一构建脚本为 `scripts/build_clothing.ps1`，需要匹配的PyTorch、CUDA、MSVC。Wardrobe的label+RGB必须一次联合六通道渲染，不能复用旧AG三通道扩展。脚本已做语法核对，重编译仍需在目标环境实际运行。

已实际输出1800更新的八机位0/49/98和cam04完整99帧/9fps视频；画面有明显彩色网格、重影和脸手模糊，尚未采用。相同输入的身体与衣物网络颜色图在光栅化前已有网格，需继续定位网络输出及其上游。六通道光栅器、StyleUNet上采样和激活的实际前反向检查通过；代码审计未找到当前训练/推理输入数值不一致的确证，也不能据此认定训练不足就是全部原因。

同一输入的初始化网络颜色图也有网格；1800检查点已经学到部分人体/衣物颜色，规则网格仍在。两层全部颜色网络的Adam step均为1800，排除了“颜色网络完全没有更新”的解释。该控制只定位现象来源，不保证继续训练即可消除它。

Windows恢复使用只读检查点映射，Adam状态使用项目D盘临时文件映射，最后Tensor引用释放后自动回收；不依赖C盘用户临时目录。保存时每个原生状态Tensor各自输出紧凑storage，避免PyTorch Windows单次大于INT_MAX写入截断；网络、FP32、损失与原生Adam公式不改。第一次合并存储块的实现曾损坏1802保存：全部模型和辅助记录经CRC恢复，受损Adam历史明确重置，事件随checkpoint的recovery字段保留，不能称无损续训。修正后1803真实更新及完整5.386GB检查点重新加载通过，全部447297872参数的状态齐全且有限。后续训练与视觉结果以HANDOFF及实际日志为准。

随后完成1803→2000更新和完整检查点重新加载；15分钟训练中C盘最小余量约2GB，未再耗尽。2000的八机位关键帧和cam04完整99帧视频已实际审看，网格、重影和脸手模糊仍明显，未采用。本轮解决了状态IO和资源占用，尚未解决衣物画质；不能据此宣称动态重建成功。

原始YAML中的历史数据类名仅作为兼容别名保留，同一个通用reader按metadata读取帧数、相机数、尺寸与FPS，不为MJJ另写算法。源模板canonical与Wardrobe canonical不同，不能直接换用后者解释旧模板。

分层模板生成与图像训练均属于探索，尚未证明能从任意新视频直接得到良好衣物动态。检查实际多机位和连续视频后才决定是否采用结果；原始数据和保留人体资产不覆盖。
