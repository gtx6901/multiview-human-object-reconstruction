# 衣物探索

此分支在共享人体主链上接入GaussianWardrobe，保持独立的身体与衣物层。使用同一入口和一份场景JSON，YAML保留作者网络、损失及优化参数。

## 当前运行

2026-11-17 03:13为优先完成周四汇报视频，已核对5300检查点的ZIP目录、phase/epoch/offset与稳定文件状态，确认无checkpoint.pt.tmp后暂停自有GPU长训。保存状态author_train、updates5300、epoch6/offset548；不是按早期画质停止。渲染任务完成后从5300恢复，不能从日志更高但未保存的更新恢复。暂停后C盘余量恢复约8.8GB。唯一训练目标仍为固定官方默认400001次更新（学习率日程400000），不再运行短画质A/B。FCS003/005与MJJ000环绕已完成。MJJ001完整统一run已完成99帧环绕，但尚未实际审看。随后恢复衣物时，我将外层输出错误重定向到入口自身管理的train/run.log，造成Windows共享写锁冲突，训练未启动、5300检查点未改变。现已将外层日志改为clothing/pipeline.log重新恢复，训练实际日志仍为train/run.log；先查真实进程，不重复起任务。

最近外观输出是4000更新：strict加载、八机位0/49/98与cam04完整99帧/9fps已完成；24张关键帧逐项与3000比较。用户反馈衣服整体效果还可以，应保留为衣物层阶段成果；皮肤格纹、重影和脸手模糊分别评价。正式review为4000，reference.mp4/png为3000唯一必要对照；降低epsilon更差，该临时改动已撤回。不能用约1%训练表现否定充分拟合的效果。

4000连续播放尚未完成审看：电脑操作工具无法可靠确认浏览器网址后已停止操作，未换控制方式；当时临时视频服务已结束。不能把24张图当作完整视频验收。当前4000→5300只做正式原生采样长训，没有通过额外图像修饰改变输出。

04-06已按用户最新任务恢复每小时续接；primary五小时约88%时保留余量，正常重置后继续，不新增定时器。最新七项视频状态在runs/delivery/manifest.json维护；当前任务进程须现场核实后再恢复训练，避免与渲染并发。

## 本轮交付队列（2026-11-17）

周四11月19日汇报优先：FCS与MJJ各自的单人静态旋转、单人动态旋转、双人交互，共六项；另有FCS单人-物体视频，共七项。七项素材与各场景配置已核对；FCS003/005和MJJ000新环绕完成，FCS008/006有选定成片，MJJ001环绕完成待审看、MJJ003双人待跑。新MJJ000身份独立八视角生成，不把同一输出改名冒充不同任务。已有GPU长训不并发第二重型任务；为交付视频可在核对完整稳定的滚动检查点后暂停自有长训、顺序渲染再恢复。

后续调研最近两年软性物体动态重建，以4060 Laptop 8GB、16GB RAM及1-2天实际训练预算筛选，再尝试MJJ单人-背包交互。先核实输入、方法真实训练设置和本机速度，不预先承诺长训可在截止前完成。

架构图保留一份交互源，位于当前聊天的visualizations目录，含系统、源文件调用、数据流共六个子图和69段源码依据。节点形状和线型含义固定；main的face.py仅用五个候选点中有真实3D支持的锚点拟合颈/头旋转。Wardrobe正常render调用gaussians/gaussian_renderer.py的render3，包含联合六通道及独立body-only监督两个pass。

本轮架构预览使用的runtime/cache/architecture_browser、architecture_preview.html/png、architecture_layout.cjs和architecture_spec.json清理被执行审查在启动前拒绝，未提供具体原因，仍保留，不换方式绕过。独立headless预览浏览器已关闭；当前训练暂停状态见开头，未操作用户浏览器。
## 使用

```powershell
python pipeline.py clothing --config configs/clothing/mjj001.json --stage render --frames 0 49 98 --video
python pipeline.py clothing --config configs/clothing/mjj001.json --stage train --updates 4000
```

`--updates` 是累计目标更新数，不是本次追加量。例如已有3000次更新时，目标2000不会再训练，目标4000会追加1000次。每个输出目录仅保留一个 `checkpoint.pt`，包含模型、优化器、游标和随机状态。

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

当前生产实现的补丁恢复了SSIM与offset损失原先被`.item()`切断的梯度；consistent_label的L1通过semantic_valid排除实例内未知语义，实例外背景仍有效；背面posmap的面索引与属性图同步水平翻转。损失系数仍取固定YAML，但计算图包含这些已声明修正。本次内存A/B保持同一当前生产计算图；这些修正尚未证明解决格纹。

第三方源码分别位于 `third_party/gaussian_wardrobe` 和 `third_party/human_parsing`，保持其原始许可证。Wardrobe固定上游提交为 `3e62461b724d546f368f72a6d5ef57a21c6e3635`；本地五处修改保存为 `third_party/gaussian_wardrobe.patch`，在对应提交上应用。Human Parsing提交为 `eb84c432cc697f494d99662a05f2335eb2f26095`。

本地编译依赖位于 `runtime/clothing`；唯一构建脚本为 `scripts/build_clothing.ps1`，需要匹配的PyTorch、CUDA、MSVC。Wardrobe的label+RGB必须一次联合六通道渲染，不能复用旧AG三通道扩展。脚本已做语法核对，重编译仍需在目标环境实际运行。

Torch模型缓存默认在`runtime/clothing/torch`，也可通过标准`TORCH_HOME`指定；VGG主干权重为其`hub/checkpoints/vgg16-397923af.pth`，LPIPS线性权重为`runtime/clothing/weights/vgg.pth`。训练入口不会自动下载缺失权重。

已实际输出并审看3000更新的八机位0/49/98和cam04完整99帧/9fps视频。与2000相同机位和时刻对照，肤色偏紫有所减轻，但四肢和衣物的规则格纹、局部重影及脸手模糊仍明显，尚未采用。相同输入的身体与衣物网络颜色图在光栅化前已有网格，需继续定位网络输出及其上游。六通道光栅器、StyleUNet上采样和激活的实际前反向检查通过；代码审计未找到当前训练/推理输入数值不一致的确证，也不能据此认定训练不足就是全部原因。

同一输入的初始化网络颜色图也有网格；1800检查点已经学到部分人体/衣物颜色，规则网格仍在。两层全部颜色网络的Adam step均为1800，排除了“颜色网络完全没有更新”的解释。该控制只定位现象来源，不保证继续训练即可消除它。

Windows恢复使用只读检查点映射，Adam状态使用项目D盘临时文件映射，最后Tensor引用释放后自动回收；不依赖C盘用户临时目录。保存时每个原生状态Tensor各自输出紧凑storage，避免PyTorch Windows单次大于INT_MAX写入截断；该IO修正保持当时的网络、FP32、既有损失实现及Adam更新不变。第一次合并存储块的实现曾损坏1802保存：全部模型和辅助记录经CRC恢复，受损Adam历史明确重置，事件随checkpoint的recovery字段保留，不能称无损续训。修正后1803真实更新及完整5.386GB检查点重新加载通过，全部447297872参数的状态齐全且有限。后续训练与视觉结果以本文当前运行段和实际日志为准。

随后完成1803→2000更新和完整检查点重新加载；15分钟训练中C盘最小余量约2GB，未再耗尽。2000的八机位关键帧和cam04完整99帧视频已实际审看，网格、重影和脸手模糊仍明显，未采用。本轮解决了状态IO和资源占用，尚未解决衣物画质；不能据此宣称动态重建成功。

2026-11-16完成2000→3000的1000次原生更新，耗时4539.855秒，保持单个滚动检查点。训练中C盘余量约1.89GB且保存前后稳定，退出后恢复约10.67GB；D盘临时Adam映射自动回收。新进程strict加载3000模型并完成上述渲染，未重跑5001次预训练，也未改变网络、FP32、既有损失实现或裁剪。该轮曾以2000为对照；当前参考文件已滚动替换为3000，见本文开头。

当前模型的颜色输出为多尺度Haar表示。2000检查点的body色网CPU重放中，最后一级Haar往返误差小于8e-7，低频分量也继承格纹；用原生SMPL-X及视角特征重放frame49/cam4后，手部UV颜色仍有格纹。该控制只覆盖一个时刻/机位的body输出，说明不能把问题仅归为最后IWT或无视角条件的诊断图；没有滤除高频或改变最终输出。

3000的Adam状态显示，大部分颜色参数处于epsilon主导区。以同一3000为起点、完整恢复模型/Adam/RNG、保持其他设置，分别用原生`eps=1e-8`与`1e-12`续训100步至3100；输入签名、最终各随机状态和1304项step=1298逐项对齐。两组均完成并审看八机位0/49/98及正面99帧同步视频：较小epsilon加重了红紫色偏，格纹仍在。因此未采用该改动，临时参数入口已撤回，该控制结束时生产保留3000，之后已继续正式训练，见本文开头。临时对照目录的删除被执行审查拒绝，目前仍在`runtime/cache/optimizer_control`，不是选定训练成果；不换方式绕过删除拒绝。

## 论文与发布代码的训练协议

[论文 §6 Training](https://arxiv.org/html/2603.04290v1#S6) 与固定提交 [`3e62461`](https://github.com/eth-ait/GaussianWardrobe/commit/3e62461b724d546f368f72a6d5ef57a21c6e3635) 的 [`main_avatar.py`](https://github.com/eth-ait/GaussianWardrobe/blob/3e62461b724d546f368f72a6d5ef57a21c6e3635/main_avatar.py#L50)、[`configs/4d_dress/avatar.yaml`](https://github.com/eth-ait/GaussianWardrobe/blob/3e62461b724d546f368f72a6d5ef57a21c6e3635/configs/4d_dress/avatar.yaml#L52) 存在协议差异：

| 项目 | 论文 | 固定代码及本项目 |
| --- | --- | --- |
| 图像迭代 | 300k | `iter_num=400000`，含起始0共400001次更新；本项目已从4000恢复正式长训，实际更新数见train/run.log。`eval_interval=1000`只是审看间隔。 |
| clip | `5e-4` | 官方4D-Dress与该项目均为`1e-4`，与作者配置相同；不能因Actor01使用`1e-3`就擅改。 |
| 阶段 | 先将face/smooth/penetration权重归零，再切换全损失 | [`main_avatar` 训练循环](https://github.com/eth-ait/GaussianWardrobe/blob/3e62461b724d546f368f72a6d5ef57a21c6e3635/main_avatar.py#L616-L669)为单阶段全loss，未见切换入口；5001次几何pretrain只训练geometry/opacity/scale/rotation，不训练color/viewdir，不能当作coarse RGB阶段。 |
| virtual bone | — | [官方项目页](https://eth-ait.github.io/GaussianWardrobe/)说明展示结果含投稿后virtual bone扩展；固定源码未见实现入口。这不证明当前网格由缺该扩展导致。 |

不能用前约1%的训练表现判断充分训练后的效果，不再用100/200步旁支代替长训；论文两阶段的切分时刻未公布、固定源码也未实现，暂不能宣称本项目复现完整论文协议。

原生数据工厂中的canonical和fit角色原本分别构造相同参数的SMPL-X模型，现共用一个只读实例。实际原生模型canonical→live→canonical的vertices/joints/A一致，全部模型参数及缓冲区指纹未变；约80MB模型Tensor无需再复制一份。此改动不改变网络/损失/输入约定，完整训练的分页收益仍待测量。

原始YAML中的历史数据类名仅作为兼容别名保留，同一个通用reader按metadata读取帧数、相机数、尺寸与FPS，不为MJJ另写算法。源模板canonical与Wardrobe canonical不同，不能直接换用后者解释旧模板。

绑定与模型构造共用同一函数，从真实`expression`参数矩阵的列数读取表情维度，并在绑定清单中记录。模型支持的维度必须与输入相同，不补零或截断。原有MJJ的100维输入保持原计算，10维输入沿用同一SMPL-X实现。统一bind入口已用真实两帧、一相机的临时输入分别验证10/100维，硬链接和数值保持一致；该输入修正不改变现有衣物画质。

分层模板生成与图像训练均属于探索，尚未证明能从任意新视频直接得到良好衣物动态。检查实际多机位和连续视频后才决定是否采用结果；原始数据和保留人体资产不覆盖。

11月17日另清除了两个已过期且无生产引用的旧preflight JSON报告；所需binding/geometry/segmentation清单保持不变。
