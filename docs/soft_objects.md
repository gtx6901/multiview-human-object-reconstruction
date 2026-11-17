# 软性物体动态重建探索

更新：2026-11-17。任务是MJJ002的提包、背包、脱包过程，实际8机位×99帧、9fps；不能把包固定到人体关节来替代其自身形变。正面原片已审看0/14/28/42/56/70/84/98，快速甩包存在明显运动模糊，黑包与黑衣局部遮挡。首先完成七项人体汇报视频，再使用空闲GPU验证软物体方案。

## 近期方法与本机筛选

筛选窗口为2024-11至2026-11。本机RTX4060 Laptop 8GB、16GB RAM，目标1–2天；其他GPU的耗时只作参考，不等于本机承诺。表中的体积是模型文件大小，未证明训练显存可容纳。不同数据集的论文排名不能直接当作本项目排名。

| 方法 | 与当前输入的关系 | 作者资源及实现证据 | 本轮选择 |
| --- | --- | --- | --- |
| Multi4D，ECCV2026 | 原生多视角动态场景；静态、持久动态、瞬态三个子集，允许非刚性运动 | 论文20k迭代、RTX4090约1.2小时；214.7MB是模型存储。公开完整源码，但未报告8GB训练峰值，需编译混合CUDA光栅器并核对实际代码日程。 | 已完成统一输入、固定标定点云与原生训练适配；coarse2000已保存，fine20000待续训；尚不能证明后期致密化资源或最终画质。 |
| FreeTimeGS，CVPR2025 | 原生多视角；带时间存在区间和速度的高斯，适合复杂运动 | 30k迭代/300帧，RTX4090约1小时。项目页链接Renderer与EasyVolcap Framework；已查公开main树与两个分支，未定位完整方法配置/训练实现。 | 保留候选；不把非官方最简复现当作者实现。 |
| SplineGS，CVPR2025 | 单目运动自适应样条；有形变表示，但不是当前八机直接接口 | 作者表中有标定设置约0.5小时、免SfM设置约1.5小时，RTX3090Ti；公开代码额外依赖深度、跟踪，custom data入口仍TBD。训练显存未核实。 | 暂不优先，避免丢弃现有多视角几何并增加前处理。 |
| USplat4D，ICLR2026 | 关注遮挡下运动传播，含背包案例；是已有4D模型上的不确定性图优化 | 补充材料H100上200图约10分钟建图、13分钟追加训练，未含MoSca初始化。官方代码要求Linux并验证H100，含专用gsplat分支及多个CUDA扩展。 | 方法相关；完整资源/Windows接入成本尚无可行证据，不按13分钟承诺总训练。 |
| AT-GS，WACV2025 | 原生多视角增量表面重建，输出动态mesh，使用前后向光流保持时序 | RTX6000 Ada；首帧独立拟合，后续每帧coarse200+fine800，约32–38秒/帧。官方Ubuntu，含RAFT/NTC，8GB峰值未核实。 | 作为明确表面输出的备选，未下载或训练。 |
| HOIGS，2026 | 人-物分离及交互条件看似贴近任务 | 论文限制明确说明固定object scale以约束刚性，大形变/压缩受限，依赖生成式初始物体。 | 排除本轮软包首选，不能因题名含HOI就认为解决软物体。 |

## 当前实现核对

官方源码已固定到上述提交，第三方目录受Git忽略，不把原仓库媒体复制进主项目；三通道与混合光栅器必须独立编译，不能复用AG/Wardrobe旧扩展。实际三通道返回五项、混合返回十四项，语义不能按旧接口猜。三通道光栅器、kNN与混合光栅器均已构建并从独立目录加载；实际日志在runtime/soft_objects/logs/{rasterizer,knn,hybrid}.log。首轮定位到系统SDK路径缺失，现使用本机已有D盘SDK头文件、库及rc工具，不安装或改动全局环境；已执行32×32、两组高斯的真实CUDA前反向，分别验证3D五输出和混合十四输出以及网络输入梯度有限；此检查只覆盖算子接口，不能证明完整模型8GB可训。实际完整模型已运行至coarse2000；初始阶段峰值约3.10GB CUDA分配，后期致密化仍需实测；没有用早期画面评判充分训练后的效果。编译以单CPU任务执行，不并发第二个GPU任务。

Neu3D reader硬编码300帧及公共焦距，不能直接拿99帧/不同内参的本地序列套进去；Technicolor支持偏心主点，但默认50帧并自动留出相机，也需要明确适配。所需适配应读取统一场景metadata，保持8机真实K/W与时间轴，不按MJJ名称选择算法。

独立Python依赖已安装到runtime/soft_objects/python：mmcv1.6.0、natsort8.4.0、lpips0.1.4、yapf0.40.1、tomli2.2.1、platformdirs4.3.6；未替换主运行环境。mmcv正确读取官方dynerf配置，官方train模块已实际导入。唯一源码补丁让混合光栅器加载已编译模块，运行时不再编译；补丁在third_party/multi4d.patch；固定提交上用git apply --unidiff-zero应用（零行上下文，避免补丁文档本身的行尾空格）。已加入统一objects入口和通用metadata适配，完整实现与验证如下；最终显存及画质仍待完整训练。

可复现构建（本机已有工具链，日志均留D盘）：

```powershell
./scripts/build_soft_objects.ps1 -Python D:/Document/3dgs/.tmp/rebuild_20261014_runtime/venv/Scripts/python.exe -CudaHome D:/Document/3dgs/.conda/3dgs -WindowsSdk D:/Document/3dgs/.tmp/rebuild_20261014_runtime/sdk
```

## Multi4D的验证与边界

优先保留作者算法和实际训练日程，先确认相机轴、单位、时间戳、投影尺寸及首帧点云。已定位的公开代码与论文存在日程差异：train.py独立执行coarse 2000次，再重置计数执行fine 20000次；dynerf默认共22000次，densify_until为16000。不能把论文20k直接当源码总更新数，也不擅自缩成20k。官方读取Neu3D或Technicolor格式，需要首帧fused.ply；现有相机外参应直接导出，不能重新让COLMAP覆盖已选统一标定。初始点云仍需真实多视角几何，不用单图生成包几何冒充实测。

先检查编译与真实前反向、有限值和首次优化器更新，再测吞吐、GPU峰值、主机内存、加载方式和高斯增长；必须覆盖初始化、阶段切换及致密化后的资源变化，不能只用首100步推算全程。若资源不足，先定位具体分配，配置中明确记录分辨率/批量等调整。早期画面只用于发现坐标或实现错误，不用于判定充分拟合后的效果。完成作者日程后再审看8机位与新视角连续结果。

首先验证完整人-包动态场景的新视角重建，再决定是否能从持久3D表示分离背包并与现有人体合成。整场图像拟合不等于可独立编辑、可仿真或具物理参数的软包资产。当前仅完成coarse阶段，没有最终软包动态结果；系统图尚未包含这一新探索支路。

## 原始来源

- [Multi4D论文](https://arxiv.org/html/2606.22197v1)，[官方代码](https://github.com/BatFaceWayne/Multi4D)，本次审查代码提交415793949758807902dbe97cea3df21da476126f。
- [FreeTimeGS论文](https://arxiv.org/html/2506.05348v1)，[作者项目与实现链接](https://zju3dv.github.io/freetimegs/)。EasyVolcap公开main审查提交c8410c7d41b813a4fdc08fd38e47a4e5f4046656；没有定位到不代表作者永远未发布。
- [SplineGS论文](https://openreview.net/pdf/2a011ba46a471a5443e3e60d920de8606401412b.pdf)，[官方代码与custom data状态](https://github.com/KAIST-VICLab/SplineGS)。
- [USplat4D论文及运行分析](https://arxiv.org/html/2510.12768v1)，[官方安装与初始化依赖](https://github.com/TAMU-Visual-AI/usplat4d)。
- [AT-GS官方代码与协议](https://github.com/fraunhoferhhi/AT-GS)，[资源与每帧预算](https://openaccess.thecvf.com/content/WACV2025/supplemental/Chen_Adaptive_and_Temporally_WACV_2025_supplemental.pdf)。
- [HOIGS方法与限制](https://arxiv.org/html/2604.04016v1#S5)。

## 已运行的完整模型（11月17日）

统一入口为pipeline.py objects --config configs/objects/mjj002.json --stage prepare|geometry|train。8机99时刻共792张图，使用实际PTS、各自去畸变内参、统一外参和同一相似变换；原片与标定未修改。图像按需加载，训练尺寸最大边1024，不复制缩小图像。

固定相机COLMAP稀疏重建保留248点；环绕机位主要形成两视图轨迹，原默认忽略两视图轨迹才导致初始仅2点。开启官方两视图轨迹选项，其他残差筛选不变。相机内外参保持原值，初始同名点重投影中位0.803px。官方COLMAP3.13 CUDA构建在D盘独立运行；MVS三视图融合得到965个真实点，未进一步降低深度/法线/重投影条件。

完整原生三分支前反向与续训核验发现瞬态分支只有一个种子点，原三近邻初始化产生Inf尺度并污染Adam。现从完整前景点集继承该种子真实邻域尺度，不增加伪点、不截断NaN。完整三分支参数、Adam、采样栈及Python/NumPy/Torch/CUDA随机状态已恢复核对；coarse→fine切换与fine续训不重复执行已检查。唯一原子检查点runs/mjj002/objects/model/checkpoint.pt。

coarse2000检查点已完整写入，约73.6MB；随后为七项视频渲染安全暂停，fine尚未完成。作者预算仍为coarse2000+fine20000，最后一次fine循环按原代码不执行optimizer.step。初始约1.3–1.5次/秒，不能据此承诺致密化后的总耗时。最终必须审看8机位及新视角连续结果；整场重建不等于已分离的背包资产。

一次临时SQLite导出核对目录runtime/cache/tmpluo4nthe删除被执行审查以blocked by policy拒绝，仍保留，未换方式清理。
