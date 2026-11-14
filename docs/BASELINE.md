# 资产与验证范围

更新：2026-11-15。此处列出保留资产和实际验证范围；衣物探索独立于人体主链。

## 选定资产

| 用途 | 本地路径 |
| --- | --- |
| FCS006 人体 | `assets/human/avatar/avatar.pt`、`assets/human/motion/motion.pt` |
| FCS006 头盔 | `assets/helmet/texture/mesh.pt`、`assets/helmet/pose/sequence.pt` |
| FCS008 两人 | `assets/scenes/fcs008/fcs/`、`assets/scenes/fcs008/partner/` |
| 相机与共享坐标 | `assets/capture/manifest.json`、`assets/scenes/fcs008/cameras.json` |
| 既有演示成片 | `runs/same_camera/`、`runs/novel_fixed/`、`runs/novel_orbit/`、`runs/four_views_30fps.mp4` |
| 已交付材料 | `reports/deck/final/八视角人—物动态重建_正文8页.pptx`、`reports/deck/final/双人交互.gif` |

原始视频、输入图像/掩码/标定、选定张量资产、外置模型权重和已交付 PPT/GIF 保留。已有成片来自先前资产与运行，不能用它证明新入口完整重建已经验证。

## 本轮实际核对

- CPU 导入和命令帮助、配置解析、1/N 身份结构、源帧交集/轨迹索引等检查。
- 对迁移的关键点定义、裁剪、预测预处理、几何、损失和最小二乘等数学块做 AST 对照；相关合成默认数据项、mask、权重和 loss 做过 CPU 一致性核对。
- 单人加双人的三个身份，各自身体、脸和 MAP 阶段以每阶段一迭代连接了 300 帧，共九个短阶段通过。这是链路和形状检查，不能证明拟合收敛或最终画质。
- 用既有资产运行了 FCS006 单人含头盔首帧、FCS008 双人首帧。输出分别在 `runs/fcs006/render/`、`runs/fcs008/render/`，相机为 21305717、帧为 0；单人为 1134×1512、双人为 1008×756，运行退出 0，已检查单/双人首帧。头盔使用 expected-depth 合成。
- 固定新视角 orbit008 首帧也已实际运行并退出 0；合成相机轨迹做过 CPU 解析/索引检查。这些检查不等于连续新视角视频通过。
- 三帧原片准备已通过。动态握手帧的体型重估因缺少全部测量关节停止，仍执行原筛选要求；当前接口允许每人显式提供已有 `proportions`。

新增实际验证：新包使用partner七个可见视角完成编码、融合、解码和avatar导出，并用该新avatar完成3帧NLF及8机位图像手部推理，均退出0。另用已选身份和完整观测，对FCS008两人全部300帧执行默认身体400/200/200、脸200、MAP600次拟合，退出0；实际审看八机位0/149/299及21305717完整300帧/30fps视频，未发现明显大幅抽动回归，接触和脸手细节仍不足。新动作在 `runs/fcs008/{fcs,partner}/motion/`，更新配置在 `runs/fcs008/config.json`；保留资产未覆盖。新身份完整序列重新收敛拟合至最终视频仍未整体验收。

## 复现与比较口径

统一入口与命令见 [PIPELINE.md](PIPELINE.md)。阶段产物写入 `runs/<scene>/`，更新后的路径保存在该目录的 `config.json`；复用缓存前核对输入、配置和实际帧范围。保留原始标定和资产来源，不覆盖它们来修正新运行。

训练损失、输入视角拟合和重投影只作诊断，不等于独立新视角质量。新视角没有逐像素真值时只做定性判断；同一组对照应使用相同帧、相机、分辨率与掩码。头盔合成沿用 expected-depth 方法，其遮挡结果仍需目视核对。

文件按用途命名，不再叠加版本或“最终”后缀；版本由 Git 管理。既有交付物名称在本轮保持。
