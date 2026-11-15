# 工程交接

更新：2026-11-15。当前目标是整理一套单人、多人共用的可用管线到 main，再从该 main 建立 codex/clothing 探索分支。必要训练、重建和审看已获授权；不是恢复此前无边界的实验。

## 当前进度

- `pipeline.py` / `src/hoi` 已接通 prepare、identity、initialize、fit、render、run、plan。主链34个Python模块，约9200行；配置位于 `configs/datasets`。
- FCS006、FCS008 使用同一算法。后者两身份的必要输入和已选资产在 `assets/scenes/fcs008`；FCS身份、动作和头盔基线仍在 `assets/human`、`assets/helmet`。
- 衣物源码在 `codex/clothing` 分支的 `src/hoi/clothing`，第三方在 `third_party/gaussian_wardrobe`、`third_party/human_parsing`，本地依赖在 `runtime/clothing`。唯一构建入口 `scripts/build_clothing.ps1` 已通过语法解析，尚未实际重编译。
- main已沿远端历史完成共享主链发布，生产树54文件、自有主链34模块约9200行。当前 `codex/clothing` 从整理后main创建并已合入共享更新；最新提交以实际Git refs为准。顶层 `pipeline.py clothing` 使用 `configs/clothing/mjj001.json` 派发各衣物阶段。旧历史、v1.0/full-pipeline保留，没有force push。

## 已实际验证

1. 原片前三帧、8相机、两身份的 prepare 已运行通过。
2. 新包对 partner 的7个可见视角完成编码、融合、高斯解码、avatar导出，退出0；再用该新avatar完成3帧NLF初始化和8机位图像手部观测，两阶段退出0。核对产物暂在 `runs/integration`，唯一滚动日志 `runs/integration.log`。
3. 完成早期9个短阶段连接后，FCS008两身份全部300帧已执行默认身体400/200/200、脸200、MAP600次完整拟合，退出0。新动作在 `runs/fcs008/{fcs,partner}/motion/`，配置在 `runs/fcs008/config.json`；未覆盖已选资产。
4. 单人含头盔和固定新视角首帧已渲染并查看。双人新拟合的八机位0/149/299与21305717完整300帧/30fps视频已实际审看，未发现明显大幅抽动回归；握手接触和脸手外观仍不足。临时保留 `runs/fcs008/render/reference.mp4` 为整理前选定动作的同相机对照，选择完成后清理。21305711靠右与原JPEG一致，属于采集构图。
5. 视频编码成功后默认删除这次编码使用的PNG；显式 `--keep-frames` 才保留，不积累轮次副本。
6. 新生成partner身份进一步完成全部300帧默认身体、脸、MAP拟合，再与另一身份合成八机位0/149/299及完整300帧/30fps视频，均退出0并已审看。新旧身份的rig相同；以相同完整观测拟合，动作Tensor逐项相同，外观Tensor有变化，不能称文件重复。未见明显大幅抽动回归，抓握接触与脸手外观仍不足。

Windows safetensors默认mmap曾在 `get_tensor → torch.storage.__getitem__` 原生访问异常。使用官方pread读取同一checkpoint后，相同上下文严格加载和完整身份前向已通过；不改网络或权重。此前一次exit1原因未独立确认，不推断成模型质量问题。16GB RAM下注意系统分页，所有显式临时、缓存和输出用D盘，不擅改系统分页设置或删除C盘个人/系统文件。

## 衣物状态

绑定输入 `assets/scenes/mjj001/clothing`，原字节YAML `configs/clothing/mjj001.yaml`，唯一检查点 `runs/mjj001/clothing/checkpoint.pt`。迁移后1800→1801曾完整更新保存；新IO实现已完成1801→1802真实更新，C盘最小余量约3.1GB，但第一版合并保存Adam存储块触发Windows PyTorch单次写入INT_MAX截断，生成ZIP无效。该错误由本次实现引入，不能称训练恢复成功。

1801迁移后未留副本，缺失的优化器字节不能无损恢复。已根据ZIP记录原偏移、实际截断差及CRC恢复1802全部模型和辅助记录（2188个storage CRC验证），丢弃受损的1304项Adam历史，明确重置优化器。恢复检查点1806852958字节，重新加载和全部模型Tensor比较通过；payload.recovery记录该事件，并随后续保存保留。后续优化轨迹因此改变，不是无损续训。旧1800视频在review/reference.mp4作为唯一必要画面对照，未生成2000画面。

输入manifest与YAML保持原字节。签名比较只忽略限定位置的绝对config路径，保留config SHA、输入hash、counts和训练语义。源模板的AG canonical与Wardrobe姿态不同；必要源配置、模板和volume保留在 `clothing/source`，不可偷换。GaussianWardrobe五处本地修改已保存 `third_party/gaussian_wardrobe.patch`。`--updates` 是累计目标，不是追加次数，1800检查点传1000不会再训练。

迁移后实际渲染八机位0/49/98及cam04完整99帧/9fps，退出0，已审看，彩色规则网格、重影和脸手模糊仍明显，未采用为良好动态结果。进一步抓取相同49帧/cam04的body/cloth颜色图，光栅化前即有同样网格：来源已缩小至网络颜色输出及其上游。独立六通道CUDA前向/颜色梯度、StyleUNet上采样/激活的CUDA与作者CPU参考检查通过，不支持把当前网格归因于这些算子；不代表整个模型训练正确。这组审看画面来自1800，未加滤波。

最后完成同一49帧/cam04的RGB网络初始值与checkpoint控制：初始化输出本身已有规则彩色网格，1800输出已经学到部分人体/衣物颜色但仍残留网格；身体和衣物全部color_net的Adam step均为1800，不能说RGB网络完全没更新。证据为唯一 `runs/mjj001/clothing/texture.log` 和可复用 `runtime/cache/review.png`，没有新增检查脚本。此结果不能保证长训会修好，仍需定位训练动力学及输入对应；不要重复已完成的CUDA/输入静态审计。

DeepSeek首次启动因headless缺少provider失败；按桌面现有provider/model加入仅本次临时overlay后，代码只读审计已完成（exit0+final，session-664dd3ce-67cf-4b51-8806-39db7ab9e3f4）。未找到当前训练/推理输入数值不一致的确证；指出渲染签名覆盖不足，但没有当前输入变化证据，不把这类校验缺口当网格原因或先加护栏。结果在 `runtime/deepseek/observation_audit/answer.md`；代码审计没有读取原图/视频/权重。当前根因仍未闭合。

## 下一步

当前优先修复衣物检查点恢复的主机提交峰值，不按loss采用。Windows torch2.5.1的mmap=True使用整文件私有映射：保留optimizer仍让整个5.38GB映射存活。普通mapcpu加载更糟，CPU分配器保留约7.16GB；只读映射+RAM arena仍增加约3.63GB，真实训练在loss构建阶段耗尽C盘余量，两者均不采用。

现实现改为只读检查点映射，并将Adam状态放在D盘每dtype一块临时可写文件映射；原生Adam数学、FP32、网络和损失均不改。真实1801检查点全部Adam值逐项相同，额外private commit约66MB；小CPU原生Adam对照、延迟梯度、重复保存/恢复、F64/int64 step及最后引用释放均通过。worker发现重复load会累积arena游标，已修为每次恢复新arena，复测通过。DeepSeek实现任务optimizer_storage因shell启动延迟超时，没有final，不能称其自行验收完成；主代理已接管并独立核对。

保存已改为每个原生状态Tensor各自的紧凑storage view（不复制数据、不保存预留尾部），避免单个3.578GB存储块。小CPU保存/加载、原生Adam数值、延迟梯度、重复恢复及临时文件释放通过。真实1802→1803更新退出0，C盘最小余量约1.98GB；完整5.386GB保存重新加载通过，1304项状态覆盖447297872参数，全部有限、Adam step=1，recovery保留。当前检查点SHA256为258e2e1d54f76bc6f0a5cd51214935af97be0fcafd09cd900baf625b08398bf1。

1803→2000统一入口训练已退出0，约920.6秒，C盘最小余量2007314432字节；完整检查点再次加载通过，1304项状态全部有限、Adam step=198，recovery保留。当前唯一检查点5386350562字节，SHA256 b54b59e337fec58dbfb957ba18a115b392fab4b3c5d10a4706e8e0fe6c295311。日志runs/mjj001/clothing/wardrobe_training.log。

2000检查点八机位0/49/98与cam04完整99帧/9fps渲染退出0，已逐机位看图、正面高分辨率图及完整视频播放到结尾，并对照1800 reference。网格、重影、脸手模糊仍明显，没有实质解决，未采用为良好动态结果。2000画面在review/cam00至cam07，唯一新完整视频review/cam04/preview.mp4；1800 reference仅作为尚未闭合颜色网络机制的必要对照，不新增轮次目录。不继续盲目加训练或重复CUDA/输入静态审计。下次先查颜色网络输出/监督与训练动力学，注意本轮Adam已重置这一混杂因素，不能当无损续训对照。

当前无GPU/DeepSeek训练任务，临时本地视频服务已停止；继续前仍核对真实进程。主链工程整理的物理清理仍受下述执行拒绝影响，不能宣称整个整理完成。额度本轮已用92%，保留准确交接，下一次按实际额度判断，不使用重置券。

头盔生成来源另有边界：当前入口只消费选定mesh/轨迹。tag中的clean_helmet_motion是选定pose报告的直接producer，要求清理前轨迹、原人体动作和phases(start/stop_exclusive/joint)；这些旧名字在当前resolver及已检查的lhmpp/work位置未找到，不能断言全盘丢失，也不能仅从最终轨迹反推出清理前自由运动。full-pipeline的baker已有后续响应模型修改，不能称它精确复现选定纹理。后续清理若发现阶段表、锚点、原壳/UV、原分割/深度输入，先保全必要输入，不能凭目录名删掉。

主链不再依赖旧work代码。用户已明确再次授权删除旧work（保留prior_control）、lhmpp/runs、lhmpp/inputs和旧7个基线脚本，不需要再次询问授权。本轮已核对迁出衣物输入/唯一检查点存在、当前生产没有这些旧路径的引用、待删树没有reparse point，并建立排除prior_control及其祖先的绝对路径删除计划。但执行审查仍在进程启动前拒绝整个批次，只返回blocked by policy，没有具体原因，删除没有执行。阻碍是执行拒绝，不是缺少用户授权；不换API、父目录或权限绕过。旧 `work/mjj/two_person/data/quaternion_fit/prior_control` 约128MiB继续保留。

保留原始视频、图片、标定、必要权重、已选资产、已交付PPT/GIF、v1.0/full-pipeline。不上传原图视频或模型权重，不用git add .。正常命名，原位迭代，版本进Git；不保留无用检查脚本、门限文件、失败结果和版本目录。

## 续接

用户明确允许必要训练并持续工作至五小时额度接近用尽。唯一自动任务id `04-06` 当前使用简单每小时06分规则；多行DTSTART规则未得到实际触发确认，04:42由用户手动恢复。不要再建重复定时器。用户要求暂停时立即停用该任务并停工。每次先查实际额度、进程、GPU及输出，保留活任务，避免重复启动；不用重置券。

worker为用户指定的GPT-6.1 Sol medium，负责明确可验收的实现；主代理独立审查、执行和看图。先找输入、算法、实现根因，不加像素补丁、关节锁或画质门槛，不按loss认定成功。

共享验证文档与Black100配置已同步remote main=d0d28ad，已合入clothing；衣物IO实现已发布f97de2e，最新审看记录随本轮提交发布，以实际refs为准。远端只有main与codex/clothing；本地已合并的codex/publish-main引用已删除，未合并历史WIP codex/interaction-cloth保留，不能强删未核对历史。临时历史源码文本、过期委派日志的删除也被执行审查拒绝，尚保留；不换API绕过。原始数据、PPT/GIF、v1.0/full-pipeline未变。旧目录删除授权已收到且持续有效，但本次执行仍被拒绝，不能借全权限重试绕过。
