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

Windows safetensors默认mmap曾在 `get_tensor → torch.storage.__getitem__` 原生访问异常。使用官方pread读取同一checkpoint后，相同上下文严格加载和完整身份前向已通过；不改网络或权重。此前一次exit1原因未独立确认，不推断成模型质量问题。16GB RAM下注意系统分页，所有显式临时、缓存和输出用D盘，不擅改系统分页设置或删除C盘个人/系统文件。

## 衣物状态

绑定输入 `assets/scenes/mjj001/clothing`，原字节YAML `configs/clothing/mjj001.yaml`，唯一检查点 `runs/mjj001/clothing/checkpoint.pt`。检查点5386350370字节，SHA256 `EF2D3A4B343C174658CCD1E7385685FEAB62422E7A3F4E992813BEA923285FB9`，迁移前后一致；meta加载确认 author_train、updates1800，common和train内容签名均一致。未改写大检查点或继续训练。

输入manifest与YAML保持原字节。签名比较只忽略限定位置的绝对config路径，保留config SHA、输入hash、counts和训练语义。源模板的AG canonical与Wardrobe姿态不同；必要源配置、模板和volume保留在 `clothing/source`，不可偷换。GaussianWardrobe五处本地修改已保存 `third_party/gaussian_wardrobe.patch`。`--updates` 是累计目标，不是追加次数，1800检查点传1000不会再训练。

迁移后实际渲染八机位0/49/98及cam04完整99帧/9fps，退出0，已审看，彩色规则网格、重影和脸手模糊仍明显，未采用为良好动态结果。进一步抓取相同49帧/cam04的body/cloth颜色图，光栅化前即有同样网格：来源已缩小至网络颜色输出及其上游。独立六通道CUDA前向/颜色梯度、StyleUNet上采样/激活的CUDA与作者CPU参考检查通过，不支持把当前网格归因于这些算子；不代表整个模型训练正确。检查点仍1800，未加滤波或盲目续训。

最后完成同一49帧/cam04的RGB网络初始值与checkpoint控制：初始化输出本身已有规则彩色网格，1800输出已经学到部分人体/衣物颜色但仍残留网格；身体和衣物全部color_net的Adam step均为1800，不能说RGB网络完全没更新。证据为唯一 `runs/mjj001/clothing/texture.log` 和可复用 `runtime/cache/review.png`，没有新增检查脚本。此结果不能保证长训会修好，仍需定位训练动力学及输入对应；不要重复已完成的CUDA/输入静态审计。

DeepSeek首次启动因headless缺少provider失败；按桌面现有provider/model加入仅本次临时overlay后，代码只读审计已完成（exit0+final，session-664dd3ce-67cf-4b51-8806-39db7ab9e3f4）。未找到当前训练/推理输入数值不一致的确证；指出渲染签名覆盖不足，但没有当前输入变化证据，不把这类校验缺口当网格原因或先加护栏。结果在 `runtime/deepseek/observation_audit/answer.md`；代码审计没有读取原图/视频/权重。当前根因仍未闭合。

## 下一步

完成衣物训练/推理输入审计，按证据定位网格，不按loss采用。主链不再依赖旧work代码。旧目录删除被执行策略拦截，缩小到单个 `work/mjj/reconstruction/stability` 仍被拒绝，工具只返回blocked by policy，没有具体原因。已向用户列出明确路径重新确认；未删除任何旧目录，不换API绕过。旧 `work/mjj/two_person/data/quaternion_fit/prior_control` 约128MiB也曾被拒绝删除，继续保留。

保留原始视频、图片、标定、必要权重、已选资产、已交付PPT/GIF、v1.0/full-pipeline。不上传原图视频或模型权重，不用git add .。正常命名，原位迭代，版本进Git；不保留无用检查脚本、门限文件、失败结果和版本目录。

## 续接

用户明确允许必要训练并持续工作至五小时额度接近用尽。唯一自动任务id `04-06` 当前使用简单每小时06分规则；多行DTSTART规则未得到实际触发确认，04:42由用户手动恢复。不要再建重复定时器。用户要求暂停时立即停用该任务并停工。每次先查实际额度、进程、GPU及输出，保留活任务，避免重复启动；不用重置券。

worker为用户指定的GPT-6.1 Sol medium，负责明确可验收的实现；主代理独立审查、执行和看图。先找输入、算法、实现根因，不加像素补丁、关节锁或画质门槛，不按loss认定成功。

本轮结束前未留训练/拟合/DeepSeek活任务，本轮只读视频服务已停止；下一次仍需先核实真实进程。main及衣物已同步，原始数据、PPT/GIF、v1.0/full-pipeline未变。旧目录清理问题等待用户回复，不能借全权限重试绕过此前拒绝。
