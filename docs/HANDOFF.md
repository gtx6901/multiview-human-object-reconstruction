# 工程交接

更新：2026-11-15。当前目标是整理一套单人、多人共用的可用管线到 main，再从该 main 建立 codex/clothing 探索分支。必要训练、重建和审看已获授权；不是恢复此前无边界的实验。

## 当前进度

- `pipeline.py` / `src/hoi` 已接通 prepare、identity、initialize、fit、render、run、plan。主链34个Python模块，约9200行；配置位于 `configs/datasets`。
- FCS006、FCS008 使用同一算法。后者两身份的必要输入和已选资产在 `assets/scenes/fcs008`；FCS身份、动作和头盔基线仍在 `assets/human`、`assets/helmet`。
- 衣物代码在 `src/hoi/clothing`，第三方在 `third_party/gaussian_wardrobe`、`third_party/human_parsing`，本地依赖在 `runtime/clothing`。唯一构建入口 `scripts/build_clothing.ps1` 已通过语法解析，尚未实际重编译。
- main已沿远端历史快进同步至 `f62ca4a`，生产树54文件、自有主链34模块约9200行。当前 `codex/clothing` 从该最新main创建；顶层 `pipeline.py clothing` 使用 `configs/clothing/mjj001.json` 派发绑定、语义、分层、canonical、maps/masks、预训练、训练和渲染。旧历史、v1.0/full-pipeline保留，没有force push。

## 已实际验证

1. 原片前三帧、8相机、两身份的 prepare 已运行通过。
2. 新包对 partner 的7个可见视角完成编码、融合、高斯解码、avatar导出，退出0；再用该新avatar完成3帧NLF初始化和8机位图像手部观测，两阶段退出0。核对产物暂在 `runs/integration`，唯一滚动日志 `runs/integration.log`。
3. 单人及双人三个身份各自身体、脸、MAP阶段用300帧、每阶段1次迭代连接，共9阶段通过。这证明接口连接，不代表收敛后的新结果画质。
4. 既有资产的单人含头盔、双人、固定新视角首帧已渲染并查看。双人21305711的完整300帧视频已生成且播放到10秒结尾；人物靠右与原JPEG一致，属于采集构图。更适合展示的21305717完整视频正在生成，以真实进程和输出为准。
5. 视频编码成功后默认删除这次编码使用的PNG；显式 `--keep-frames` 才保留，不积累轮次副本。

Windows safetensors默认mmap曾在 `get_tensor → torch.storage.__getitem__` 原生访问异常。使用官方pread读取同一checkpoint后，相同上下文严格加载和完整身份前向已通过；不改网络或权重。此前一次exit1原因未独立确认，不推断成模型质量问题。16GB RAM下注意系统分页，所有显式临时、缓存和输出用D盘，不擅改系统分页设置或删除C盘个人/系统文件。

## 衣物状态

绑定输入 `assets/scenes/mjj001/clothing`，原字节YAML `configs/clothing/mjj001.yaml`，唯一检查点 `runs/mjj001/clothing/checkpoint.pt`。检查点5386350370字节，SHA256 `EF2D3A4B343C174658CCD1E7385685FEAB62422E7A3F4E992813BEA923285FB9`，迁移前后一致；meta加载确认 author_train、updates1800，common和train内容签名均一致。未改写大检查点或继续训练。

输入manifest与YAML保持原字节。签名比较只忽略限定位置的绝对config路径，保留config SHA、输入hash、counts和训练语义。源模板的AG canonical与Wardrobe姿态不同；必要源配置、模板和volume保留在 `clothing/source`，不可偷换。GaussianWardrobe五处本地修改已保存 `third_party/gaussian_wardrobe.patch`。顶层入口已接通，下一步实际恢复/渲染验证；模板和训练次数不能作为动态效果验收。`--updates` 是累计目标，不是追加次数，1800检查点传1000不会再训练。

## 下一步

完成展示机位视频检查和分支同步；按实际画面选择结果。主链不再依赖旧work代码。旧目录删除被执行策略拦截，缩小到单个 `work/mjj/reconstruction/stability` 仍被拒绝，工具只返回blocked by policy，没有具体原因。已向用户列出明确路径重新确认；未删除任何旧目录，不换API绕过。旧 `work/mjj/two_person/data/quaternion_fit/prior_control` 约128MiB也曾被拒绝删除，继续保留。

保留原始视频、图片、标定、必要权重、已选资产、已交付PPT/GIF、v1.0/full-pipeline。不上传原图视频或模型权重，不用git add .。正常命名，原位迭代，版本进Git；不保留无用检查脚本、门限文件、失败结果和版本目录。

## 续接

用户明确允许必要训练并持续工作至五小时额度接近用尽。唯一自动任务id `04-06` 当前使用简单每小时06分规则；多行DTSTART规则未得到实际触发确认，04:42由用户手动恢复。不要再建重复定时器。用户要求暂停时立即停用该任务并停工。每次先查实际额度、进程、GPU及输出，保留活任务，避免重复启动；不用重置券。

worker为用户指定的GPT-6.1 Sol medium，负责明确可验收的实现；主代理独立审查、执行和看图。先找输入、算法、实现根因，不加像素补丁、关节锁或画质门槛，不按loss认定成功。
