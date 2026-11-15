# 工程交接

更新：2026-11-15。先读最新人类消息和真实进程/日志/产物，再继续。本文记录当前状态；旧 work 记录不再作为入口。衣物源码及专属文档仅在 codex/clothing 分支。

## 工程目标与组织

- main 只保留一套可用的单人/多人通用人体管线；codex/clothing 从整理后 main 建立，持续包含 main 共享修正，衣物实现不混 main。
- 统一 pipeline.py / hoi 入口，源码 src/hoi，配置 configs，必要输入 assets，输出 runs/<scene>/<actor或用途>，第三方与权重外置。不同数据集共用算法，仅换配置。
- main共56个跟踪文件，其中src/hoi为34个Python模块，含pipeline.py共9303行（含空行/注释）；规模目标不是删除必要功能的理由。文件/函数 snake_case，类 PascalCase，产物按用途命名；原位迭代，历史进 Git，不叠放版本目录。
- 不加无用检查/门限脚本、像素补丁、关节锁或画质门槛，不按 loss 宣称成功。自有代码/配置/文档定期同步 remote，不 force、不 git add .。
- 保留原始数据、标定、必要依赖/权重、选定资产、已交付 PPT/GIF、v1.0/full-pipeline。原图/视频/权重不上传；assets/runs/lhmpp/runtime/data 不加入 Git 跟踪。

## 人体管线与实际验证

- prepare → identity → initialize → 身体/手拟合 → 脸拟合 → 四元数 MAP → 联合渲染已贯通。单人 FCS006 和双人 FCS008 使用相同实现；旋转内部为归一化四元数，时间约束作用在相对旋转上。
- 原片前三帧、八相机、两身份的 prepare 已通过。没有把它说成全部300帧原片端到端验收。
- partner 七个真实可见视角完成编码/融合/解码/avatar导出；该新身份的三帧 NLF、八机位图像手观测，以及全部300帧默认身体400/200/200、脸200、MAP600拟合均已通过。
- 新身份合成的八机位0/149/299及21305717完整300帧/30fps视频已实际审看，无明显大幅抽动回归；接触和脸手外观仍不足。FCS008既有身份重跑也完成同样拟合与审看。单人含头盔和固定新视角已渲染查看。
- 新旧partner身份 rig 相同；以同一完整观测拟合，motion Tensor逐项相同，外观Tensor有变化，不能称文件重复。验证产物在 runs/integration、runs/fcs008，选定资产未覆盖。必要对照尚在取舍中，不增加轮次副本。
- Windows safetensors默认mmap原生异常，用官方pread读取同一权重后strict加载与完整身份前向已通过，不改模型。
- run已补齐分割输入依赖：重建身份/初始化/已配置但丢失的手观测且index缺失时先prepare；完整缓存不因缺index重跑。NLF与手观测各自复用，未配置手观测且已有初始化时继续保留该可选输入为空。调度/配置传递及部分缓存缺失的内联mock检查通过，不留测试脚本。
- 视频编码成功后默认删除本次PNG中间帧，显式--keep-frames才保留。
- 主链只读调用审查未发现按FCS/MJJ或人物名切换算法，也未发现可移除的整个旧模块；删除了6个无调用/无文档API引用的旧旋转与分割辅助函数及其无用导入。保留代码AST逐项不变，Black、diff检查和两份场景plan通过；未为该清理重复GPU拟合。

## 衣物唯一状态与问题

- 入口 pipeline.py clothing，配置 configs/clothing/mjj001.json；原字节网络/优化YAML configs/clothing/mjj001.yaml，输入 assets/scenes/mjj001/clothing，构建 scripts/build_clothing.ps1。--updates是累计目标。
- 唯一检查点 runs/mjj001/clothing/checkpoint.pt：author_train/update2000，5386350562字节，SHA256 b54b59e337fec58dbfb957ba18a115b392fab4b3c5d10a4706e8e0fe6c295311。1304项Adam状态覆盖447297872参数、step198、全部有限，完整重新加载通过。
- 必须保留恢复说明：第一次合并保存3.578GB单storage触发Windows PyTorch INT_MAX写截断，损坏1802保存，这是本次实现引入的问题。全部模型/辅助记录经2188个storage CRC恢复，但原Adam历史不可无损恢复，已明确重置。checkpoint.recovery随以后保存保留，不能称无损续训。
- Windows检查点现用只读映射，Adam状态为D盘每dtype临时可写文件映射；保存逐个状态Tensor的紧凑storage view，避免整块大写入。原生Adam数学、FP32、网络/损失不改，重复恢复/延迟梯度/保存加载/临时文件回收已验证。
- 1803→2000训练约920.6秒，C盘最低2007314432字节，未再次耗尽；这只是资源验收。2000全八机位0/49/98及cam04完整99帧/9fps视频已审看，网格、重影、脸手模糊仍明显，未采用为良好动态结果。
- 当前画面在 review/cam00至cam07、完整视频review/cam04/preview.mp4；review/reference.mp4为1800必要对照。没有盲目继续增加更新数。

## 已排查与新证据

- 光栅化前颜色图已有网格，随机初始化也有；1800时所有颜色网络Adam step均1800，排除完全没更新。六通道CUDA、StyleUNet上采样/激活对作者CPU参考的实际前反向已通过，不重复这些审计。
- DeepSeek training_contract已exit0+final：原生5001预训练只监督位置/不透明度/尺度/旋转，不训练颜色/viewdir；图像总训练400001步，eval_interval1000只是审看间隔。没有颜色冻结/断梯度的确证。MJJ的clip/loss与作者4D-Dress配置相同，不能按Actor01配置差异擅改。
- 2000的body色网frame49无viewdir CPU前向4.44秒；Haar往返最大误差小于8e-7，低频分量本身也有继承的多尺度格纹。不能仅归因于最后一个IWT算子。
- 已用单个原生SMPL-X forward重放frame49/cam4矩阵，再调用原生get_viewdir_feat及实际色网/视角网权重，完成CPU真实视角条件对照，CUDA未初始化。加入真实视角条件会改变颜色，但手部UV输出仍有格纹。该证据仅覆盖body、一个时刻/机位，未解决画质，不是新最终视频。
- 首次原生Dataset CPU探针因自己先import cv2、后设置OPENCV_TEMP_PATH导致EXR临时路径失败，是诊断脚本导入顺序错误；正常入口原本正确设置相对ASCII临时路径，不误报生产根因。
- 重型CPU Dataset探针随后触发分页峰值，自己的进程已退出，未得到其完整前向，不再重复该方式。改为分阶段单模型/单色网重放才完成上述对照，不改生产算法。
- DeepSeek model_ownership已exit0+final，移除了工厂中参数相同的两套SMPL-X重复构造，canonical/fit角色共用一个只读实例，不共享另有设备约定的全局模型。主代理核对commons/调用方只读；一个真实原生模型canonical→live→canonical的vertices/joints/A逐项相同，全部参数/缓冲区指纹未变，约80157452字节的模型Tensor无需复制第二份。完整训练的分页收益尚未测量。
- DeepSeek training_protocol已exit0+final；主代理直接核对论文§6及固定main_avatar源码：论文写300k、clip5e-4、两阶段先关闭face/smooth/penetration，固定4D-Dress代码为400001更新、clip1e-4、单阶段全损失，没有该coarse RGB切换。论文没有给出阶段分界，不能自行补造。本项目遵循固定代码，不能声称已复现完整论文协议。官网展示另含投稿后virtual-bone扩展，固定代码未见入口，不据此归因当前网格。
- 已把2000渲染与同尺寸、同像素裁剪原图实际比较：frame49全部8机位，以及cam04/cam06的0/49/98。身体主要动作相符，局部肢体/轮廓仍有偏差；快速摆臂和弯腰的输入本身有明显运动模糊，无法从这几张图单独区分曝光、不同步和拟合误差。两机位0/49/98绑定图与assets/mjj/dynamic对应0/49/99的JPEG SHA逐项相同，未发现这一步额外合成模糊。未据此宣称标定正确或已找到网格唯一原因。
- 19:30左右资源恢复后，在2000权重及其下一真实采样项243（pose30/cam3）做一次原生forward/total-loss/backward/clip，只跳过optimizer.step且不保存。两层颜色网络均有梯度；全局裁剪前总norm0.6560，body/cloth颜色0.6046/0.2214，位置0.0461/0.0160，该样本不支持“几何梯度吞掉颜色裁剪预算”的假设。多尺度颜色输出头观察到非零梯度；不能由一个视角断言长期收敛正常或裁剪完全无影响。原生clip仍1e-4，未改网络/损失/参数；约4秒反向、峰值VRAM6646752256字节。进程结束后检查点完整SHA仍相同，没有新增探针脚本/报告或训练轮次。
- 新查原片：8视频顺序解码都只有99帧，首/末PTS为2046/202750 ticks，time_base=1/18432，全部98步严格2048ticks=1/9秒；各机容器头尾PTS一致。旧JPEG0066/0067在全部8机SHA相同；实际顺序解码并同像素矫正匹配确认旧capture65/66/67/68/69对应真实decode65/66/66/67/68。当前native去掉旧67后在该段正确连续，并不是源视频丢掉一帧。容器时间一致仍不证明物理触发或曝光一致。
- 用前次真实frame49蒙皮矩阵，纯NumPy把body/cloth canonical点及LBS投到8机原图，隔离了颜色网络；主要身体位置对应，局部轮廓偏差仍在。runtime/cache/review.png现在是这组模板点重投影图。runtime/cache/review.mp4为99帧原片8视角拼接，已验证播放到11秒及首尾画面，跨视角动作相位的完整审看尚未完成，不据此认定物理同步。临时本地HTTP服务已停止。
- 从此发现共用入口的真实问题：缺timeline时prepare_capture把旧清单frames=100误标actual_frames，默认end读到EOF并丢失PTS。现顺序探测实际帧数/整数PTS，保存必要时间NPZ并按源path/size/mtime复用；旧显式/inline时间轴保留来源。分割与instances共用读取，未改拟合算法。真实MJJ8机集成将100纠正为99、PTS与整数基准逐项一致、两下游一致、缓存命中不再解码；FCS006 inline和FCS008旧NPZ各8机保持300时刻且无需重测。内联替身、真实临时fixture、AST/Black/diff通过，无永久测试脚本或MJJ旧外参输入包留存。

## 清理与来源边界

- 用户手动执行scripts/cleanup_legacy.ps1后：lhmpp/runs、lhmpp/inputs、旧7个脚本已不存在；work除prior_control外无其他文件。保留目录612文件、134432521字节；衣物检查点SHA未变。场景人物输入及衣物source必要文件仍存在，嵌套生产路径也已核对；历史source字符串不要求旧文件存在。D盘可用约从60GiB升至86GiB。
- 代理只运行过脚本默认预览，未运行-Execute，不把用户手动删除说成代理执行。脚本默认预览，执行前后比较prior_control文件SHA256。
- 四个未跟踪旧副本仍留存：configs/paths.json、docs/README.md、docs/DESIGN.md、docs/DECISIONS.md。规范换行后与v1.0逐字节相同、生产没有引用；这四个明确路径的删除也被执行审查在启动前拒绝，仅返回blocked by policy，无具体原因，未删除。旧临时源码文本/过期委派日志也曾被拒绝。不要换工具/权限/父目录或扩展用户脚本绕过，不能宣称整个物理整理完成。
- 当前刚体入口仅消费选定mesh/轨迹。full-pipeline中的clean_helmet_motion是选定pose直接producer，需要清理前轨迹、原人体动作及phases(start/stop_exclusive/joint)，当前已检查位置未找到这些输入。baker也有后续修改，不能声称精确复现选定纹理。若发现阶段表/锚点/原壳/UV/分割/深度输入先保全，不伪造或从清理后轨迹反推自由运动。

## 下一步与续接

1. 先查真实Git、进程/GPU、C/D空间与检查点。当前没有训练/DeepSeek计算任务，视频服务已停止；源码最新发布以实际refs为准，本轮共享手观测复用和衣物模型所有权修正已审查及验证。
2. 23:23 GPU已空闲约96MiB、C约7.1GB，当前无项目计算任务。资源随用户应用变化，每次重查；此前重型CPU探针使C曾仅533MB，不重跑该方式，不终止用户应用、不改分页、不删C个人/系统文件。
3. 时间索引/原生算子/颜色梯度路径已核验，当前2000仍是未完成的早期图像训练。资源允许后按作者1000更新审看间隔继续到累计3000，随后全8机0/49/98和cam04完整99帧视频实际审看；这是完成原有RGB训练阶段，不改网络/FP32/损失/clip，也不预设画质达标。若启动，在覆盖新渲染前将现有2000 cam04视频作为唯一reference，替换已比较过的1800旧对照，保持单检查点和一份必要视频对照。保留Adam重置边界，不盲目排除曝光/拟合问题、不按loss采用。
4. 用户明确授权唯一每小时任务04-06续接，直到五小时额度接近用尽；不要重复建聊天/定时器。用户要求暂停时立即停用该任务并停工。额度未重置或接近耗尽不启动重任务、不频繁轮询，不用重置券，不标工程完成。
5. 复用指定GPT-6.1 Sol medium worker做明确任务；DeepSeek官方skill会话session-664dd3ce-67cf-4b51-8806-39db7ab9e3f4，只发代码/数值。成功须exit0且final，超时/权限/runner失败不循环修权限。主代理负责设计、独立审查、运行和实际看图。仅实质成果、失败或需用户行动时通知。
6. DeepSeek最新native_contract任务exit1/402 Insufficient Balance，final为空，未执行代码修改、无遗留进程；用户充值前不重复调用。现有worker删除了无调用的vector_angle_deg；等待后主代理接管并完成integration的_scalar及不可达step_only=True模式清理，False生产路径经常量折叠AST逐项一致，保留恢复/保存/中断语义，不重新预训练。
7. 衣物bootstrap不再强制MJJ的100维expression，按真实非空T,D矩阵的列数构造同一原生模型；模型若静默裁剪维度则明确报不匹配，不补零/截断。实际原生SMPL-X 10/100维CPU模型共享前10维的零/非零expression输出vertices/joints逐项相同、表达确实改变几何、CUDA未初始化；另有内联shape/factory/错误裁剪stub及AST/Black/diff检查通过，没有新增测试文件。当前MJJ100维计算不变，该通用输入修正不宣称改善网格画质。
8. 衣物GPU任务存活时保持codex/clothing检出，不切到main（衣物源码会移走）、不修改运行中源码。先核对真实进程/日志，结束后再做分支切换与共享修改发布。
