# 工程交接

更新：2026-11-16。先读最新人类消息，核对真实进程/GPU、日志、检查点、C/D空间与Git；保留活任务，不重复启动。衣物任务的当前运行状态只在clothing分支的`docs/CLOTHING.md`开头维护，避免两处快照矛盾。

## 工程与分支

`main`保留一套单人/多人共用的人体管线：入口`pipeline.py`/`hoi`，源码`src/hoi`，配置`configs`，输入`assets`，输出`runs/<scene>/<actor或用途>`，第三方与权重外置。同一算法处理不同数据，差异由配置表达。main现有55个跟踪文件、34个src模块，含入口9240行（含空行/注释）。

main的`af372b0`已发布，通用初始化修正位于`c43a36d`；clothing的`e285ec6`已合并该main并发布。后续以真实refs为准。衣物实现留在`codex/clothing`，运行状态、命令与恢复记录见[该分支衣物说明](https://github.com/gtx6901/multiview-human-object-reconstruction/blob/codex/clothing/docs/CLOTHING.md)。共享修改先进入main，再合入探索分支；衣物训练活着时保持当前checkout，不切换导致源码移走，也不修改运行源。

文件/函数snake_case，类PascalCase，产物按用途命名。原位迭代，历史进入Git，不叠版本目录，不留无用检查/门限脚本、失败轮次或重复缓存；约一万行不是删除必要功能的理由。只提交明确文件，定期同步remote，不force、不git add .。本机依赖与配置见[DEPENDENCIES.md](DEPENDENCIES.md)。

## 人体管线的实际验证

FCS008已完成八机位、300源时刻的完整原片prepare、新NLF和真实图像手观测，两人均按身体三阶段400/200/200步、脸200步、MAP600步完成拟合。全8机0/149/299和正面300帧30fps视频已实际审看，未见明显大幅抽动或整体形态回退；抓握接触、脸手外观仍不足。当前图像/视频位于`runs/fcs008/render`，`review.png`为关键帧合图，render报告的动作SHA已核对。

该轮复用已选avatar；此前也已通过统一入口，从七个真实可见视角重新生成partner身份并完成完整拟合。FCS006单人含头盔及固定新视角已实际回放。不能把缓存身份的验证说成新受试者身份也从零重建完成。

已修正两项共用入口问题：显式请求完整片段时，不再复用三帧、乱序或多帧的观测缓存；初始化不再只认双髋或自动读取旧motion。固定NLF姿态和avatar尺度，用有效BODY关节沿现有0.02尺度的径向Huber目标求初始平移，手从avatar姿态开始，再根据图像手观测拟合。既有动作只能通过显式initialization复用。内部归一化四元数，时间约束作用于相对旋转；二维资格独立于三角化inlier，三维仍用valid与有限坐标。

缺少时间轴时已改为顺序探测实际帧数和整数PTS，不用旧清单声明帧数合成时间；MJJ原片实际99帧，旧JPEG曾在66附近重复一帧。容器PTS一致不代表物理曝光同步。具体输入、缓存、源帧和坐标约定见[PIPELINE.md](PIPELINE.md)。

新问题先查输入、算法与实现，不加像素补丁、关节锁或画质门槛，不按loss采用。明显画面问题不反复做无必要定量试验。`render/contact.png`是21305717机位149帧的原图、手部二维观测、最终FK投影和外观近景；原图有遮挡与模糊，此单帧不能解释全部接触误差。选定资产及已交付PPT/GIF不覆盖。

## 保护、清理与来源

用户已手动清理旧work试验、lhmpp/runs、lhmpp/inputs和7个旧脚本；一次性清理脚本已退役。work只保留`mjj/two_person/data/quaternion_fit/prior_control`祖先链，共612文件、134432521字节，保持保护。

仍有以下拒删文件：`configs/paths.json`和`docs/README.md`、`docs/DESIGN.md`、`docs/DECISIONS.md`是4个无生产引用的未跟踪旧副本；`runtime/cache/optimizer_control`为已否定的epsilon控制，66文件、10793937102字节。删除被执行审查在启动前以blocked by policy拒绝，没有具体理由。旧临时源码文本/过期委派日志也曾被拒删；不换工具、权限或父目录绕过，不宣称整个物理清理已完成。

保护原始视频/图片/标定、必要依赖/权重、选定资产、PPT/GIF及v1.0/full-pipeline。assets/runs/lhmpp/runtime/data不加入Git跟踪。临时缓存放D盘，不终止用户应用、不改分页、不删C盘个人/系统文件。

当前刚体入口消费已有mesh和轨迹。头盔原producer还需要清理前轨迹、当时人体动作及phases(start/stop_exclusive/joint)表，已检查位置未找到；纹理baker也有后续修改。发现原壳/UV/深度/锚点等来源先保全，不从清理后轨迹反推自由运动或声称精确复现。

## 自动续接与协作

只使用已有每小时任务04-06，不创建聊天、定时器或守护脚本。用户要求停止/暂停时立即停用续跑并停止自有工作；恢复后继续原目标，不在完成阶段、交接或推送后自行结束全部工作。

实时区分primary五小时与secondary周额度。仅周剩余严格低于4%时可使用可用免费reset，前后核实；primary不足不能代替该条件，不付费、不调用私有接口。11月16日已使用一次免费reset，最近核实还剩1次，实际余额以官方工具为准。

复用指定GPT-6.1 Sol medium worker做明确的确定性任务，Codex设计、审查、执行与看图。worker工具曾等待审批，目前只要求文字patch/审查，不修改子会话权限或代点审批。DeepSeek既有官方会话session-664dd3ce-67cf-4b51-8806-39db7ab9e3f4最新返回402，用户充值前不重试；恢复后按本地skill，仅发送代码/数值，exit0且final才算完成。
