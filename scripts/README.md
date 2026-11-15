# 工程脚本

重建统一使用项目根目录的 `pipeline.py`，安装后也可使用 `hoi`。实现位于 `src/hoi`，场景差异由 `configs` 表达，完整用法见 [管线说明](../docs/PIPELINE.md)。

`cleanup_legacy.ps1` 供用户手动清理已授权的旧文件。默认只显示清单；显式传入 `-Execute` 才永久删除旧 `work`（保留 `work/mjj/two_person/data/quaternion_fit/prior_control`）、`lhmpp/runs`、`lhmpp/inputs` 和旧七个运行时脚本。原始数据、当前资产、当前输出、权重、报告和 Git 历史不在清理范围内。

```powershell
.\scripts\cleanup_legacy.ps1
.\scripts\cleanup_legacy.ps1 -Execute
```

脚本检查绝对路径与目录链接，排除保留目录及其祖先；执行前后比较保留文件的 SHA256，遇到错误立即停止。它不会改变系统权限或执行策略。

衣物探索分支另有 `build_clothing.ps1`，用于构建该分支的外置依赖。旧源码历史保留在 Git 标签和提交中，不恢复到当前生产脚本目录。
