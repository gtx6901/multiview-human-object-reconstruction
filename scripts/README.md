# Baseline scripts

当前基线只保留“从冻结资产渲染最终结果”的 7 个脚本。

## Runtime / render

- `hoi_paths.py`：统一路径解析（`configs/paths.json`）
- `avatar_rig.py`：把 6D 姿态应用到 LHM 高斯骨架
- `mesh_renderer.py`：头盔网格可微渲染
- `hybrid_compositor.py`：人体高斯与头盔的两层深度合成
- `render_gtcam_video.py`：同机位 300 帧渲染
- `render_orbit_video.py`：环绕新视角 300 帧渲染
- `compose_four_views.py`：GT / 同机位 / 固定新视角 / 旋转新视角四视图合成

## 不在当前工作区

拟合、标定、分割、关键点、头盔几何/贴图生成、环境安装脚本不在当前基线目录。
完整代码保留在 git tag `full-pipeline`，需要重跑生成链时从该 tag 取回。

```powershell
git show full-pipeline:scripts/fit_motion.py
```

需要试验时，只提取所需文件到 `work/` 隔离目录；不要把整个 tag 的
`scripts/` 检出到当前基线，否则会破坏 7 个运行时脚本的约定。
