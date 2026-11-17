# 虚拟场景合成

把已经拟合好的人体 Gaussian 资产放进 glTF 场景，加入光源和阴影，渲染单帧或视频。此入口不重新重建身份或动作，只改变场景、摆位、机位和光照。

## 命令

```powershell
python pipeline.py render --config runs/fcs008/config.json `
  --environment configs/environments/dust2.json `
  --frames 0 149 299 --video
```

`--config` 提供各人物的 `avatar` / `motion`；`--environment` 指定场景规格。输出在 `runs/<scene>/environment/<规格名>/`，视频按统一规则写入 `runs/delivery/review/`。不加 `--video` 时逐帧写 PNG；加 `--video` 时只保留首/中/尾三张审看图。

## 场景规格

```json
{
  "model": "assets/environments/dust2/dust2_a.webp.glb",
  "model_scale": 1.0,
  "placement": {
    "position": [-19, 0, -6.8],
    "yaw_degrees": 240,
    "meters_per_body_height": 1.75,
    "snap_to_floor": true,
    "reference_serial": "21305717"
  },
  "camera": {"eye": [-5, 2.5, -1.0], "target": [0.4, 0.9, 0], "vertical_fov": 46, "width": 1260, "height": 756},
  "lighting": {
    "position": [-26, 14, 1.5],
    "target": [-19, 0, -6.5],
    "color": [1.0, 0.93, 0.83],
    "intensity": 1.0,
    "ambient": [0.38, 0.43, 0.52],
    "shadow_resolution": 1024,
    "shadow_bias": 0.02
  },
  "background": [0.62, 0.76, 0.91]
}
```

- `model` 是 glTF/GLB，相对项目根解析；同一材质的网格合并成一次光栅化。
- `placement.position` 是人物组在世界中的落点，`yaw_degrees` 绕竖直轴旋转，`meters_per_body_height` 把归一化人体缩放到场景单位。`snap_to_floor` 用场景三角面把脚底贴到地面。
- 多人保持原始相对位置，作为一个整体放置；人物之间的动作关系不变。
- `camera` 用相机位置、注视点和竖直视场角定义，宽高可被命令行覆盖。
- `background` 是天空/清屏色，只在场景没有覆盖的像素可见。

## 渲染行为

1. glTF 按材质分组，读取基础色贴图和基础色因子，转线性空间。
2. 从相机光栅化场景，得到深度、世界坐标、法线和 albedo。
3. 人体高斯用同一个 gsplat 光栅化器渲染，得到 RGB、alpha 和 expected depth。
4. 人物在光源空间的 expected-depth 阴影图与场景本身的阴影图叠加，按法线做 Lambert 漫反射，再加环境光。
5. 人体与场景按深度合成；场景没覆盖的像素用 `background`。

人体外观仍由冻结的 DPT 解码器生成，包含拍摄时的光照，不做 PBR 重光照；场景光源只影响场景表面和人物投下的阴影。阴影是屏幕空间/光源空间的近似，不是精确体积阴影。

## 坐标与尺度

场景模型多为摄影测量或低模，没有可靠的米制尺度。`meters_per_body_height`、`position`、`yaw_degrees` 是人工放置参数，不是测量结果；报告里记为视觉摆位。相机使用 OpenCV 约定（x 右、y 下、z 前），与主渲染器一致。

## 当前场景

| 名称 | 文件 | 状态 |
|---|---|---|
| dust2 | `configs/environments/dust2.json` | 可用；A 点社区复刻，34 MB GLB，MIT/CC0 贴图 |
| times_square | `configs/environments/times_square.json` | 未采用；城市级摄影测量，街面近景不可用 |

外滩：没有找到可用的开源街景模型，候选与许可证见 [bund_asset_survey.md](bund_asset_survey.md)。
