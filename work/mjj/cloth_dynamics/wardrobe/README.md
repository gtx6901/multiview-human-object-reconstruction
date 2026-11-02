# GaussianWardrobe — MJJ001 stage 1

This is a separate GaussianWardrobe implementation path alongside Animatable
Gaussians. It binds the existing MJJ001 assets without copying RGB/masks/models
or changing AG state. The public source is pinned to Wardrobe HEAD
`3e62461b724d546f368f72a6d5ef57a21c6e3635`.

## Source and runtime

- Wardrobe source: `wardrobe/author` (read-only; no assets downloaded)
- Shared MJJ001 input: `data/mjj001_animatable`
- Bound author layout: `wardrobe/data/mjj001`
- Local SMPL-X: `lhmpp/pretrained_models/human_model_files/smplx`
- Local extensions: `runtime/extensions`, `runtime/extensions/styleunet`
- Local deps: `lhmpp/deps`, `lhmpp`
- LPIPS linear weights: `work/mjj/runtime/perceptual/deps/lpips/weights/v0.1/vgg.pth`
  (fallback: Wardrobe `network/lpips/weights/v0.1/vgg.pth`)
- VGG trunk cache: existing `torch.hub` cache only; no download

## Stage 1 binding

Run from `wardrobe/`:

```powershell
python prepare_inputs.py --source ../data/mjj001_animatable --out data/mjj001
```

The script:

- resolves `--source` and `--out` before changing directory;
- reads `dataset_meta.json`, `calibration.json`, `smpl_params.npz`;
- preserves all 99 actual source frames at 9 fps and the 8-camera native
  2048x1536 calibration; no resize, rectification or person recenter;
- writes `cameras.pkl` (native `extrinsics`, `intrinsics`, `imgSize`, serial);
- binds `images`, `masks` and `smpl_params.npz` by NTFS hardlink;
- records pose-index -> actual source-frame mapping in `source_frames.json`;
- records inode identity, source/destination hashes and float64 projection
  invariance in `binding_manifest.json`;
- accepts **real** optional semantic data:
  `--labels <dir> --body-masks <dir> --body-template <ply> --cloth-template <ply>`.
  It never turns foreground masks into cloth labels, and never substitutes bare
  SMPL-X for observed clothing.

Current expected missing requirements (not generated):

- independent `labels_color/` (body/cloth semantic labels; source `--labels`
  with `camXX/%08d.png`)
- independent `body_masks/` (rendered/segmented body masks)
- `template_body.ply`, `template_cloth_offset.ply`
- `template_body_offset.ply`, `template_body_offset.pkl`,
  `template_cloth_offset.pkl`, beta-shaped templates
- `cano_weight_volume.npz`, `cano_weight_volume_shape_zero.npz`
- `smpl_pos_map_offset_body/`, `smpl_pos_map_offset_cloth/`
- `valid_faces.npy`, `with_face.npy`, `cano_smpl_offset_map.exr`,
  `cano_smpl_segment_map.exr`, `cano_smpl_smplx_fix_offset_map.exr`,
  `cano_smpl_hand_map.exr`, canonical position/normal maps

## Adapter and preflight

`integration.py` installs author-first imports, Wardrobe `smplx`/`smplxd`
classes, local 100-D SMPL-X and Unicode-safe OpenCV read/write. It compares the
Wardrobe extension source trees with AG source trees and only reports
`reuse_compatible` when source hashes match; it does not blindly import a
same-named pyd.

The official Wardrobe 4DDress reader hardcodes 940x1280/`%05d.png`; the Other
reader hardcodes 1024/cameras/*.json. Neither matches native 2048x1536 MJJ.
`MvRgbDatasetMJJNative` is a thin adapter subclassing the actual
`MvRgbDataset4DDress`/`MvRgbDatasetBase`; it reads `cameras.pkl`, native
image/mask/label paths, and for `use_label=True` adds `label_valid_img`:
`(~instance_mask) OR (semantic_valid > 0)`. Instance exterior remains valid
background; foreground parser class 0 (unknown) is invalid for label
supervision. It does not replace `mask_img`, does not create body masks, and
does not alter RGB/LPIPS/geometry. The config uses this adapter name; it must be
installed before training:

```python
from wardrobe.integration import install_native_reader
install_native_reader()
```

CPU preflight:

```powershell
python integration.py --doctor
python integration.py --reader-check
```

`--doctor` reports exact present/missing semantic, template, volume, map and
runtime assets and extension compatibility. `--reader-check` instantiates the
adapter and verifies one actual image/mask/SMPL item on CPU. When real
`labels_color` and `semantic_valid` both exist it sets `use_label=True` and
reports the reader's `/255` labels, unique disk rows, background-valid,
known-foreground-valid and unknown-foreground-invalid counts. Otherwise it keeps
the old image-only check. It still reports `layers_loaded=false`; one reader item
is not multi-layer training.

Real author position maps (GPU; no CPU fallback):

```powershell
python integration.py --gen-pos-maps --layer both
python integration.py --gen-pos-maps --layer body
python integration.py --gen-pos-maps --layer cloth
```

This keeps `--doctor`/`--reader-check` unchanged and calls only the actual
author `gen_data.gen_pos_maps` through
`runpy.run_module(..., run_name="__main__")` with explicit `sys.argv`:
`-c <config> -d <absolute data> -o smpl_pos_map_offset_<layer>
-t template_<layer>_offset -rc -ro -lw cano_weight_volume_shape_zero`.
`both` runs body then cloth serially in one runtime. It does not rewrite the
author renderer, normals, LBS, neighbor/face/offset or 512->1024 code. The
process `sys.argv`, cwd and `sys.path` are restored afterward.

Before calling, the production entry requires the actual
`template_body_offset.ply`, `template_cloth_offset.ply`, their matching `.pkl`
files, `cano_weight_volume.npz` and
`cano_weight_volume_shape_zero.npz`. Missing inputs fail instead of allowing the
author script's bare-SMPL fallback (which is wrong for cloth). Images, models
and templates are not copied.

After the calls, `layer_geometry_manifest.json` gains a `position_maps` field
with input SHA-256, author/integration source hashes, model hash, exact calls,
actual output counts and missing paths. It preserves existing votes,
canonicalization and body-mask generation fields. The author script does not
generate `cano_smpl_hand_map.exr`; `MultiLAvatarNet` also needs that body hand
mask. Cloth `cano_smpl_segment_map.exr` and body
`cano_smpl_smplx_fix_offset_map.exr` are generated. An external
`smplx_masks/cano_smpl_smplx_map.exr` is checked but never copied or synthesized.
Missing masks remain listed until Codex verifies canonical projection/alignment;
no empty or all-one fake masks are created. Existing body-mask supervision stays
white body / black background, not the blue fallback.

Local minimal author adaptation: `main_avatar.img_loss` has optional
`valid_mask_img`; when supplied it computes
`sum(|pred-gt| * valid) / (valid_pixel_count * channels)`. All-ones valid is
exactly the old L1. Only `consistent_label` passes
`items.get('label_valid_img')`; body-label, RGB, LPIPS, mask and geometry losses
are unchanged. A mask does not alter the prediction, only excludes unknown
observations from label supervision.

## Config

`configs/mjj001.yaml` mirrors `configs/4d_dress/avatar.yaml`:
8 cameras, 99 times, official two layers `["body","cloth"]`, `MultiLAvatarNet`,
`coverage/offset_mode original`, 512 StyleUNet input / 1024 canonical output,
official losses (`l1`, `lpips`, `face_lpips`, `ssim`, `offset`, `laplacian`,
`body_loss`, consistency/opacity terms), optimizer LR `.00015`, checkpoint/eval
intervals and 1.0 image factor. Only subject/data paths, 8 camera ids and 99
times are adapted. `num_workers: 0` is a Windows process setting.

## Required author algorithms (review contracts)

- `gen_data/create_mesh_avatar.py::prepare_template_data_smplx`: canonical
  body/pose template and shape offset.
- `gen_data/gen_weight_volume.py`: separate original-shape and shape-zero
  canonical body/cloth support volumes; both are prerequisite artifacts, not
  whole-person foreground substitutes.
- `gen_data/create_mesh_avatar.py`: label/cloth mesh inverse LBS
  (`skinning_mask`, `query_weights`, `diffuse_weights_smplx`,
  `diffuse_offset_smplx`, `knn_weights`) to zero-shape canonical templates.
- `gen_data/cano_to_shaped.py`: restores canonical templates to the beta-shaped
  space using `shape_offsets` and `query_weights`.
- `utils/smpl_util.py::skinning` and
  `network/multi_layer_avatar.py::MultiLAvatarNet.transform_cano2live`: layer
  LBS with per-layer weight volumes/maps. These are the algorithmic contracts
  to review next; no plane/clothes proxy is a substitute.
- `network/multi_layer_avatar.py`: body/cloth/outer visibility and label
  composition (`render_filtered`, `filter_gaussian`).

## SCHP semantic parsing (measurement preparation)

Production parser adapter: `segmentation.py`. It uses the pinned official SCHP
source (`wardrobe/parser`, HEAD `eb84c432cc697f494d99662a05f2335eb2f26095`)
and the local `runtime/parser/lip.pth` (267,449,349 bytes). It does not train,
download, upload, or modify pixel content.

```powershell
python segmentation.py `
  --data data/mjj001 `
  --checkpoint ../runtime/parser/lip.pth `
  --device cuda
```

Optional real pose subset:

```powershell
python segmentation.py --data data/mjj001 `
  --checkpoint ../runtime/parser/lip.pth --device cuda `
  --frames 0,49,98
```

The SCHP LIP-20 input is BGR with official mean/std
`[.406,.456,.485]/[.225,.224,.229]`, 473x473 bilinear `align_corners=True`,
and the official inverse affine (`transform_logits`). The crop center/scale is
derived from the existing SAM2 foreground bbox using the original
`_xywh2cs`/`get_affine_transform` convention. No K change, no additional resize,
no synthesized fill.

The original InPlaceABNSync CUDA extension is bypassed only for inference by a
process-local, state-key-compatible module:
`effective_weight=abs(weight)+eps; mul=rsqrt(running_var+eps)*effective_weight;
y=(x-running_mean)*mul+bias;` then the original activation
(`none`/`leaky_relu` slope .01/`elu`). The checkpoint is loaded strictly; only
`module.` prefixes are stripped. No missing layer is silently ignored.

Outputs, only these three training/measurement forms:

- `parsing/<cam>/<pose>.png` — raw 20-class LIP labels;
- `labels_color/<cam>/<pose>.png` — **disk uint8 BGR** body `[0,0,255]`
  and cloth `[0,255,0]`; after the author reader divides by 255, training
  float BGR is exactly body `[0,0,1]` and cloth `[0,1,0]`. PNG read-back is
  verified exactly;
- `semantic_valid/<cam>/<pose>.png` — 0/255, true only for SAM instance
  foreground with raw class > 0.

Class mapping:

- body-visible: 2 hair, 4 sunglasses, 13 face, 14/15 arms, 16/17 legs,
  18/19 shoes;
- cloth: 1 hat, 3 glove, 5 upper-clothes, 6 dress, 7 coat, 8 socks,
  9 pants, 10 jumpsuits, 11 scarf, 12 skirt;
- class 0 inside the SAM foreground stays UNKNOWN; no threshold, no forced
  body/cloth relabel, no hole fill/filter. The reader exposes it through
  `semantic_valid` and `label_valid_img`, not as background.

`segmentation_manifest.json` records model/source hashes, the
`segmentation.py` self hash in the algorithm key, real source frames, class
mapping, output coverage, per-item input hashes, elapsed time and CUDA peak
memory. The manifest is atomically updated after each completed item, so an
interrupted run can resume. Existing outputs are reused only when the
algorithm/model/input run key and per-item input hash match; changed
inputs/models overwrite current outputs in place. Expanding `--frames` under the
same run key preserves previously completed items. No version directories are
created. A single local `segmentation_contact_sheet.png` is created for review
when frames 0/49/98 exist: top is the original crop, bottom is a clear
body-red / cloth-green / unknown-gray semantic display; it does not modify
training pixels.

This remains measurement preparation, not multi-layer Wardrobe training,
template/weight-volume construction, or visual quality acceptance. Parser
`labels_color` is SCHP-derived supervision, not independent cloth segmentation,
and visible skin classes are not a complete author `body_masks` target.

## Beta-shaped layer geometry front-end

`layer_geometry.py` builds target-Wardrobe beta body/cloth fronts from the
existing AG canonical whole-surface template and real 8-view SCHP semantics. It
does not train and does not create a shape-zero weight volume.

```powershell
python layer_geometry.py `
  --config configs/mjj001.yaml `
  --data data/mjj001 `
  --template ../runs/mjj001/template_convergence/eval/template_00003960.ply `
  --weight-volume ../data/mjj001_animatable/cano_weight_volume.npz `
  --source-canonical-config ../author/config.py `
  --target-weight-volume data/mjj001/cano_weight_volume.npz `
  --device cuda
```

`--weight-volume` is the source AG volume used only for source-coordinate
weights. `--target-weight-volume` (default `data/mjj001/cano_weight_volume.npz`)
is mandatory for actual target-coordinate LBS and is never replaced by the
source volume.

Optional real pose subset:

```powershell
python layer_geometry.py `
  --config configs/mjj001.yaml `
  --data data/mjj001 `
  --template ../runs/mjj001/template_convergence/eval/template_00003960.ply `
  --weight-volume ../data/mjj001_animatable/cano_weight_volume.npz `
  --source-canonical-config ../author/config.py `
  --target-weight-volume data/mjj001/cano_weight_volume.npz `
  --device cuda `
  --frames 0,49,98
```

- Uses `integration.bootstrap`, `install_native_reader`, and the real
  `MvRgbDatasetMJJNative` reader with `layers=None, use_label=True,
  load_smpl_pos_map=False, use_body_label=False, mode='3dgs'`.
- Reads the AG source canonical config under an independent importlib module
  name; it does not overwrite the target author `config` or zero the target
  wrist constants 57/60.
- The actual SMPL-X (real beta, `flat_hand_mean=True`, expression/jaw 0) is
  forwarded with source and target canonical poses to obtain `A_source` and
  `A_target`. The author relative-LBS conversion is
  `T_j = A_target_j @ inv(A_source_j)` and
  `V_target = sum_j w_source_j * (T_j * [V_source,1])`. Face topology is
  unchanged and the source template is never modified.
- The source AG volume is queried only on source vertices. The target Wardrobe
  volume is required and is queried on `V_target`; a missing target volume is a
  hard failure, not a source-volume fallback.
- Per frame uses target `cano2live_jnt_mats` linear blend of `V_target` with
  `w_target` to current world meters.
- Per camera uses PyTorch3D `cameras_from_opencv_projection` with actual R/t/K,
  native HxW, `MeshRasterizer` with `faces_per_pixel=1`, `blur_radius=0`,
  `cull_backfaces=False`; no projected-vertex/backface-only voting.
- Only known body/cloth foreground pixels vote: disk BGR body `[0,0,255]`,
  cloth `[0,255,0]`; unknown and outside do not vote.
- `cloth_faces = cloth_votes > body_votes AND cloth_votes > 0`;
  equal/zero body-vs-cloth faces remain unassigned. No fill/smooth/hole repair.
- Body output is the target reader's SMPL-X canonical body prior, explicitly not
  the occluded body observation. Cloth output uses converted `V_target` and
  selected source faces; no hole sealing.

Fixed beta-shaped vote outputs in `--data`:

- `beta_template_body.ply`
- `beta_template_cloth.ply`
- `layer_geometry_votes.npz` (faces, raw body/cloth votes, body/cloth/unassigned masks)
- `layer_geometry_manifest.json`

The manifest records source/target canonical pose vectors, the actual
global/body/full-pose angle differences, source/target config SHA, model SHA,
source/target template/volume SHA and the relative-LBS conversion. Source and
target volumes are loaded sequentially; both are not held resident together.

Canonicalization runs after voting and can also be invoked without re-voting:

```powershell
python layer_geometry.py `
  --config configs/mjj001.yaml `
  --data data/mjj001 `
  --canonicalize-only `
  --device cpu
```

`--canonicalize-only` does not require `--template` or `--weight-volume`; it
reuses `beta_template_body.ply` and `beta_template_cloth.ply` and does not
re-run rasterization or voting. It writes:

- `template_body_offset.ply`
- `template_cloth_offset.ply`
- `template_body_offset.pkl`
- `template_cloth_offset.pkl`

Canonicalization uses the target Wardrobe vendored SMPL-X model, target
`config.cano_smpl_*` pose/global/transl, `flat_hand_mean=True`, zero expression,
real beta and zero beta: `V_beta` and `V_zero`. Local author interface fix:
`MvRgbDatasetBase.getitem` canonical `cano_smpl` now omits only the frame
`jaw_pose`/`expression`; live SMPL and live-no-root still carry the real
per-frame jaw/expression/hands. The target wrist constants 57/60 are preserved;
the AG source template is converted into this target canonical space. The body
prior is the target reader's zero-expression `cano_smpl` and must numerically
match `V_beta`; the difference is reported and no silent translation/scaling is
applied. `body_zero = V_zero`, `body_offset = V_beta - V_zero`. CUDA peak memory
is reset before bootstrap/volume load and includes the full vote, conversion and
canonicalization process.

For `cloth_beta`, the full shape displacement `D = V_beta - V_zero` is
interpolated at query vertices using the author K=30 inverse-distance math
(`knn_points`, `sqrt(dist).clamp(0.0001,1)`, normalized `1/dist`), in chunks of
4096. The reference is `V_beta`; `cloth_offset = sum(w * D[idx])`,
`cloth_zero = cloth_beta - cloth_offset`. This avoids the author 256^3 fixed
[-1,1] grid clamp. It is an explicit spatial mapping with representation
differences from the author's grid method, not claimed per-pixel equivalence.
The full canonical-pose shape difference includes joint-center changes.

The old `beta_template_body.ply` produced before this reader fix may contain
frame-0 jaw/expression and is not valid input for the new canonical contract.
Do not offset or relax the body-prior check. Re-run the normal 99x8 vote command
so it overwrites the beta templates and automatically runs the new
canonicalization; `--canonicalize-only` on an old file will intentionally fail.
The manifest records the dataset source path/hash alongside the SMPL-X/model
hash and canonicalization fields.

PLY faces/vertices preserve order with `process=False` and no unreferenced
vertex removal; offsets are `np.float32 (N,3)` and satisfy `beta = zero +
offset`. Body zero is the model zero-shape prior, never an inferred observation
of the occluded body. The unique `layer_geometry_manifest.json` records all
canonicalization hashes, pose/beta, formula, roundtrip maxabs and prior
boundaries; no version directory is created.

These are beta-to-zero template assets, not a zero-shape weight volume or layer
position maps. If no cloth face has observed support, the vote command fails
rather than inventing cloth. Multi-layer training and clothing quality
acceptance remain future steps.

### Body-mask production

Body masks are generated from the actual Native reader's live SMPL-X geometry,
not from SAM masks, skin labels, or pixel fitting:

```powershell
python layer_geometry.py `
  --config configs/mjj001.yaml `
  --data data/mjj001 `
  --body-masks-only `
  --frames 0,49,98 `
  --device cuda
```

Omit `--frames` for all 99 x 8; `--frames` accepts a comma list. This mode is
mutually exclusive with `--canonicalize-only`, does not require
`--template`/`--weight-volume`, and does not re-vote or canonicalize. It uses
`layers=None, load_smpl_pos_map=False, use_label=False,
use_body_label=False`, actual `live_smpl_v`/`smpl_faces`, `flat_hand_mean=True`,
and the native 8-camera R/t/K. PyTorch3D rasterization with
`faces_per_pixel=1`, `blur_radius=0`, `cull_backfaces=False` yields
`mask=(pix_to_face>=0)`.

Fixed output is `body_masks/camXX/%08d.png`: 3-channel uint8, body prior 255
white, background 0. One `body_masks_contact_sheet.png` shows the original
image plus a cyan body-prior contour labelled `PRIOR`; it does not modify saved
masks and creates no per-frame raster review copies. The existing
`layer_geometry_manifest.json` gains `body_mask_generation` and preserves votes
and canonicalization; no new report version is created.

The white target corresponds to the configured
`network.multi_layer_avatar.MultiLAvatarNet.render` body Gaussian colors (1.0),
independent body render background 0, and binary-white `label_body_rgb_map`, not
the blue one-hot fallback. The mask is a full-body geometric shape prior that
includes the unknown body under clothing. It is **not** observed naked-body
ground truth and is not a SAM/skin segmentation.

The author `main_avatar.forward_one_pass` total-loss accumulation for SSIM and
offset now uses the real tensors (`ssim_out`, `offset_loss`); logging and
`batch_losses` still use `.item()`. No loss weight or objective was added.

## Shared real weight-volume production

`resume_weight_volume.py` (at `cloth_dynamics/`) is now the single production
entry for AG and Wardrobe canonical weight volumes. It runs the original author
Poisson pipeline on CPU; it does not use the old cached 55-grid mode.

```powershell
# AG beta-shaped volume
python ../resume_weight_volume.py `
  --backend ag `
  --config ../configs/mjj001_template.yaml

# Wardrobe beta-shaped volume
python ../resume_weight_volume.py `
  --backend wardrobe `
  --config configs/mjj001.yaml

# AG zero-shape volume
python ../resume_weight_volume.py --backend ag `
  --config ../configs/mjj001_template.yaml --zero-shape

# Wardrobe zero-shape volume
python ../resume_weight_volume.py --backend wardrobe `
  --config configs/mjj001.yaml --zero-shape

# No 55 solves, no output
python ../resume_weight_volume.py --backend ag `
  --config ../configs/mjj001_template.yaml --dry-run
```

The command uses real canonical beta (or beta=0 with `--zero-shape`), fixed
canonical pose/global/transl, zero expression, real author LBS gradients, depth
7, gradientWeight 0.05, real PointInterpolant subprocess solves, the author
clip/normalization/xyz transpose, author bounds and real libigl SDF +
barycentric LBS. It uses only the three fixed scratch files
`current_values.xyz`, `current_gradients.xyz`, `current_grid.grd`; no
`grid_00..54` cache is created or reused. A failed solve is not allowed to use a
previous joint grid.

Outputs (one current file each) are
`data/cano_weight_volume.npz` or `data/cano_weight_volume_shape_zero.npz`, with
the original six fields and atomic replacement. A single
`runtime/results/weight_volume.json` report overwrites in place and records
source/model/input/PointInterpolant hashes, 55 solves, shapes and finite/
normalization diagnostics (not quality). `--dry-run` performs no solves and
writes no volume.

Wardrobe CPU startup uses `WARDROBE_DEVICE=cpu`; `integration.bootstrap` sets it
before importing author config so `SURFACE_LABEL_COLOR` does not allocate CUDA.
The AG backend reuses the existing bootstrap and forces the process config
device to CPU.

## Current execution state (2026-11-02)

Both layers have all99 native position maps and true canonical/normal/offset/LBS
inputs. SCHP 8x99 labels use the author's numeric body/upper/lower palette.
Semantic voting is complete: exact source face ordering transfers upper/lower
labels to the canonical cloth atlas. Equal nonzero votes use ordinary categorical
argmax with explicit tie provenance; every current cloth face has votes.
Body hand support derives from actual LBS, and body fix support represents the
entire explicit SMPL-X prior, not observed skin under clothing.

The author generator now flips back face IDs together with back attributes.
Both existing neighbor arrays were regenerated with that corrected correspondence
and the original stencil/algebra. Original position/normal/offset/pose maps and
geometry are unchanged. Zero-shape atlas cameras match original generation.
`training_masks.status` is generated. The actual input preview is
`../runtime/results/wardrobe_masks.jpg`; it is not a trained dynamic reconstruction.

The original FP32 network is running5001 native pretrain updates, then1000 real
RGB/label updates at the author's first evaluation interval. It uses native Adam
CPU state and UNet activation checkpointing, with unused color nets on CPU only
during pretraining. One rolling `current/training.pt` saves model/Adam/cursor/RNG;
there are no epoch copies. Live progress is `../runtime/results/wardrobe_training.log`.

After the actual image checkpoint, queued rendering calls:

```powershell
python integration.py --render --config configs/mjj001.yaml --frames 0 49 98 --video --video-camera 4
```

This produces all8 native-camera keyframes and all99/9fps on the verified front
camera FCM22020159. Video display is resized after native rendering and fully
decoded to check completeness. The output is one `current/review` directory.
Rendering requires an actual `author_train` checkpoint; no pretrain/random-color
image is treated as the dynamic result. No visual acceptance is claimed yet.

The latest run/queued process identities are recorded in `../../ACTIVE_RUN.md`.
