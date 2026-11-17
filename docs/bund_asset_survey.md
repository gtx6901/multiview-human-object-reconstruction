# 外滩 / Shanghai open-3D-asset survey
Scope: glTF/GLB-usable environment for 2 × 1.75 m reconstructed humans at street level,
1260×756 render, nvdiffrast + gsplat, RTX 4060 Laptop 8 GB / 16 GB RAM.
Compiled 2026-11-18. Licenses verified on-page / via Sketchfab API v3 where noted.

## Bottom line

1. **There is no open-licensed (CC0 / CC-BY / PD) street-level 3D model of 外滩 The Bund.**
   Querying Sketchfab's API (`downloadable=true`, `tags=shanghai`) returns exactly **3**
   downloadable architecture models, and none is the Bund waterfront promenade.
   Everything Bund-specific that exists is either CC BY-**NC-ND** or a paid Sketchfab
   Store / 3DExport item.
2. The one genuinely valuable open asset is **Poly Haven's CC0 "Shanghai Bund" HDRI** — a
   360° night capture taken *on* the Bund promenade facing Lujiazui. It gives you the
   authentic neon skyline, river reflections and warm promenade light for free, but it is
   lighting/backdrop, not geometry.
3. Practical recipe: **CC0 Bund HDRI (backdrop + IBL) + self-built promenade floor/railing
   + OSM-derived Bund building massing + CC0/CC-BY street props.** This is the only route
   that is both legally clean and visually recognisable as 外滩.

## Ranked candidates

| # | Asset / author | Source | License | Format · size · tris · textures | Street-level or skyline only | Score | Main risk |
|---|---|---|---|---|---|---|---|
| 1 | **Shanghai Bund HDRI** — Greg Zaal | https://polyhaven.com/a/shanghai_bund | **CC0** | 4K EXR 21.71 MB (1K/2K/8K/16K, EXR+HDR also offered; 8K tonemapped JPG 4.76 MB) · no geometry · night, 17 EV, 3300 K | 360° backdrop captured at promenade height; skyline across the river + promenade light | **9/10** as backdrop + IBL; 0 as geometry | Baked night lighting only; no relighting; 4K equirect is soft at 40° FOV; no ground plane in the pano |
| 2 | **Bund block rebuilt from OSM** — OpenStreetMap + Blender-GIS | https://www.openstreetmap.org · https://github.com/domlysz/BlenderGIS | Data **ODbL** (attribution + share-alike on derived DB); tool GPL | Footprints extruded to `height`/`building:levels`; export glTF from Blender; ~50–150 k tris for the 1.5 km Bund frontage; you supply textures | **Yes** — real footprints of 和平饭店 / 海关大楼 / 汇丰银行 / 外滩源 + raised promenade edge | **7/10** | Box massing only; needs facade textures and roof detail; ODbL share-alike |
| 3 | **City- Shanghai-Sandboxie** — Michael Zhang (`beyond.zht`) | https://sketchfab.com/search?q=City-%20Shanghai-Sandboxie (uid `3eab4438b9b34ceeaa35367429732970`) | **CC BY 4.0** | GLB **47.96 MB**, 417 252 tris (site 418 402), **0 textures / 0 px** | City-massing of central Shanghai incl. river; ground plane present but *untextured white* | **5/10** | Completely untextured — you must author materials; check in viewer whether the promenade deck is modelled |
| 4 | **Kenney City Kit (Commercial)** + **City Kit (Roads)** | https://kenney.nl/assets/city-kit-commercial · https://kenney.nl/assets/city-kit-roads | **CC0** | OBJ/GLB, 50 files, metres-scale kit, vertex-colour flat materials, <10 MB per kit | **Yes** — road tiles + sidewalks give a clean ground plane and mid-ground | **6/10** | Not Shanghai at all; cartoon/clean look, only usable as scaffolding or stylised fallback |
| 5 | **Asian Themed Low Poly Night City Buildings** — `99.Miles` | https://sketchfab.com/search?q=Asian%20Themed%20Low%20Poly%20Night%20City%20Buildings (uid `9f0343aff4814b758dc6e905aba5b5e0`) | **CC BY 4.0** | GLB 22.33 MB · 6 435 tris · 12 tex @ 2048 px (+ emissive variant) | Skyline only — 19 loose towers, **no ground** | **4/10** | No ground plane; baked-emissive night look fights a lit render |
| 6 | **Vintage Shanghai Crosswalk Signals** — Signalrenders | uid `385f9fe39f9a4ee7a39a63186c60472a` | **CC BY 4.0** | GLB 2.40 MB · 50 682 tris · 0 tex (vertex colour) | Street prop (双黄 signals) | prop | Tiny; use 2–3 instances max |
| 7 | **Double-aspect Neon Chinese Pedestrian Signals** — Signalrenders | uid `fe7a874e3bbc49b38ee50df266c9438f` | **CC BY 4.0** | GLB 0.71 MB · 13 170 tris · 0 tex | Street prop | prop | — |
| 8 | **Shanghai Gardens** — jakob_van_damme | uid `323fa75b2d8444d494867212d8b2b7bb` | **CC BY 4.0** | GLB 19.02 MB · 48 668 tris (GLB) · 12 tex @ 2048 | Hand-painted island/pagoda scene, not a street | **3/10** | Stylised; no street level |

Ranking is for "put two 1.75 m humans on a recognisable Bund street and light it", not raw asset quality.

## Excluded / flagged

| Asset | Why excluded |
|---|---|
| **East gate of Broadway Mansions** (上海大厦), Tigershill — `6229b42dd6334965b15fb10128c86a6e` | **CC BY-NC-ND 4.0** → non-commercial and **no derivatives**; also 357 022 tris, GLB 132 MB, 3 × 8192 px textures. On the Bund, but unusable. |
| **Shanghai Oriental Pearl Tower**, Gabriel Quintana — `2e78d2610131461e99443fa230f38394` | Paid **Sketchfab Store** ("Buy Royalty Free"), 22.7 k tris, plus **NoAI** tag. Not free. |
| **Oriental Pearl TV Tower**, Zhang Shangbin — `47329f529b654b20b63356328223e18a` | "Buy Royalty Free" → paid Store. |
| **Shanghai Tower**, Zhang Shangbin — `4990a0d0ff0e448db209d31a204ce6fb` | View-only, no download button. |
| **Shanghai Yangpu Riverside Area**, Tigershill — `187ae0d4e5d44cf2982d69cfda405491` | 987 k tris, **not downloadable** (view only), NoAI. |
| **shanghai bund** cityscape, 3DExport id 300848 (also 452928) | Paid marketplace, 3DExport Basic/Extended license — not CC, no direct link without purchase. |
| **CGTrader "shanghai City"**, 3D Warehouse Bund models, 知末/3d66/爱给 | Paid or account-walled; 3D Warehouse uses the Trimble 3D Warehouse license (free to use, **not** a CC license) and requires a Trimble login. |
| **Google Photorealistic 3D Tiles / Cesium ion** | Technically the best-looking Bund on earth, but Google's terms forbid offline caching and redistribution; also far over budget for 8 GB VRAM. |
| **Sketchfab login** | Downloading any Sketchfab model (even CC0/CC-BY) requires a **free** account — e-mail signup, no payment. GLB download is a separate entry in the download dialog. |

## Recommendation — try these first

### Pick 1 (do this today): Poly Haven CC0 "Shanghai Bund" HDRI
It is the only asset that is simultaneously free, CC0, *of the actual Bund*, and usable at
your resolution. It also solves lighting: a 16 EV night environment map gives believable
neon rim light on gsplat humans without any manual light rig.

Exact steps:
1. Open https://polyhaven.com/a/shanghai_bund (author Greg Zaal, CC0 — no account, no login).
2. Size **4K**, format **EXR** → direct file
   `https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/4k/shanghai_bund_4k.exr` (21.71 MB).
   Preview JPG (handy for framing):
   `https://dl.polyhaven.org/file/ph-assets/HDRIs/extra/Tonemapped%20JPG/shanghai_bund.jpg` (4.76 MB).
3. Convert equirect → cubemap at 1024–2048 px/face (enough for 1260×756) or keep as
   equirect and sample with a direction vector in your nvdiffrast shader.
4. Use as (a) skybox/background shader, (b) IBL / split-sum or importance-sampled diffuse
   for the humans, (c) low-strength fill.
5. Grab CC0 PBR maps for the promenade from https://polyhaven.com/textures (all CC0):
   concrete/asphalt/granite, 2K–4K is plenty.

### Pick 2: geometry — OSM Bund block (or the CC-BY massing as a shortcut)
* Fast/honest shortcut: **City- Shanghai-Sandboxie** (CC BY 4.0). Free Sketchfab account →
  搜索 exact title → *Download 3D Model* → **GLB** (47.96 MB, textures embedded — there are
  none, so it is a white massing). Credit "Michael Zhang, CC BY 4.0, Sketchfab" in your
  report/video description. Renders trivially on 8 GB. Expect to spend the real time on
  material assignment.
* Better recognisability: Blender → **Blender-GIS** add-on → *Import OSM* over
  `51.4900, 31.2400` (外滩 / 中山东一路), extrude by `height`, export glTF. Assign granite
  and window-grid materials; keep 和平饭店's green pyramid roof and the 海关大楼 clock tower
  as separate meshes so the silhouette reads. Attribute "© OpenStreetMap contributors (ODbL)".

## Camera / framing suggestion (1260×756, 5:3)

- 40–50 mm equivalent (hfov ≈ 40–48°) puts a 1.75 m figure at ~1.7 m vertical coverage from
  4 m — i.e. a full-body two-shot at 4.5–5.5 m.
- Camera height **1.55–1.65 m** (eye level), humans standing on y = 0 on the promenade deck,
  railing just behind them, Huangpu and Lujiazui beyond.
- Put the two humans 1.0–1.5 m apart, 3/4 to camera, one slightly forward for overlap.
- If you want towers in frame, go 28 mm (vfov ≈ 44°) and keep the horizon at ~55 % frame
  height — Shanghai Tower across the river subtends a very large angle, so an HDRI backdrop
  must be rotated/pitched so its horizon matches your ground plane exactly, or the humans
  will look like they float.

## Risks to plan around

- **HDRI-only means baked lighting.** You cannot move the sun or change the mood; the skyline
  is a painting that will not parallax when the camera moves. Fine for a locked-off 1260×756
  shot, wrong for a dolly.
- **Horizon alignment** between HDRI and your ground plane is the #1 way this looks fake.
- **The Puxi side (万国建筑博览群) may be behind the capture point** — pan the EXR first; if the
  historic facades are missing/dim you must model them.
- **Scale**: the HDRI carries no metric scale; your ground plane and human heights define it.
- **Untextured massing** (candidate 3) will read as a clay model unless you author materials.
- **ODbL share-alike** applies to a derived *database*, not to a rendered video; attribution
  is still required.
- Watch VRAM: none of the above is close to 8 GB. The only heavy item seen was Broadway
  Mansions (132 MB GLB, 8K maps) — excluded anyway.

## Unverified leads (check license badge before use)

- Printables / MiniWorld3D "Skyscrapers of Shanghai, China" free STL
  (https://www.printables.com/model/428501-skyscrapers-of-shanghai-china) — Printables blocks
  automated fetches (403), so license unconfirmed; if it is CC-BY it would be a neat
  printable-grade Lujiazui skyline (no textures, STL).
- Wikimedia Commons 360° panoramas of the Bund — usually CC BY-SA 4.0 (share-alike), usable
  as a billboard but the SA obligation is awkward for a project deliverable.
- 3DSEARCH listing "Shanghai City Map 上海 The Bund & Huangpu River – Free 3D Model"
  (https://3dsearch.net/model/shanghai-city-map-shang-hai-the-bund-huangpu-river-mw3180946)
  — Cloudflare-blocked; likely a Sketchfab mirror, title suggests a flat city map, not
  street-level geometry.
