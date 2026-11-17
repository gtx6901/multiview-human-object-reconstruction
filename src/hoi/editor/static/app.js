import * as THREE from "three";
import { GLTFLoader } from "/static/vendor/loaders/GLTFLoader.js";

const $ = (id) => document.getElementById(id);
const DEG = Math.PI / 180;

const state = {
  scene: "dust2",
  spec: null,
  meta: null,
  up: [0, 1, 0],
  frames: [],
  frameIndex: 0,
  mode: "live",
  drag: null,
  busy: false,
  pending: null,
  previewTimer: null,
  poll: null,
  model: null,
  proxies: null,
  modelKey: null,
};

const vec = {
  add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  mul: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
  len: (a) => Math.hypot(a[0], a[1], a[2]),
  norm: (a) => {
    const l = Math.hypot(a[0], a[1], a[2]) || 1;
    return [a[0] / l, a[1] / l, a[2] / l];
  },
};

function rotateAxis(v, axis, angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const k = vec.norm(axis);
  const cross = vec.cross(k, v);
  const dot = vec.dot(k, v);
  return [
    v[0] * c + cross[0] * s + k[0] * dot * (1 - c),
    v[1] * c + cross[1] * s + k[1] * dot * (1 - c),
    v[2] * c + cross[2] * s + k[2] * dot * (1 - c),
  ];
}

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function setStatus(text) {
  $("status").textContent = text;
}

function number(id, fallback) {
  const value = parseFloat($(id).value);
  return Number.isFinite(value) ? value : fallback;
}

/* ---------- renderer ---------- */

const renderer = new THREE.WebGLRenderer({ canvas: $("live"), antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
const scene3 = new THREE.Scene();
scene3.background = new THREE.Color(0x101318);
const camera = new THREE.PerspectiveCamera(46, 4 / 3, 0.02, 4000);
const ambient = new THREE.HemisphereLight(0xbcc7d6, 0x38414d, 0.9);
const sun = new THREE.DirectionalLight(0xffe2c4, 1.1);
scene3.add(ambient, sun);
const helper = new THREE.AxesHelper(1);
scene3.add(helper);

function resize() {
  const wrap = $("viewport");
  const width = wrap.clientWidth;
  const height = Math.max(1, Math.round(width * 3 / 4));
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

/* ---------- scene loading ---------- */

function disposeObject(object) {
  object.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      for (const material of materials) {
        for (const key of Object.keys(material)) {
          const value = material[key];
          if (value && value.isTexture) value.dispose();
        }
        material.dispose();
      }
    }
  });
}

function loadModel(spec) {
  const key = JSON.stringify([spec.model, spec.model_scale ?? 1, spec.model_offset ?? [0, 0, 0]]);
  if (state.modelKey === key) return Promise.resolve();
  if (state.model) {
    scene3.remove(state.model);
    disposeObject(state.model);
    state.model = null;
  }
  state.modelKey = key;
  setStatus("加载场景模型…");
  return new Promise((resolve, reject) => {
    new GLTFLoader().load(
      `/asset?path=${encodeURIComponent(spec.model)}`,
      (gltf) => {
        const model = gltf.scene;
        model.scale.setScalar(spec.model_scale ?? 1);
        model.position.set(...(spec.model_offset ?? [0, 0, 0]));
        scene3.add(model);
        state.model = model;
        setStatus("场景已加载");
        resolve();
      },
      undefined,
      (error) => {
        setStatus(`模型加载失败：${error.message || error}`);
        reject(error);
      },
    );
  });
}

function viewUp() { return vec.mul(vec.norm(state.up), -1); }

function buildProxies() {
  if (state.proxies) {
    scene3.remove(state.proxies);
    disposeObject(state.proxies);
  }
  const group = new THREE.Group();
  const up = new THREE.Vector3(...viewUp()).normalize();
  const quaternion = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), up);
  const height = state.spec.placement.meters_per_body_height ?? 1.75;
  const geometry = new THREE.CapsuleGeometry(0.16, Math.max(0.2, height - 0.5), 6, 14);
  const material = new THREE.MeshStandardMaterial({
    color: 0x7fb2ff, roughness: 0.55, metalness: 0, transparent: true, opacity: 0.72,
  });
  for (const side of [-1, 1]) {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.quaternion.copy(quaternion);
    mesh.position.copy(up.clone().multiplyScalar(height * 0.5)).add(
      new THREE.Vector3(0.35 * side, 0, 0),
    );
    group.add(mesh);
  }
  group.position.set(...state.spec.placement.position);
  scene3.add(group);
  state.proxies = group;
}

/* ---------- camera ---------- */

function placement() {
  return state.spec.placement;
}

function worldEyeTarget() {
  const placement = state.spec.placement;
  const cameraSpec = state.spec.camera;
  const position = new THREE.Vector3(...placement.position);
  const eye = new THREE.Vector3(...cameraSpec.eye).add(position);
  const target = new THREE.Vector3(...cameraSpec.target).add(position);
  return { eye, target };
}

function applyCamera() {
  const { eye, target } = worldEyeTarget();
  const forward = eye.clone().sub(target).normalize().multiplyScalar(-1);
  const right = forward.clone().cross(new THREE.Vector3(0, 1, 0));
  if (right.lengthSq() < 1e-8) right.set(1, 0, 0);
  right.normalize();
  const down = forward.clone().cross(right).normalize();
  camera.up.copy(down.clone().multiplyScalar(-1));
  camera.position.copy(eye);
  camera.lookAt(target);
  camera.fov = state.spec.camera.vertical_fov ?? 46;
  camera.updateProjectionMatrix();
}

function render() {
  renderer.render(scene3, camera);
}

/* ---------- spec <-> inputs ---------- */

function loadScenes() {
  return api("/api/scenes").then((data) => {
    const select = $("scene");
    select.innerHTML = "";
    for (const scene of data.scenes) {
      const option = document.createElement("option");
      option.value = scene.name;
      option.textContent = scene.exists ? scene.name : `${scene.name}（模型缺失）`;
      select.appendChild(option);
    }
    const names = data.scenes.map((item) => item.name);
    const requested = new URLSearchParams(location.search).get("scene");
    state.scene = requested && names.includes(requested) ? requested
      : (names.includes("dust2") ? "dust2" : names[0]);
    select.value = state.scene;
  });
}

async function loadScene(name) {
  state.scene = name;
  setStatus("加载场景…");
  const data = await api(`/api/scene?name=${encodeURIComponent(name)}`);
  state.spec = data.spec;
  state.up = data.up || [0, 1, 0];
  state.frameIndex = 0;
  applySpecToInputs();
  await loadModel(state.spec);
  buildProxies();
  applyCamera();
  render();
  schedulePreview(50);
}

function applySpecToInputs() {
  const spec = state.spec;
  $("px").value = spec.placement.position[0].toFixed(2);
  $("pz").value = spec.placement.position[2].toFixed(2);
  $("yaw").value = spec.placement.yaw_degrees ?? 0;
  $("scale").value = spec.placement.meters_per_body_height ?? 1.75;
  $("snap").checked = spec.placement.snap_to_floor !== false;
  const motion = spec.camera.motion || { type: "static" };
  $("motion").value = motion.type || "static";
  $("radius").value = motion.radius ?? 6;
  $("height").value = motion.height ?? 0;
  $("turns").value = motion.turns ?? 1;
  $("rise").value = motion.rise ?? 1.2;
  $("fov").value = spec.camera.vertical_fov ?? 46;
  writeCameraInputs();
  const light = spec.lighting;
  $("intensity").value = light.intensity ?? 1;
  const ambientValue = light.ambient || [0.4, 0.4, 0.4];
  $("ambient").value = (ambientValue[0] + ambientValue[1] + ambientValue[2]) / 3;
  $("color").value = rgbHex(light.color || [1, 1, 1]);
  updateMotionFields();
}

function writeCameraInputs() {
  const cameraSpec = state.spec.camera;
  $("ex").value = cameraSpec.eye[0].toFixed(2);
  $("ey").value = cameraSpec.eye[1].toFixed(2);
  $("ez").value = cameraSpec.eye[2].toFixed(2);
  $("tx").value = cameraSpec.target[0].toFixed(2);
  $("ty").value = cameraSpec.target[1].toFixed(2);
  $("tz").value = cameraSpec.target[2].toFixed(2);
}

function readCameraInputs() {
  const cameraSpec = state.spec.camera;
  cameraSpec.eye = [number("ex", 0), number("ey", 0), number("ez", 0)];
  cameraSpec.target = [number("tx", 0), number("ty", 0), number("tz", 0)];
}

function rgbHex(color) {
  const channel = (value) => Math.round(Math.max(0, Math.min(1, value)) * 255).toString(16).padStart(2, "0");
  return `#${channel(color[0])}${channel(color[1])}${channel(color[2])}`;
}

function hexRgb(hex) {
  return [1, 3, 5].map((index) => parseInt(hex.slice(index, index + 2), 16) / 255);
}

function collectSpec() {
  const spec = state.spec;
  spec.placement.position = [number("px", spec.placement.position[0]), spec.placement.position[1], number("pz", spec.placement.position[2])];
  spec.placement.yaw_degrees = number("yaw", 0);
  spec.placement.meters_per_body_height = number("scale", 1.75);
  spec.placement.snap_to_floor = $("snap").checked;
  const type = $("motion").value;
  if (type === "static") {
    delete spec.camera.motion;
    readCameraInputs();
  } else {
    spec.camera.motion = {
      type,
      radius: number("radius", 6),
      height: number("height", 0),
      turns: number("turns", 1),
      rise: number("rise", 1.2),
      look_height: spec.camera.target[1],
      ease: true,
    };
  }
  spec.camera.vertical_fov = number("fov", 46);
  const light = spec.lighting;
  light.intensity = number("intensity", 1);
  const ambientValue = number("ambient", 0.4);
  light.ambient = [ambientValue, ambientValue, ambientValue];
  light.color = hexRgb($("color").value);
  return spec;
}

/* ---------- interaction ---------- */

function basis() {
  const cameraSpec = state.spec.camera;
  const up = vec.norm(viewUp());
  const forward = vec.norm(vec.sub(cameraSpec.target, cameraSpec.eye));
  let right = vec.cross(forward, up);
  if (vec.len(right) < 1e-4) right = [1, 0, 0];
  right = vec.norm(right);
  const camUp = vec.norm(vec.cross(right, forward));
  return { up, forward, right, camUp };
}

function orbit(dx, dy) {
  const cameraSpec = state.spec.camera;
  const up = vec.norm(viewUp());
  const { right } = basis();
  const speed = 0.0022 * number("sensitivity", 1);
  let offset = vec.sub(cameraSpec.eye, cameraSpec.target);
  offset = rotateAxis(offset, up, -dx * speed);
  const next = rotateAxis(offset, right, -dy * speed);
  if (Math.abs(vec.dot(vec.norm(next), up)) < 0.985) offset = next;
  cameraSpec.eye = vec.add(cameraSpec.target, offset);
  writeCameraInputs();
}

function pan(dx, dy) {
  const cameraSpec = state.spec.camera;
  const { right, camUp } = basis();
  const distance = vec.len(vec.sub(cameraSpec.eye, cameraSpec.target));
  const height = $("viewport").clientHeight || 1;
  const scale = (2 * distance * Math.tan(number("fov", 46) * DEG / 2)) / height;
  const move = vec.add(vec.mul(right, -dx * scale), vec.mul(camUp, dy * scale));
  cameraSpec.eye = vec.add(cameraSpec.eye, move);
  cameraSpec.target = vec.add(cameraSpec.target, move);
  writeCameraInputs();
}

function dolly(factor) {
  const cameraSpec = state.spec.camera;
  let offset = vec.sub(cameraSpec.eye, cameraSpec.target);
  const distance = vec.len(offset);
  offset = vec.mul(vec.norm(offset), Math.min(60, Math.max(0.6, distance * factor)));
  cameraSpec.eye = vec.add(cameraSpec.target, offset);
  writeCameraInputs();
}

function moveCharacter(event) {
  const rect = $("live").getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1,
  );
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(ndc, camera);
  const position = new THREE.Vector3(...state.spec.placement.position);
  const up = new THREE.Vector3(...viewUp()).normalize();
  const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(up, position);
  const hit = new THREE.Vector3();
  if (!raycaster.ray.intersectPlane(plane, hit)) return;
  const previous = [...state.spec.placement.position];
  state.spec.placement.position = [hit.x, state.spec.placement.position[1], hit.z];
  const delta = vec.sub(state.spec.placement.position, previous);
  state.spec.camera.eye = vec.sub(state.spec.camera.eye, delta);
  state.spec.camera.target = vec.sub(state.spec.camera.target, delta);
  writeCameraInputs();
  $("px").value = hit.x.toFixed(2);
  $("pz").value = hit.z.toFixed(2);
  buildProxies();
  applyCamera();
  schedulePreview(220);
}

$("live").addEventListener("contextmenu", (event) => event.preventDefault());
$("live").addEventListener("pointerdown", (event) => {
  const mode = event.button === 2 || event.shiftKey ? "pan" : (event.ctrlKey ? "move" : "orbit");
  if (mode !== "orbit" && mode !== "pan" && mode !== "move") return;
  state.drag = { mode, x: event.clientX, y: event.clientY };
  $("live").setPointerCapture(event.pointerId);
});
$("live").addEventListener("pointermove", (event) => {
  if (!state.drag) return;
  const dx = event.clientX - state.drag.x;
  const dy = event.clientY - state.drag.y;
  state.drag.x = event.clientX;
  state.drag.y = event.clientY;
  if (state.drag.mode === "move") {
    moveCharacter(event);
  } else if (state.drag.mode === "pan") {
    pan(dx, dy);
    applyCamera();
  } else {
    orbit(dx, dy);
    applyCamera();
  }
  render();
});
$("live").addEventListener("pointerup", (event) => {
  if (!state.drag) return;
  state.drag = null;
  $("live").releasePointerCapture(event.pointerId);
  schedulePreview(120);
});
$("live").addEventListener("wheel", (event) => {
  event.preventDefault();
  dolly(Math.exp(event.deltaY * 0.0012));
  applyCamera();
  render();
  schedulePreview(260);
}, { passive: false });

function preset(eye, target) {
  state.spec.camera.eye = eye;
  state.spec.camera.target = target;
  writeCameraInputs();
  applyCamera();
  render();
  schedulePreview(120);
}

$("view-front").addEventListener("click", () => preset([0, 1.6, 4.5], [0, 1.0, 0]));
$("view-side").addEventListener("click", () => preset([4.5, 1.6, 0], [0, 1.0, 0]));
$("view-back").addEventListener("click", () => preset([0, 1.6, -4.5], [0, 1.0, 0]));
$("view-top").addEventListener("click", () => preset([0.01, 6, 0], [0, 0.8, 0]));
$("view-focus").addEventListener("click", () => {
  const basisNow = basis();
  const offset = vec.mul(vec.norm(vec.sub(state.spec.camera.eye, state.spec.camera.target)), 4.5);
  preset(vec.add([0, 1.0, 0], offset), [0, 1.0, 0]);
});

/* ---------- preview render ---------- */

function schedulePreview(delay = 400) {
  if (state.previewTimer) clearTimeout(state.previewTimer);
  state.previewTimer = setTimeout(() => preview(), delay);
}

async function preview(size = "final") {
  if (state.busy) {
    state.pending = size;
    return;
  }
  state.busy = true;
  if (size === "final") {
    $("spinner").classList.remove("hidden");
    setStatus("渲染预览…");
  }
  try {
    const spec = collectSpec();
    const result = await api("/api/preview", {
      scene: state.scene, spec, frame: currentFrame(), width: 504, height: 378,
    });
    state.meta = result;
    state.frames = result.frames;
    state.frameIndex = Math.max(0, state.frames.indexOf(result.frame));
    $("frame").max = state.frames.length - 1;
    $("frame").value = state.frameIndex;
    $("frame-label").textContent = `${result.frame} / ${state.frames[state.frames.length - 1]}`;
    $("render").src = result.image;
    if (size === "final") setStatus("就绪");
  } catch (error) {
    setStatus(`渲染失败：${error.message}`);
  } finally {
    state.busy = false;
    $("spinner").classList.add("hidden");
    if (state.pending) {
      const next = state.pending;
      state.pending = null;
      preview(next);
    }
  }
}

function currentFrame() {
  return state.frames[state.frameIndex] ?? 0;
}

function updateMotionFields() {
  const type = $("motion").value;
  for (const label of document.querySelectorAll("#motion-params label")) {
    label.classList.toggle("hidden", !label.dataset.kind.split(" ").includes(type));
  }
  const locked = type !== "static";
  for (const id of ["ex", "ey", "ez", "tx", "ty", "tz"]) $(id).disabled = locked;
}

/* ---------- ui ---------- */

function setMode(mode) {
  state.mode = mode;
  $("mode-live").classList.toggle("active", mode === "live");
  $("mode-render").classList.toggle("active", mode === "render");
  $("live").classList.toggle("hidden", mode !== "live");
  $("render").classList.toggle("hidden", mode !== "render");
  if (mode === "render" && !$("render").src) preview();
}

$("mode-live").addEventListener("click", () => setMode("live"));
$("mode-render").addEventListener("click", () => setMode("render"));
$("frame").addEventListener("input", () => {
  state.frameIndex = parseInt($("frame").value, 10);
  $("frame-label").textContent = currentFrame();
  schedulePreview(350);
});
for (const id of ["px", "pz", "yaw", "scale", "fov", "ex", "ey", "ez", "tx", "ty", "tz", "radius", "height", "turns", "rise", "intensity", "ambient", "color", "motion", "snap"]) {
  $(id).addEventListener("change", () => {
    const previous = [...state.spec.placement.position];
    collectSpec();
    const delta = vec.sub(state.spec.placement.position, previous);
    if (vec.len(delta) > 1e-6) {
      state.spec.camera.eye = vec.sub(state.spec.camera.eye, delta);
      state.spec.camera.target = vec.sub(state.spec.camera.target, delta);
      writeCameraInputs();
    }
    buildProxies();
    applyCamera();
    render();
    schedulePreview(320);
  });
}
$("scene").addEventListener("change", () => loadScene($("scene").value));

$("save").addEventListener("click", async () => {
  try {
    collectSpec();
    await api(`/api/scene?name=${encodeURIComponent(state.scene)}`, { spec: state.spec });
    setStatus("配置已保存");
  } catch (error) {
    setStatus(`保存失败：${error.message}`);
  }
});

$("render").addEventListener("click", async () => {
  try {
    collectSpec();
    const result = await api("/api/render", { scene: state.scene, spec: state.spec });
    $("job").classList.remove("hidden");
    $("job-video").classList.add("hidden");
    setStatus(`开始渲染 ${result.total} 帧`);
    if (state.poll) clearInterval(state.poll);
    state.poll = setInterval(pollJob, 2000);
  } catch (error) {
    setStatus(`渲染失败：${error.message}`);
  }
});

async function pollJob() {
  try {
    const job = await api("/api/job");
    if (!job || job.status === "idle") return;
    const total = job.total || 1;
    $("job-bar").style.width = `${Math.min(100, ((job.done || 0) / total) * 100).toFixed(1)}%`;
    $("job-text").textContent = `${job.done || 0} / ${total}`;
    if (job.status === "done") {
      clearInterval(state.poll);
      state.poll = null;
      $("job-text").textContent = "渲染完成";
      $("job-video").href = "/video.mp4";
      $("job-video").classList.remove("hidden");
      setStatus("渲染完成");
    } else if (job.status === "failed") {
      clearInterval(state.poll);
      state.poll = null;
      setStatus("渲染失败");
    }
  } catch (error) {
    clearInterval(state.poll);
    state.poll = null;
    setStatus(`进度查询失败：${error.message}`);
  }
}

window.addEventListener("resize", () => {
  resize();
  applyCamera();
  render();
});

(async () => {
  try {
    resize();
    await loadScenes();
    $("scene").value = state.scene;
    await loadScene(state.scene);
    render();
  } catch (error) {
    setStatus(`启动失败：${error.message}`);
  }
})();
