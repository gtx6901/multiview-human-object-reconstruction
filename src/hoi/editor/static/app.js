const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");

const state = {
  scene: "dust2",
  spec: null,
  meta: null,
  frames: [],
  frameIndex: 0,
  drag: null,
  marker: null,
  busy: false,
  pending: null,
  poll: null,
};

const DEG = Math.PI / 180;
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

async function loadScenes() {
  const data = await api("/api/scenes");
  const select = $("scene");
  select.innerHTML = "";
  for (const scene of data.scenes) {
    const option = document.createElement("option");
    option.value = scene.name;
    option.textContent = scene.exists ? scene.name : `${scene.name}（模型缺失）`;
    select.appendChild(option);
  }
  const names = data.scenes.map((item) => item.name);
  state.scene = names.includes("dust2") ? "dust2" : names[0];
  select.value = state.scene;
}

async function loadScene(name) {
  state.scene = name;
  setStatus("加载场景…");
  const data = await api(`/api/scene?name=${encodeURIComponent(name)}`);
  state.spec = data.spec;
  state.frameIndex = 0;
  applySpecToInputs();
  await preview("final");
}

function applySpecToInputs() {
  const spec = state.spec;
  const placement = spec.placement;
  $("px").value = placement.position[0].toFixed(2);
  $("pz").value = placement.position[2].toFixed(2);
  $("yaw").value = placement.yaw_degrees ?? 0;
  $("scale").value = placement.meters_per_body_height ?? 1.75;
  $("snap").checked = placement.snap_to_floor !== false;

  const camera = spec.camera;
  const motion = camera.motion || { type: "static" };
  $("motion").value = motion.type || "static";
  $("radius").value = motion.radius ?? 6;
  $("height").value = motion.height ?? 0;
  $("turns").value = motion.turns ?? 1;
  $("rise").value = motion.rise ?? 1.2;
  $("fov").value = camera.vertical_fov ?? 46;
  writeCameraInputs();

  const light = spec.lighting;
  $("intensity").value = light.intensity ?? 1;
  const ambient = light.ambient || [0.4, 0.4, 0.4];
  $("ambient").value = (ambient[0] + ambient[1] + ambient[2]) / 3;
  const color = light.color || [1, 1, 1];
  $("color").value = rgbHex(color);
  updateMotionFields();
}

function writeCameraInputs() {
  const camera = state.spec.camera;
  $("ex").value = camera.eye[0].toFixed(2);
  $("ey").value = camera.eye[1].toFixed(2);
  $("ez").value = camera.eye[2].toFixed(2);
  $("tx").value = camera.target[0].toFixed(2);
  $("ty").value = camera.target[1].toFixed(2);
  $("tz").value = camera.target[2].toFixed(2);
}

function readCameraInputs() {
  const camera = state.spec.camera;
  camera.eye = [number("ex", 0), number("ey", 0), number("ez", 0)];
  camera.target = [number("tx", 0), number("ty", 0), number("tz", 0)];
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
  const placement = spec.placement;
  placement.position = [number("px", placement.position[0]), placement.position[1], number("pz", placement.position[2])];
  placement.yaw_degrees = number("yaw", 0);
  placement.meters_per_body_height = number("scale", 1.75);
  placement.snap_to_floor = $("snap").checked;
  const camera = spec.camera;
  const type = $("motion").value;
  if (type === "static") {
    delete camera.motion;
    readCameraInputs();
  } else {
    camera.motion = {
      type,
      radius: number("radius", 6),
      height: number("height", 0),
      turns: number("turns", 1),
      rise: number("rise", 1.2),
      look_height: camera.target[1],
      ease: true,
    };
  }
  camera.vertical_fov = number("fov", 46);
  const light = spec.lighting;
  light.intensity = number("intensity", 1);
  const ambient = number("ambient", 0.4);
  light.ambient = [ambient, ambient, ambient];
  light.color = hexRgb($("color").value);
  return spec;
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

async function preview(size = "final") {
  if (size !== "final" && state.busy) {
    state.pending = size;
    return;
  }
  if (size === "final" && state.busy) {
    state.pending = "final";
    return;
  }
  state.busy = true;
  if (size === "final") {
    $("spinner").classList.remove("hidden");
    setStatus("渲染预览…");
  }
  try {
    const spec = collectSpec();
    const width = size === "final" ? 504 : 384;
    const height = size === "final" ? 378 : 288;
    const result = await api("/api/preview", {
      scene: state.scene,
      spec,
      frame: currentFrame(),
      width,
      height,
    });
    state.meta = result;
    state.frames = result.frames;
    state.frameIndex = Math.max(0, state.frames.indexOf(result.frame));
    const slider = $("frame");
    slider.max = state.frames.length - 1;
    slider.value = state.frameIndex;
    $("frame-label").textContent = `${result.frame} / ${state.frames[state.frames.length - 1]}`;
    await draw(result.image);
    if (size === "final") setStatus("就绪");
  } catch (error) {
    setStatus(`失败：${error.message}`);
  } finally {
    state.busy = false;
    if (size === "final") $("spinner").classList.add("hidden");
    if (state.pending) {
      const next = state.pending;
      state.pending = null;
      preview(next);
    }
  }
}

function draw(src) {
  return new Promise((resolve) => {
    const image = new Image();
    image.onload = () => {
      canvas.width = image.width;
      canvas.height = image.height;
      ctx.drawImage(image, 0, 0);
      drawMarker();
      resolve();
    };
    image.src = src;
  });
}

function project(point) {
  const meta = state.meta;
  if (!meta) return null;
  const [x, y, z] = point;
  const W = meta.W;
  const K = meta.K;
  const cx = W[0][0] * x + W[0][1] * y + W[0][2] * z + W[0][3];
  const cy = W[1][0] * x + W[1][1] * y + W[1][2] * z + W[1][3];
  const cz = W[2][0] * x + W[2][1] * y + W[2][2] * z + W[2][3];
  if (cz <= 0.01) return null;
  return [(K[0][0] * cx) / cz + K[0][2], (K[1][1] * cy) / cz + K[1][2]];
}

function drawMarker() {
  if (!state.meta) return;
  const point = project(state.meta.position);
  if (!point) return;
  state.marker = point;
  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,.9)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(point[0], point[1], 9, 0, Math.PI * 2);
  ctx.stroke();
  ctx.fillStyle = "rgba(255,77,79,.9)";
  ctx.beginPath();
  ctx.arc(point[0], point[1], 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function unproject(u, v) {
  const meta = state.meta;
  if (!meta) return null;
  const K = meta.K;
  const W = meta.W;
  const dx = (u - K[0][2]) / K[0][0];
  const dy = (v - K[1][2]) / K[1][1];
  const R = [
    [W[0][0], W[0][1], W[0][2]],
    [W[1][0], W[1][1], W[1][2]],
    [W[2][0], W[2][1], W[2][2]],
  ];
  const t = [W[0][3], W[1][3], W[2][3]];
  const direction = [
    R[0][0] * dx + R[1][0] * dy + R[2][0],
    R[0][1] * dx + R[1][1] * dy + R[2][1],
    R[0][2] * dx + R[1][2] * dy + R[2][2],
  ];
  const origin = [
    -(R[0][0] * t[0] + R[1][0] * t[1] + R[2][0] * t[2]),
    -(R[0][1] * t[0] + R[1][1] * t[1] + R[2][1] * t[2]),
    -(R[0][2] * t[0] + R[1][2] * t[1] + R[2][2] * t[2]),
  ];
  const floor = meta.position[1];
  if (Math.abs(direction[1]) < 1e-6) return null;
  const scale = (floor - origin[1]) / direction[1];
  return [origin[0] + scale * direction[0], floor, origin[2] + scale * direction[2]];
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    u: ((event.clientX - rect.left) / rect.width) * canvas.width,
    v: ((event.clientY - rect.top) / rect.height) * canvas.height,
    dx: event.movementX,
    dy: event.movementY,
  };
}

function cameraBasis() {
  const camera = state.spec.camera;
  const up = vec.norm(state.meta?.up || [0, 1, 0]);
  const forward = vec.norm(vec.sub(camera.target, camera.eye));
  let right = vec.cross(forward, up);
  if (vec.len(right) < 1e-4) right = [1, 0, 0];
  right = vec.norm(right);
  const camUp = vec.norm(vec.cross(right, forward));
  return { up, forward, right, camUp };
}

function orbit(dx, dy) {
  const camera = state.spec.camera;
  const up = vec.norm(state.meta?.up || [0, 1, 0]);
  let offset = vec.sub(camera.eye, camera.target);
  const basis = cameraBasis();
  offset = rotateAxis(offset, up, -dx * 0.006);
  const next = rotateAxis(offset, basis.right, -dy * 0.006);
  const vertical = Math.abs(vec.dot(vec.norm(next), up));
  if (vertical < 0.985) offset = next;
  camera.eye = vec.add(camera.target, offset);
  writeCameraInputs();
}

function pan(dx, dy, height) {
  const camera = state.spec.camera;
  const basis = cameraBasis();
  const distance = vec.len(vec.sub(camera.eye, camera.target));
  const scale = (2 * distance * Math.tan(number("fov", 46) * DEG / 2)) / height;
  const move = vec.add(vec.mul(basis.right, -dx * scale), vec.mul(basis.camUp, dy * scale));
  camera.eye = vec.add(camera.eye, move);
  camera.target = vec.add(camera.target, move);
  writeCameraInputs();
}

function dolly(factor) {
  const camera = state.spec.camera;
  let offset = vec.sub(camera.eye, camera.target);
  const distance = vec.len(offset);
  const next = Math.min(40, Math.max(0.8, distance * factor));
  offset = vec.mul(vec.norm(offset), next);
  camera.eye = vec.add(camera.target, offset);
  writeCameraInputs();
}

let lastDraft = 0;
function scheduleDraft() {
  const now = performance.now();
  if (now - lastDraft < 220) return;
  lastDraft = now;
  preview("draft");
}

canvas.addEventListener("contextmenu", (event) => event.preventDefault());

canvas.addEventListener("pointerdown", (event) => {
  const point = canvasPoint(event);
  const onMarker = state.marker && Math.hypot(point.u - state.marker[0], point.v - state.marker[1]) < 26;
  const mode = onMarker ? "move" : (event.button === 2 || event.shiftKey) ? "pan" : "orbit";
  if (mode !== "move" && $("motion").value !== "static") {
    setStatus("动态镜头由参数生成；切回“固定”后可手动转视角");
    return;
  }
  state.drag = { mode, lastX: event.clientX, lastY: event.clientY };
  canvas.setPointerCapture(event.pointerId);
  canvas.classList.add("dragging");
});

canvas.addEventListener("pointermove", (event) => {
  if (!state.drag) return;
  const dx = event.clientX - state.drag.lastX;
  const dy = event.clientY - state.drag.lastY;
  state.drag.lastX = event.clientX;
  state.drag.lastY = event.clientY;
  if (state.drag.mode === "move") {
    const point = canvasPoint(event);
    const world = unproject(point.u, point.v);
    if (!world) return;
    state.meta.position = world;
    $("px").value = world[0].toFixed(2);
    $("pz").value = world[2].toFixed(2);
    drawMarker();
  } else if (state.drag.mode === "pan") {
    pan(dx, dy, canvas.height);
    scheduleDraft();
  } else {
    orbit(dx, dy);
    scheduleDraft();
  }
});

canvas.addEventListener("pointerup", async (event) => {
  if (!state.drag) return;
  state.drag = null;
  canvas.classList.remove("dragging");
  canvas.releasePointerCapture(event.pointerId);
  await preview("final");
});

canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  if ($("motion").value !== "static") return;
  dolly(Math.exp(event.deltaY * 0.0012));
  scheduleDraft();
}, { passive: false });

function preset(eye, target) {
  const camera = state.spec.camera;
  camera.eye = eye;
  camera.target = target;
  writeCameraInputs();
  preview("final");
}

$("view-front").addEventListener("click", () => preset([0, 1.6, 4.5], [0, 1.0, 0]));
$("view-side").addEventListener("click", () => preset([4.5, 1.6, 0], [0, 1.0, 0]));
$("view-top").addEventListener("click", () => preset([0.01, 6, 0], [0, 0.8, 0]));
$("view-back").addEventListener("click", () => preset([0, 1.6, -4.5], [0, 1.0, 0]));
$("view-focus").addEventListener("click", () => {
  const distance = 4.5;
  const basis = cameraBasis();
  const offset = vec.mul(vec.norm(vec.sub(state.spec.camera.eye, state.spec.camera.target)), distance);
  const target = [0, 1.0, 0];
  preset(vec.add(target, offset), target);
});

$("frame").addEventListener("input", () => {
  state.frameIndex = parseInt($("frame").value, 10);
  $("frame-label").textContent = currentFrame();
});
$("frame").addEventListener("change", () => preview("final"));
$("motion").addEventListener("change", () => {
  updateMotionFields();
  preview("final");
});
for (const id of ["px", "pz", "yaw", "scale", "fov", "ex", "ey", "ez", "tx", "ty", "tz", "radius", "height", "turns", "rise"]) {
  $(id).addEventListener("change", () => preview("final"));
}
for (const id of ["intensity", "ambient", "color"]) {
  $(id).addEventListener("change", () => preview("final"));
}
$("snap").addEventListener("change", () => preview("final"));
$("preview").addEventListener("click", () => preview("final"));
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
  if (state.busy) return;
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
    if (job.status === "idle") return;
    const total = job.total || 1;
    const ratio = Math.min(1, (job.done || 0) / total);
    $("job-bar").style.width = `${(ratio * 100).toFixed(1)}%`;
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

(async () => {
  try {
    await loadScenes();
    $("scene").value = state.scene;
    await loadScene(state.scene);
  } catch (error) {
    setStatus(`启动失败：${error.message}`);
  }
})();
