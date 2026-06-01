import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open as openDialog, save as saveDialog } from "@tauri-apps/plugin-dialog";
import * as THREE from "three";
import { GLTFExporter } from "three/examples/jsm/exporters/GLTFExporter.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import "./styles.css";

type LoadResponse = {
  loaded: boolean;
  result_version: number;
};

type ModelPaths = {
  onnxDir: string;
  ggufPath: string;
  yoloPath: string;
  backboneName: string;
  lbsPath: string;
  bvhTemplatePath: string;
};

type MeshTopology = {
  vertexCount: number;
  indices: number[];
};

type VideoProgress = {
  frameIndex: number;
  totalFrames: number;
  persons: number;
  frameMs: number;
};

type VideoMeshPreview = {
  frameIndex: number;
  personIndex: number;
  trackId?: number;
  frameWidth?: number;
  frameHeight?: number;
  bbox?: [number, number, number, number] | null;
  predCamT?: [number, number, number] | null;
  kps3d?: [number, number, number][] | null;
  kps2d?: [number, number][] | null;
  meshVertices: [number, number, number][];
};

type VideoMeshFrame = {
  frameIndex: number;
  people: VideoMeshPreview[];
};

type PersonResult = {
  bbox: [number, number, number, number];
  focal_length: number;
  pred_cam_t: [number, number, number];
  global_rot: [number, number, number];
  yolo_kps: [number, number, number][] | null;
  kps_2d: [number, number][] | null;
  kps_3d: [number, number, number][] | null;
  mesh_vertices: [number, number, number][] | null;
};

type ProcessResponse = {
  width: number;
  height: number;
  persons: PersonResult[];
};

type ImageResultFile = ProcessResponse & {
  kind?: string;
  version?: number;
  sourcePath?: string;
};

type VideoResponse = {
  framesRead: number;
  framesProcessed: number;
  personsTotal: number;
  meshPreviews: number;
  sourceFps: number;
  totalMs: number;
  bvhPath: string;
  csvPath: string;
};

type CopyMotionOutputsResponse = {
  paths: string[];
};

type ImageBvhResponse = {
  people: number;
  paths: string[];
};

type HardwareSnapshot = {
  gpuName: string;
  totalVram: number;
  usedVram: number;
  freeVram: number;
  gpuUtilization: number;
  temperature: number;
  powerDraw: number;
  powerLimit: number;
  processRam: number;
  totalRam: number;
  usedRam: number;
  message: string;
};

type MeshNormalization = {
  centerX: number;
  centerZ: number;
  minY: number;
};

type MeshRenderPerson = {
  meshVertices: [number, number, number][];
  kps3d?: [number, number, number][] | null;
  imageCenterX?: number;
  imageDepth?: number;
  motionOffset?: [number, number, number] | null;
  normalization?: MeshNormalization;
};

const canvas = document.querySelector<HTMLCanvasElement>("#canvas")!;
const ctx = canvas.getContext("2d")!;
const meshViewerEl = document.querySelector<HTMLDivElement>("#mesh-viewer")!;
const meshColumnEl = document.querySelector<HTMLDivElement>(".mesh-column")!;
const mediaViewerEl = document.querySelector<HTMLDivElement>(".media-viewer")!;
const videoPreviewEl = document.querySelector<HTMLVideoElement>("#video-preview")!;
const videoOverlayCanvas = document.querySelector<HTMLCanvasElement>("#video-body-overlay")!;
const outputEl = document.querySelector<HTMLPreElement>("#output")!;
const hardwareCanvas = document.querySelector<HTMLCanvasElement>("#hardware-graph")!;
const hardwareCtx = hardwareCanvas.getContext("2d")!;
const OUTPUT_PREVIEW_LIMIT = 180_000;
const FSB_MESH_VERTEX_COUNT = 18_439;

function el<T extends HTMLElement>(id: string): T {
  return document.querySelector<T>(id)!;
}

function setText(id: string, text: string) {
  el<HTMLElement>(id).textContent = text;
}

function cssVar(name: string, fallback: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

function setOutput(text: string, kind: "json" | "text" | "csv" = "text", basePath = "") {
  lastOutputText = text;
  lastOutputKind = kind;
  if (basePath) lastOutputBasePath = basePath;
  renderOutputText();
}

function renderOutputText() {
  const collapsed = el<HTMLElement>("#summary-panel").classList.contains("collapsed");
  if (collapsed) {
    outputEl.textContent = "";
    setText("#output-state", lastOutputText ? "Collapsed" : "No output");
    return;
  }
  if (lastOutputText.length > OUTPUT_PREVIEW_LIMIT) {
    const head = lastOutputText.slice(0, 120_000);
    const tail = lastOutputText.slice(-40_000);
    outputEl.textContent =
      `${head}\n\n--- Preview truncated for UI speed. Copy and Save still use the full ${lastOutputText.length.toLocaleString()} characters. ---\n\n${tail}`;
    setText("#output-state", "Expanded preview");
    return;
  }
  outputEl.textContent = lastOutputText;
  setText("#output-state", lastOutputText ? "Expanded" : "No output");
}

function showToast(message: string, tone: "success" | "warning" = "success") {
  const toast = el<HTMLDivElement>("#toast");
  toast.textContent = message;
  toast.classList.toggle("warning", tone === "warning");
  toast.classList.add("show");
  if (toastTimer) window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    toast.classList.remove("show");
    toastTimer = null;
  }, 2600);
}

function delay(ms: number) {
  return new Promise<void>((resolve) => window.setTimeout(resolve, ms));
}

function requireEngineLoaded() {
  if (engineLoaded) return true;
  showToast("Load engine first", "warning");
  return false;
}

function setEngineLoaded(loaded: boolean) {
  engineLoaded = loaded;
  el<HTMLButtonElement>("#unload-engine").disabled = !loaded;
  setText("#engine-chip", loaded ? "Engine ready" : "Engine unloaded");
  setText("#dll-state", loaded ? "Loaded" : "Pending");
}

function value(id: string) {
  return el<HTMLInputElement>(id).value.trim();
}

function setValue(id: string, text: string) {
  el<HTMLInputElement>(id).value = text;
}

function intValue(id: string) {
  return Number.parseInt(value(id), 10) || 0;
}

function floatValue(id: string) {
  return Number.parseFloat(value(id)) || 0;
}

function boolValue(id: string) {
  return el<HTMLInputElement>(id).checked;
}

let meshTopologyPromise: Promise<MeshTopology> | null = null;
let meshScene: THREE.Scene | null = null;
let meshCamera: THREE.PerspectiveCamera | null = null;
let meshRenderer: THREE.WebGLRenderer | null = null;
let meshControls: OrbitControls | null = null;
let meshGroup: THREE.Group | null = null;
let meshGrid: THREE.GridHelper | null = null;
let videoOverlayRenderer: THREE.WebGLRenderer | null = null;
let videoOverlayScene: THREE.Scene | null = null;
let videoOverlayCamera: THREE.OrthographicCamera | null = null;
let videoOverlayGroup: THREE.Group | null = null;
let videoOverlayRenderVersion = 0;
let videoOverlayLastKey = "";
let videoDrivenSyncPending = false;
let lastMeshVertices: [number, number, number][] | null = null;
let lastMeshTopology: MeshTopology | null = null;
let lastMeshPeople: PersonResult[] = [];
let lastImageMeshPeople: MeshRenderPerson[] = [];
let lastImageMeshStatus = "Mesh idle";
let lastImageMeshVertices: [number, number, number][] | null = null;
let lastImageMeshTopology: MeshTopology | null = null;
let activeWork = 0;
let imageProgressTimer: number | null = null;
let imageStartedAt = 0;
let hardwarePollMs = Number.parseInt(localStorage.getItem("sam3dbody.hardwarePollMs") ?? "1000", 10);
let engineLoaded = false;
let lastOutputKind: "json" | "text" | "csv" = "text";
let lastOutputBasePath = "";
let lastOutputText = "";
let toastTimer: number | null = null;
let activeMode: MediaKind = "image";
let videoMeshFrames: VideoMeshFrame[] = [];
let videoMeshFrameMap = new Map<number, VideoMeshPreview[]>();
let motionCursor = 0;
let motionPlaying = false;
let motionTimer: number | null = null;
let motionLoop = true;
let motionSeekVersion = 0;
let videoRunToken = 0;
let videoAbortRequested = false;
let motionReference: MeshNormalization | null = null;
type MotionTrackReference = {
  floorBottomY: number;
  floorCenterY: number;
  maxBoxHeight: number;
  camFloorY?: number;
  camBaseZ?: number;
  meshNorm?: MeshNormalization;
};
let lastVideoFps = 30;
let videoMaxPeopleSeen = 0;
let generatedMotionPath = "";
let generatedMotionVideoPath = "";
let mediaZoom = 1;
let mediaPanX = 0;
let mediaPanY = 0;
let mediaDragging = false;
let mediaDragStartX = 0;
let mediaDragStartY = 0;
let mediaDragOriginX = 0;
let mediaDragOriginY = 0;
let mediaViewWidth = 960;
let mediaViewHeight = 540;
let lastImageResult: ProcessResponse | null = null;
let lastImagePath = "";
let lastImageElement: HTMLImageElement | null = null;
let lastImageFailed = false;
let previewLoadVersion = 0;
let currentTheme = localStorage.getItem("sam3dbody.theme") || "dark";
let currentAccent = localStorage.getItem("sam3dbody.accent") || "teal";
let currentUiScale = Number.parseInt(localStorage.getItem("sam3dbody.uiScale") ?? "100", 10);
let motionTrackReferences = new Map<number, MotionTrackReference>();
const userPathFields = [
  "#library-path",
  "#onnx-dir",
  "#gguf-path",
  "#yolo-path",
  "#backbone-name",
  "#bvh-template-path",
  "#lbs-path",
] as const;

function isJsonPath(path: string) {
  return path.split(".").pop()?.toLowerCase() === "json";
}

const mhrBodyEdges: [number, number][] = [
  [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 62], [6, 8], [8, 41],
  [5, 9], [6, 10], [9, 10], [9, 11], [11, 13], [13, 15], [13, 17],
  [10, 12], [12, 14], [14, 18], [14, 20], [5, 69], [6, 69],
];
const mhrRightHandEdges: [number, number][] = [
  [41, 24], [24, 23], [23, 22], [22, 21], [41, 28], [28, 27], [27, 26], [26, 25],
  [41, 32], [32, 31], [31, 30], [30, 29], [41, 36], [36, 35], [35, 34], [34, 33],
  [41, 40], [40, 39], [39, 38], [38, 37],
];
const mhrLeftHandEdges: [number, number][] = [
  [62, 45], [45, 44], [44, 43], [43, 42], [62, 49], [49, 48], [48, 47], [47, 46],
  [62, 53], [53, 52], [52, 51], [51, 50], [62, 57], [57, 56], [56, 55], [55, 54],
  [62, 61], [61, 60], [60, 59], [59, 58],
];
const videoOverlayFitJoints = [
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 41, 62, 69,
] as const;
const cocoEdges: [number, number][] = [
  [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
];

const hardwareHistory: HardwareSnapshot[] = [];
const hardwareHistoryLimit = 120;
const gridY = -0.7;

function setProgress(id: string, current: number, total: number) {
  const pct = total > 0 ? Math.min(100, Math.max(0, (current / total) * 100)) : 0;
  el<HTMLElement>(id).style.width = `${pct}%`;
}

function formatBytes(bytes: number) {
  if (!bytes) return "-";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit < 2 ? 0 : 1)} ${units[unit]}`;
}

function cleanPath(path: string) {
  if (path.startsWith("\\\\?\\UNC\\")) return `\\\\${path.slice(8)}`;
  if (path.startsWith("\\\\?\\")) return path.slice(4);
  return path;
}

function fileUrl(path: string) {
  const normalized = cleanPath(path).replace(/\\/g, "/");
  return `file:///${encodeURI(normalized).replace(/#/g, "%23")}`;
}

function setVideoPreviewSource(path: string) {
  const cleaned = cleanPath(path);
  if (!cleaned) return;
  videoPreviewEl.removeAttribute("src");
  clearVideoBodyOverlay();
  videoPreviewEl.onerror = () => {
    if (!videoPreviewEl.src.startsWith("file:")) {
      videoPreviewEl.src = fileUrl(cleaned);
      videoPreviewEl.load();
    }
  };
  videoPreviewEl.src = convertFileSrc(cleaned);
  videoPreviewEl.load();
}

function stripExtension(path: string) {
  const dot = path.lastIndexOf(".");
  const slash = Math.max(path.lastIndexOf("\\"), path.lastIndexOf("/"));
  return dot > slash ? path.slice(0, dot) : path;
}

function dateStamp() {
  const now = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
}

function suggestedOutputName(sourcePath: string, suffix: string, extension: string) {
  const base = fileName(stripExtension(sourcePath || "sam3dbody"));
  return `${base}_${dateStamp()}${suffix}.${extension}`;
}

function suggestedSiblingPath(sourcePath: string, suffix: string, extension: string) {
  const dir = parentDir(sourcePath);
  const name = suggestedOutputName(sourcePath, suffix, extension);
  return dir ? `${dir}\\${name}` : name;
}

function setMeter(barId: string, textId: string, current: number, total: number, text: string) {
  setProgress(barId, current, total || current || 1);
  setText(textId, text);
}

function updateHardware(snapshot: HardwareSnapshot) {
  hardwareHistory.push(snapshot);
  if (hardwareHistory.length > hardwareHistoryLimit) hardwareHistory.shift();

  setText(
    "#hardware-detail",
    `${snapshot.gpuName} | ${snapshot.temperature || "-"} C | app RAM ${formatBytes(snapshot.processRam)}`,
  );
  setMeter(
    "#hw-vram-bar",
    "#hw-vram-text",
    snapshot.usedVram,
    snapshot.totalVram,
    `${formatBytes(snapshot.usedVram)} / ${formatBytes(snapshot.totalVram)}`,
  );
  setMeter(
    "#hw-gpu-bar",
    "#hw-gpu-text",
    snapshot.gpuUtilization,
    100,
    `${snapshot.gpuUtilization || 0}%`,
  );
  setMeter(
    "#hw-power-bar",
    "#hw-power-text",
    snapshot.powerDraw,
    snapshot.powerLimit,
    snapshot.powerLimit ? `${snapshot.powerDraw.toFixed(0)} / ${snapshot.powerLimit.toFixed(0)} W` : "-",
  );
  setMeter(
    "#hw-ram-bar",
    "#hw-ram-text",
    snapshot.usedRam,
    snapshot.totalRam,
    `${formatBytes(snapshot.usedRam)} / ${formatBytes(snapshot.totalRam)}`,
  );
  drawHardwareGraph();
}

function drawHardwareGraph() {
  const rect = hardwareCanvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (hardwareCanvas.width !== Math.floor(width * dpr)) {
    hardwareCanvas.width = Math.floor(width * dpr);
    hardwareCanvas.height = Math.floor(height * dpr);
  }
  hardwareCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  hardwareCtx.clearRect(0, 0, width, height);
  hardwareCtx.fillStyle = cssVar("--bg-inner", "#0d1013");
  hardwareCtx.fillRect(0, 0, width, height);

  const labelWidth = 38;
  const plotX = labelWidth;
  const plotWidth = Math.max(1, width - labelWidth - 4);
  hardwareCtx.font = "10px Inter, sans-serif";
  hardwareCtx.fillStyle = cssVar("--text-muted", "#9ea8b3");
  hardwareCtx.textAlign = "right";
  hardwareCtx.textBaseline = "middle";
  for (const [pct, label] of [
    [1, "100%"],
    [0.5, "50%"],
    [0, "0%"],
  ] as const) {
    const y = Math.round(height - pct * height) + 0.5;
    hardwareCtx.fillText(label, labelWidth - 7, Math.max(8, Math.min(height - 8, y)));
  }

  hardwareCtx.strokeStyle = cssVar("--border", "#1f262d");
  hardwareCtx.lineWidth = 1;
  for (const pct of [0, 0.25, 0.5, 0.75, 1]) {
    const y = Math.round(height * pct) + 0.5;
    hardwareCtx.beginPath();
    hardwareCtx.moveTo(plotX, y);
    hardwareCtx.lineTo(width, y);
    hardwareCtx.stroke();
  }

  drawHardwareLine(plotX, plotWidth, height, "#e0a12b", (sample) =>
    sample.totalVram ? sample.usedVram / sample.totalVram : 0,
  );
  drawHardwareLine(plotX, plotWidth, height, "#25b8ab", (sample) => sample.gpuUtilization / 100);
  drawHardwareLine(plotX, plotWidth, height, "#c56cf0", (sample) =>
    sample.powerLimit ? sample.powerDraw / sample.powerLimit : 0,
  );
  drawHardwareLine(plotX, plotWidth, height, "#6aa6ff", (sample) =>
    sample.totalRam ? sample.usedRam / sample.totalRam : 0,
  );
}

function drawHardwareLine(
  start: number,
  width: number,
  height: number,
  color: string,
  valueFor: (sample: HardwareSnapshot) => number,
) {
  if (hardwareHistory.length < 2) return;
  const step = width / Math.max(1, hardwareHistoryLimit - 1);
  const startX = start + width - step * (hardwareHistory.length - 1);
  hardwareCtx.beginPath();
  hardwareCtx.strokeStyle = color;
  hardwareCtx.lineWidth = 1.8;
  hardwareHistory.forEach((sample, index) => {
    const value = Math.min(1, Math.max(0, valueFor(sample)));
    const x = startX + step * index;
    const y = height - value * height;
    if (index === 0) hardwareCtx.moveTo(x, y);
    else hardwareCtx.lineTo(x, y);
  });
  hardwareCtx.stroke();
}

async function pollHardware() {
  try {
    updateHardware(await invoke<HardwareSnapshot>("hardware_snapshot"));
  } catch (error) {
    setText("#hardware-detail", String(error));
  } finally {
    window.setTimeout(pollHardware, hardwarePollMs);
  }
}

function initHardwarePollRate() {
  const select = el<HTMLSelectElement>("#hardware-poll-rate");
  if (![250, 500, 1000, 1500].includes(hardwarePollMs)) {
    hardwarePollMs = 1000;
  }
  select.value = String(hardwarePollMs);
  select.addEventListener("change", () => {
    hardwarePollMs = Number.parseInt(select.value, 10) || 1000;
    localStorage.setItem("sam3dbody.hardwarePollMs", String(hardwarePollMs));
  });
}

function initCollapsibles() {
  const hardwarePanel = el<HTMLElement>("#hardware-panel");
  const hardwareToggle = el<HTMLButtonElement>("#hardware-toggle");
  hardwareToggle.addEventListener("click", () => {
    const collapsed = hardwarePanel.classList.toggle("collapsed");
    hardwareToggle.textContent = collapsed ? "+" : "-";
    hardwareToggle.setAttribute("aria-label", collapsed ? "Expand hardware" : "Collapse hardware");
    drawHardwareGraph();
  });

  const summaryPanel = el<HTMLElement>("#summary-panel");
  const outputToggle = el<HTMLButtonElement>("#toggle-output");
  outputToggle.addEventListener("click", () => {
    const collapsed = summaryPanel.classList.toggle("collapsed");
    outputToggle.textContent = collapsed ? "Expand" : "Collapse";
    outputToggle.classList.toggle("arrow-down", collapsed);
    outputToggle.classList.toggle("arrow-up", !collapsed);
    renderOutputText();
  });
}

function beginImageProgress() {
  activeWork += 1;
  imageStartedAt = performance.now();
  el("#image-progress").classList.add("indeterminate");
  setText("#image-progress-text", "Starting inference");
  if (imageProgressTimer) window.clearInterval(imageProgressTimer);
  imageProgressTimer = window.setInterval(() => {
    const elapsed = ((performance.now() - imageStartedAt) / 1000).toFixed(1);
    setText("#image-progress-text", `Running CUDA inference | ${elapsed}s elapsed`);
  }, 250);
}

function finishImageProgress(ok: boolean) {
  activeWork = Math.max(0, activeWork - 1);
  if (imageProgressTimer) {
    window.clearInterval(imageProgressTimer);
    imageProgressTimer = null;
  }
  el("#image-progress").classList.remove("indeterminate");
  setProgress("#image-progress-bar", ok ? 1 : 0, 1);
}

function storageKeyForPathField(id: string) {
  return `sam3dbody.path.${id.replace(/^#/, "")}`;
}

function restoreUserPaths() {
  for (const id of userPathFields) {
    const saved = localStorage.getItem(storageKeyForPathField(id));
    if (saved) setValue(id, saved);
  }
}

function saveUserPath(id: string) {
  const text = value(id);
  const key = storageKeyForPathField(id);
  if (text) localStorage.setItem(key, text);
  else localStorage.removeItem(key);
}

function saveUserPaths() {
  for (const id of userPathFields) saveUserPath(id);
}

function applyModelPaths(models: ModelPaths, persist = false) {
  if (!value("#onnx-dir")) setValue("#onnx-dir", models.onnxDir);
  if (!value("#gguf-path")) setValue("#gguf-path", models.ggufPath);
  if (!value("#yolo-path")) setValue("#yolo-path", models.yoloPath);
  if (!value("#backbone-name")) setValue("#backbone-name", models.backboneName);
  if (!value("#lbs-path")) setValue("#lbs-path", models.lbsPath);
  if (!value("#bvh-template-path")) setValue("#bvh-template-path", models.bvhTemplatePath);
  if (persist) saveUserPaths();
  setText("#model-chip", "Models ready");
  setText("#model-state", "Ready");
  setText("#model-progress-text", models.onnxDir);
  setProgress("#model-progress-bar", 1, 1);
}

function withPathStem(inputPath: string, suffix: string) {
  if (!inputPath) return "";
  const dot = inputPath.lastIndexOf(".");
  return `${dot > 0 ? inputPath.slice(0, dot) : inputPath}${suffix}`;
}

function defaultBvhPath(videoPath: string) {
  return withPathStem(videoPath, ".sam3dbody_motion.bvh");
}

function defaultObjPath(imagePath: string) {
  return withPathStem(imagePath, ".sam3dbody_mesh.obj");
}

function fileName(inputPath: string) {
  return inputPath.split(/[\\/]/).pop() ?? inputPath;
}

function parentDir(inputPath: string) {
  const idx = Math.max(inputPath.lastIndexOf("\\"), inputPath.lastIndexOf("/"));
  return idx > 0 ? inputPath.slice(0, idx) : "";
}

async function fillDefaultPaths() {
  try {
    restoreUserPaths();
    const path = await invoke<string | null>("bundled_engine_path");
    if (path) {
      setText("#dll-state", "Bundled");
    }

    const models = await invoke<ModelPaths | null>("default_model_paths");
    if (models) {
      setText("#model-chip", "Models ready");
      setText("#model-state", "Ready");
      setText("#model-progress-text", "Local model bundle found. Use Find Models or Browse to fill fields.");
      setProgress("#model-progress-bar", 1, 1);
    } else {
      setText("#model-chip", "Models missing");
      setText("#model-state", "Missing");
      setText("#model-progress-text", "Use manual links or point the paths at your local onnx folder");
    }

    setEngineLoaded(await invoke<boolean>("engine_status"));
  } catch {
    setText("#model-progress-text", "Tauri bridge unavailable");
  }
}

function setMediaMode(kind: MediaKind) {
  mediaViewerEl.classList.toggle("video-mode", kind === "video");
  updateVideoOverlayVisibility();
}

function resizeMediaCanvas() {
  const rect = mediaViewerEl.getBoundingClientRect();
  mediaViewWidth = Math.max(1, Math.floor(rect.width));
  mediaViewHeight = Math.max(1, Math.floor(rect.height));
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const targetWidth = Math.max(1, Math.floor(mediaViewWidth * dpr));
  const targetHeight = Math.max(1, Math.floor(mediaViewHeight * dpr));
  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    canvas.width = targetWidth;
    canvas.height = targetHeight;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  resizeVideoBodyOverlay();
}

function imageLayout(result: ProcessResponse) {
  const baseScale = Math.min(mediaViewWidth / result.width, mediaViewHeight / result.height);
  const scale = baseScale * mediaZoom;
  const width = result.width * scale;
  const height = result.height * scale;
  return {
    scale,
    x: (mediaViewWidth - width) / 2 + mediaPanX,
    y: (mediaViewHeight - height) / 2 + mediaPanY,
    width,
    height,
  };
}

function loadImageSource(src: string) {
  return new Promise<HTMLImageElement | null>((resolve) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => resolve(null);
    image.src = src;
  });
}

async function loadPreviewImage(imagePath: string) {
  if (!imagePath) return null;
  const cleanedPath = cleanPath(imagePath);
  const direct = await loadImageSource(convertFileSrc(cleanedPath));
  if (direct) return direct;

  try {
    const dataUrl = await invoke<string>("image_data_url", { path: cleanedPath });
    return loadImageSource(dataUrl);
  } catch {
    return null;
  }
}

function drawCurrentImagePreview() {
  resizeMediaCanvas();
  clampMediaPan();
  ctx.clearRect(0, 0, mediaViewWidth, mediaViewHeight);
  ctx.fillStyle = cssVar("--bg-inner", "#11151c");
  ctx.fillRect(0, 0, mediaViewWidth, mediaViewHeight);
  if (!lastImageResult) return;

  const layout = imageLayout(lastImageResult);
  if (lastImageElement) {
    ctx.drawImage(lastImageElement, layout.x, layout.y, layout.width, layout.height);
  }

  ctx.strokeStyle = cssVar("--border-strong", "#44546a");
  ctx.lineWidth = 1;
  ctx.strokeRect(layout.x, layout.y, layout.width, layout.height);

  if (lastImageFailed) {
    ctx.fillStyle = "#6f7b87";
    ctx.font = "14px Inter, sans-serif";
    ctx.fillText("Image preview unavailable", layout.x + 16, layout.y + 28);
  }

  const showLandmarks = boolValue("#image-landmarks-toggle");
  const showSkeleton = boolValue("#image-skeleton-toggle");
  const showBbox = boolValue("#image-bbox-toggle");
  for (const person of lastImageResult.persons) {
    const [x1, y1, x2, y2] = person.bbox;
    if (showBbox) {
      ctx.strokeStyle = "#23c7b7";
      ctx.lineWidth = 2;
      ctx.strokeRect(
        layout.x + x1 * layout.scale,
        layout.y + y1 * layout.scale,
        (x2 - x1) * layout.scale,
        (y2 - y1) * layout.scale,
      );
    }

    if (showSkeleton && person.kps_2d?.length) {
      drawKeypointEdges(person.kps_2d, mhrBodyEdges, layout, "#25b8ab", 2);
      drawKeypointEdges(person.kps_2d, mhrRightHandEdges, layout, "#6aa6ff", 1.4);
      drawKeypointEdges(person.kps_2d, mhrLeftHandEdges, layout, "#e85d75", 1.4);
    } else if (showSkeleton && person.yolo_kps?.length) {
      const yoloPoints = person.yolo_kps.map(([x, y]) => [x, y] as [number, number]);
      drawKeypointEdges(yoloPoints, cocoEdges, layout, "#25b8ab", 2, person.yolo_kps);
    }

    if (showLandmarks && person.kps_2d) {
      ctx.fillStyle = "#6aa6ff";
      for (const [x, y] of person.kps_2d) {
        if (!isValidImagePoint(x, y, lastImageResult.width, lastImageResult.height)) {
          continue;
        }
        ctx.beginPath();
        ctx.arc(layout.x + x * layout.scale, layout.y + y * layout.scale, 2.4, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    if (showLandmarks && person.yolo_kps) {
      ctx.fillStyle = "#f59e0b";
      for (const [x, y, conf] of person.yolo_kps) {
        if (conf < 0.3 || !isValidImagePoint(x, y, lastImageResult.width, lastImageResult.height)) continue;
        ctx.beginPath();
        ctx.arc(layout.x + x * layout.scale, layout.y + y * layout.scale, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
}

function isValidImagePoint(x: number, y: number, width: number, height: number) {
  return Number.isFinite(x) && Number.isFinite(y) && x >= -2 && y >= -2 && x <= width + 2 && y <= height + 2;
}

function drawKeypointEdges(
  points: [number, number][],
  edges: [number, number][],
  layout: { scale: number; x: number; y: number },
  color: string,
  lineWidth: number,
  confidence?: [number, number, number][],
) {
  if (!lastImageResult) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const [a, b] of edges) {
    const pa = points[a];
    const pb = points[b];
    if (!pa || !pb) continue;
    if (confidence && ((confidence[a]?.[2] ?? 0) < 0.3 || (confidence[b]?.[2] ?? 0) < 0.3)) continue;
    if (
      !isValidImagePoint(pa[0], pa[1], lastImageResult.width, lastImageResult.height) ||
      !isValidImagePoint(pb[0], pb[1], lastImageResult.width, lastImageResult.height)
    ) {
      continue;
    }
    ctx.beginPath();
    ctx.moveTo(layout.x + pa[0] * layout.scale, layout.y + pa[1] * layout.scale);
    ctx.lineTo(layout.x + pb[0] * layout.scale, layout.y + pb[1] * layout.scale);
    ctx.stroke();
  }
}

async function drawResults(result: ProcessResponse, imagePath = "") {
  setMediaMode("image");
  const version = ++previewLoadVersion;
  if (lastImagePath !== imagePath) resetMediaTransform(false);
  const image = await loadPreviewImage(imagePath);
  if (version !== previewLoadVersion) return;
  lastImageResult = result;
  lastImagePath = imagePath;
  lastImageElement = image;
  lastImageFailed = !lastImageElement;
  drawCurrentImagePreview();
  setText("#media-zoom-state", `${Math.round(mediaZoom * 100)}%`);
}

async function previewImageOnly(imagePath: string) {
  setMediaMode("image");
  const version = ++previewLoadVersion;
  resetMediaTransform(false);
  const image = await loadPreviewImage(imagePath);
  if (version !== previewLoadVersion) return;
  lastImagePath = imagePath;
  lastImageElement = image;
  lastImageFailed = !lastImageElement;
  lastImageResult = {
    width: lastImageElement?.naturalWidth || mediaViewWidth,
    height: lastImageElement?.naturalHeight || mediaViewHeight,
    persons: [],
  };
  drawCurrentImagePreview();
}

function getMeshTopology() {
  meshTopologyPromise ??= invoke<MeshTopology>("mesh_topology");
  return meshTopologyPromise;
}

function initMeshViewer() {
  if (meshRenderer) return;

  meshScene = new THREE.Scene();
  meshScene.background = new THREE.Color(cssVar("--bg-inner", "#0d1013"));

  meshCamera = new THREE.PerspectiveCamera(32, 1, 0.01, 100);
  meshCamera.position.set(0, 0.2, 2.4);

  meshRenderer = new THREE.WebGLRenderer({ antialias: true });
  meshRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  meshViewerEl.appendChild(meshRenderer.domElement);

  meshControls = new OrbitControls(meshCamera, meshRenderer.domElement);
  meshControls.enableDamping = true;
  meshControls.dampingFactor = 0.08;
  meshControls.autoRotate = boolValue("#auto-rotate-toggle");
  meshControls.autoRotateSpeed = 1.2;

  const hemi = new THREE.HemisphereLight(0xf2f7ff, 0x172027, 2.4);
  meshScene.add(hemi);
  const key = new THREE.DirectionalLight(0xffffff, 2.0);
  key.position.set(2.4, 3.0, 2.0);
  meshScene.add(key);

  meshGrid = createGridHelper();
  meshScene.add(meshGrid);

  meshGroup = new THREE.Group();
  meshScene.add(meshGroup);

  window.addEventListener("resize", resizeMeshViewer);
  resizeMeshViewer();
  animateMeshViewer();
}

function resizeMeshViewer() {
  if (!meshRenderer || !meshCamera) return;
  const rect = meshViewerEl.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  meshRenderer.setSize(width, height, false);
  meshCamera.aspect = width / height;
  meshCamera.updateProjectionMatrix();
}

function animateMeshViewer() {
  requestAnimationFrame(animateMeshViewer);
  if (!meshRenderer || !meshScene || !meshCamera) return;
  if (meshControls) {
    meshControls.autoRotate = boolValue("#auto-rotate-toggle");
    meshControls.update();
  }
  meshRenderer.render(meshScene, meshCamera);
}

function disposeMeshObject(object: THREE.Object3D) {
  object.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (mesh.geometry) mesh.geometry.dispose();
    const material = mesh.material;
    if (Array.isArray(material)) {
      material.forEach((entry) => entry.dispose());
    } else if (material) {
      material.dispose();
    }
  });
}

function clearMeshPreview() {
  if (!meshGroup) return;
  for (const child of [...meshGroup.children]) {
    disposeMeshObject(child);
    meshGroup.remove(child);
  }
}

function initVideoBodyOverlay() {
  if (videoOverlayRenderer) return;
  videoOverlayScene = new THREE.Scene();
  videoOverlayGroup = new THREE.Group();
  videoOverlayScene.add(videoOverlayGroup);
  videoOverlayScene.add(new THREE.AmbientLight(0xffffff, 1.25));
  const light = new THREE.DirectionalLight(0xffffff, 1.6);
  light.position.set(0, -0.5, 2);
  videoOverlayScene.add(light);
  videoOverlayCamera = new THREE.OrthographicCamera(0, 1, 0, 1, -10000, 10000);
  videoOverlayCamera.position.set(0, 0, 1000);
  videoOverlayCamera.lookAt(0, 0, 0);
  videoOverlayRenderer = new THREE.WebGLRenderer({
    canvas: videoOverlayCanvas,
    alpha: true,
    antialias: true,
  });
  videoOverlayRenderer.setClearColor(0x000000, 0);
  videoOverlayRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  resizeVideoBodyOverlay();
}

function clearVideoBodyOverlay() {
  videoOverlayRenderVersion += 1;
  videoOverlayLastKey = "";
  if (videoOverlayGroup) {
    for (const child of [...videoOverlayGroup.children]) {
      disposeMeshObject(child);
      videoOverlayGroup.remove(child);
    }
  }
  videoOverlayRenderer?.clear();
}

function videoBodyOverlayEnabled() {
  return activeMode === "video" && boolValue("#video-overlay-toggle") && videoMeshFrames.length > 0;
}

function updateVideoOverlayVisibility() {
  const enabled = videoBodyOverlayEnabled();
  mediaViewerEl.classList.toggle("overlay-enabled", enabled);
  if (!enabled) {
    clearVideoBodyOverlay();
    return;
  }
  const frame = videoMeshFrames[motionCursor];
  if (frame) void renderVideoBodyOverlay(frame);
}

function resizeVideoBodyOverlay() {
  const width = Math.max(1, mediaViewWidth);
  const height = Math.max(1, mediaViewHeight);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const targetWidth = Math.floor(width * dpr);
  const targetHeight = Math.floor(height * dpr);
  if (videoOverlayCanvas.width !== targetWidth || videoOverlayCanvas.height !== targetHeight) {
    videoOverlayCanvas.width = targetWidth;
    videoOverlayCanvas.height = targetHeight;
    videoOverlayLastKey = "";
  }
  if (!videoOverlayRenderer || !videoOverlayCamera) return;
  videoOverlayRenderer.setPixelRatio(dpr);
  videoOverlayRenderer.setSize(width, height, false);
  videoOverlayCamera.left = 0;
  videoOverlayCamera.right = width;
  videoOverlayCamera.top = 0;
  videoOverlayCamera.bottom = height;
  videoOverlayCamera.position.set(0, 0, 1000);
  videoOverlayCamera.lookAt(0, 0, 0);
  videoOverlayCamera.updateProjectionMatrix();
}

function gridSizeValue() {
  return Math.min(32, Math.max(8, intValue("#grid-size") || 8));
}

function createGridHelper() {
  const divisions = gridSizeValue();
  const size = 1.8 * (divisions / 8);
  const grid = new THREE.GridHelper(
    size,
    divisions,
    new THREE.Color(cssVar("--border-strong", "#303842")),
    new THREE.Color(cssVar("--border", "#1f262d")),
  );
  grid.position.y = gridY;
  grid.visible = boolValue("#grid-toggle");
  return grid;
}

function updateGridSize() {
  setText("#grid-size-label", String(gridSizeValue()));
  if (!meshScene) return;
  if (meshGrid) {
    meshScene.remove(meshGrid);
    disposeMeshObject(meshGrid);
  }
  meshGrid = createGridHelper();
  meshScene.add(meshGrid);
}

function fitMeshCamera() {
  if (!meshGroup || !meshCamera || !meshControls) return;
  const box = new THREE.Box3().setFromObject(meshGroup);
  const size = new THREE.Vector3();
  box.getSize(size);
  const maxDim = Math.max(size.x, size.y, size.z, 0.5);
  meshCamera.near = maxDim / 100;
  meshCamera.far = maxDim * 100;
  meshCamera.position.set(0, maxDim * 0.02, -maxDim * 1.55);
  meshCamera.lookAt(0, 0, 0);
  meshCamera.updateProjectionMatrix();
  meshControls.target.set(0, 0, 0);
  meshControls.update();
}

function transformMeshPoint([x, y, z]: [number, number, number]) {
  return new THREE.Vector3(-x, -y, z);
}

function transformedCenter(points: [number, number, number][]) {
  const box = new THREE.Box3();
  for (const point of points) {
    box.expandByPoint(transformMeshPoint(point));
  }
  const center = new THREE.Vector3();
  box.getCenter(center);
  return center;
}

function meshNormalization(points: [number, number, number][]) {
  const box = new THREE.Box3();
  for (const point of points) {
    box.expandByPoint(transformMeshPoint(point));
  }
  const center = new THREE.Vector3();
  box.getCenter(center);
  return {
    centerX: center.x,
    centerZ: center.z,
    minY: box.min.y,
  };
}

function transformedMeshBounds(points: [number, number, number][]) {
  const box = new THREE.Box3();
  for (const point of points) {
    box.expandByPoint(transformMeshPoint(point));
  }
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(size);
  return { box, center, size };
}

function videoFrameSize(frame: VideoMeshFrame) {
  const personWithSize = frame.people.find((person) => person.frameWidth && person.frameHeight);
  return {
    width: Math.max(1, personWithSize?.frameWidth || videoPreviewEl.videoWidth || mediaViewWidth),
    height: Math.max(1, personWithSize?.frameHeight || videoPreviewEl.videoHeight || mediaViewHeight),
  };
}

function videoContentRect(frame: VideoMeshFrame) {
  const source = videoFrameSize(frame);
  const scale = Math.min(mediaViewWidth / source.width, mediaViewHeight / source.height);
  const width = source.width * scale;
  const height = source.height * scale;
  return {
    x: (mediaViewWidth - width) * 0.5,
    y: (mediaViewHeight - height) * 0.5,
    width,
    height,
    sourceWidth: source.width,
    sourceHeight: source.height,
  };
}

function videoOverlayPoint(rect: ReturnType<typeof videoContentRect>, point: [number, number]) {
  return {
    x: rect.x + (point[0] / rect.sourceWidth) * rect.width,
    y: rect.y + (point[1] / rect.sourceHeight) * rect.height,
  };
}

function validVideoKps2d(person: VideoMeshPreview) {
  return (person.kps2d ?? []).filter(
    (point) =>
      point &&
      point.every(Number.isFinite) &&
      point[0] >= 0 &&
      point[1] >= 0 &&
      point[0] <= Math.max(1, person.frameWidth || 1) &&
      point[1] <= Math.max(1, person.frameHeight || 1),
  );
}

function videoPersonOverlayRect(person: VideoMeshPreview, rect: ReturnType<typeof videoContentRect>) {
  const points = validVideoKps2d(person);
  if (points.length >= 8) {
    const projected = points.map((point) => videoOverlayPoint(rect, point));
    const xs = projected.map((point) => point.x).sort((a, b) => a - b);
    const ys = projected.map((point) => point.y).sort((a, b) => a - b);
    const q = (values: number[], pct: number) => values[Math.min(values.length - 1, Math.max(0, Math.round((values.length - 1) * pct)))];
    let left = q(xs, 0.03);
    let right = q(xs, 0.97);
    let top = q(ys, 0.02);
    let bottom = q(ys, 0.98);
    const width = Math.max(8, right - left);
    const height = Math.max(8, bottom - top);
    const padX = Math.max(8, width * 0.18);
    const padTop = Math.max(6, height * 0.06);
    const padBottom = Math.max(6, height * 0.04);
    left -= padX;
    right += padX;
    top -= padTop;
    bottom += padBottom;
    return { left, top, right, bottom };
  }

  const bbox = person.bbox!;
  return {
    left: rect.x + (bbox[0] / rect.sourceWidth) * rect.width,
    top: rect.y + (bbox[1] / rect.sourceHeight) * rect.height,
    right: rect.x + (bbox[2] / rect.sourceWidth) * rect.width,
    bottom: rect.y + (bbox[3] / rect.sourceHeight) * rect.height,
  };
}

type OverlayAffineFit = {
  kind: "affine";
  matrix: [number, number, number, number, number, number];
  zScale: number;
  centerZ: number;
};

type OverlayBoxFit = {
  kind: "box";
  left: number;
  top: number;
  right: number;
  bottom: number;
};

function solve3x3(matrix: number[][], vector: number[]) {
  const a = matrix.map((row, index) => [...row, vector[index]]);
  for (let col = 0; col < 3; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < 3; row += 1) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    if (Math.abs(a[pivot][col]) < 1e-8) return null;
    if (pivot !== col) [a[pivot], a[col]] = [a[col], a[pivot]];
    const divisor = a[col][col];
    for (let item = col; item < 4; item += 1) a[col][item] /= divisor;
    for (let row = 0; row < 3; row += 1) {
      if (row === col) continue;
      const factor = a[row][col];
      for (let item = col; item < 4; item += 1) {
        a[row][item] -= factor * a[col][item];
      }
    }
  }
  return [a[0][3], a[1][3], a[2][3]] as [number, number, number];
}

function solveAffine2d(source: [number, number][], target: { x: number; y: number }[]) {
  if (source.length < 4 || source.length !== target.length) return null;
  const normal = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const rhsX = [0, 0, 0];
  const rhsY = [0, 0, 0];
  source.forEach(([x, y], index) => {
    const values = [x, y, 1];
    for (let row = 0; row < 3; row += 1) {
      rhsX[row] += values[row] * target[index].x;
      rhsY[row] += values[row] * target[index].y;
      for (let col = 0; col < 3; col += 1) {
        normal[row][col] += values[row] * values[col];
      }
    }
  });
  const xSolution = solve3x3(normal, rhsX);
  const ySolution = solve3x3(normal, rhsY);
  if (!xSolution || !ySolution) return null;
  return [
    xSolution[0],
    xSolution[1],
    xSolution[2],
    ySolution[0],
    ySolution[1],
    ySolution[2],
  ] as [number, number, number, number, number, number];
}

function applyAffine2d(
  matrix: [number, number, number, number, number, number],
  x: number,
  y: number,
) {
  return {
    x: matrix[0] * x + matrix[1] * y + matrix[2],
    y: matrix[3] * x + matrix[4] * y + matrix[5],
  };
}

function bounds2d(points: ({ x: number; y: number } | [number, number])[]) {
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  points.forEach((point) => {
    const x = Array.isArray(point) ? point[0] : point.x;
    const y = Array.isArray(point) ? point[1] : point.y;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  });
  return {
    width: Math.max(0.001, maxX - minX),
    height: Math.max(0.001, maxY - minY),
  };
}

function overlaySourcePoint(point: [number, number, number]) {
  const transformed = transformMeshPoint(point);
  return [transformed.x, -transformed.y] as [number, number];
}

function videoOverlayFrameSignature(people: VideoMeshPreview[]) {
  return people
    .map((person) => {
      const joints = videoOverlayFitJoints
        .map((index) => person.kps2d?.[index])
        .filter((point): point is [number, number] => !!point)
        .map((point) => `${Math.round(point[0])},${Math.round(point[1])}`)
        .join(";");
      const bbox = person.bbox?.map((value) => Math.round(value)).join(",") ?? "";
      return `${person.trackId ?? person.personIndex}:${person.meshVertices.length}:${bbox}:${joints}`;
    })
    .join("|");
}

function videoPersonOverlayFit(
  person: VideoMeshPreview,
  rect: ReturnType<typeof videoContentRect>,
  bounds: ReturnType<typeof transformedMeshBounds>,
): OverlayAffineFit | OverlayBoxFit {
  const source: [number, number][] = [];
  const target: { x: number; y: number }[] = [];
  const kps3d = person.kps3d ?? [];
  const kps2d = person.kps2d ?? [];
  videoOverlayFitJoints.forEach((index) => {
    const sourcePoint = kps3d[index];
    const targetPoint = kps2d[index];
    if (!sourcePoint || !targetPoint) return;
    if (!sourcePoint.every(Number.isFinite) || !targetPoint.every(Number.isFinite)) return;
    if (
      targetPoint[0] < 0 ||
      targetPoint[1] < 0 ||
      targetPoint[0] > rect.sourceWidth ||
      targetPoint[1] > rect.sourceHeight
    ) {
      return;
    }
    source.push(overlaySourcePoint(sourcePoint));
    target.push(videoOverlayPoint(rect, targetPoint));
  });

  const matrix = solveAffine2d(source, target);
  if (matrix) {
    const sourceBounds = bounds2d(source);
    const targetBounds = bounds2d(target);
    const xyScale = Math.min(targetBounds.width / sourceBounds.width, targetBounds.height / sourceBounds.height);
    return {
      kind: "affine",
      matrix,
      zScale: Math.min(500, Math.max(0.001, xyScale * 0.42)),
      centerZ: bounds.center.z,
    };
  }

  return { kind: "box", ...videoPersonOverlayRect(person, rect) };
}

async function renderVideoBodyOverlay(frame: VideoMeshFrame) {
  updateVideoOverlayVisibilityClassOnly();
  if (!videoBodyOverlayEnabled() || !frame.people.length) {
    clearVideoBodyOverlay();
    return;
  }
  initVideoBodyOverlay();
  resizeVideoBodyOverlay();
  if (!videoOverlayGroup || !videoOverlayRenderer || !videoOverlayScene || !videoOverlayCamera) return;
  const rect = videoContentRect(frame);
  const key = `${frame.frameIndex}:${frame.people.length}:${mediaViewWidth}x${mediaViewHeight}:${rect.sourceWidth}x${rect.sourceHeight}:${videoOverlayFrameSignature(frame.people)}`;
  if (key === videoOverlayLastKey) return;
  const version = ++videoOverlayRenderVersion;

  for (const child of [...videoOverlayGroup.children]) {
    disposeMeshObject(child);
    videoOverlayGroup.remove(child);
  }

  const topology = await getMeshTopology();
  if (version !== videoOverlayRenderVersion) return;

  const people = sortedVideoPeople(frame.people).filter(
    (person) => person.meshVertices.length && (bboxLooksValid(person.bbox) || validVideoKps2d(person).length >= 8),
  );

  people.forEach((person, index) => {
    const bounds = transformedMeshBounds(person.meshVertices);
    if (bounds.box.isEmpty()) return;
    const fit = videoPersonOverlayFit(person, rect, bounds);
    const positions = new Float32Array(person.meshVertices.length * 3);

    person.meshVertices.forEach((point, vertexIndex) => {
      const transformed = transformMeshPoint(point);
      if (fit.kind === "affine") {
        const projected = applyAffine2d(fit.matrix, transformed.x, -transformed.y);
        positions[vertexIndex * 3] = projected.x;
        positions[vertexIndex * 3 + 1] = projected.y;
        positions[vertexIndex * 3 + 2] = (transformed.z - fit.centerZ) * fit.zScale;
      } else {
        const boxWidth = Math.max(8, fit.right - fit.left);
        const boxHeight = Math.max(8, fit.bottom - fit.top);
        const scaleByHeight = boxHeight / Math.max(0.001, bounds.size.y);
        const scaleByWidth = boxWidth / Math.max(0.001, bounds.size.x);
        const scale = Math.min(scaleByHeight, scaleByWidth);
        positions[vertexIndex * 3] = (transformed.x - bounds.center.x) * scale + (fit.left + fit.right) * 0.5;
        positions[vertexIndex * 3 + 1] = -(transformed.y - bounds.box.min.y) * scale + fit.bottom;
        positions[vertexIndex * 3 + 2] = (transformed.z - bounds.center.z) * scale * 0.6;
      }
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    if (topology.vertexCount === person.meshVertices.length) {
      geometry.setIndex(topology.indices);
    }
    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({
      color: index % 2 === 0 ? 0xb7c7da : 0xc6d2e2,
      depthWrite: true,
      metalness: 0.05,
      opacity: 0.54,
      roughness: 0.62,
      side: THREE.DoubleSide,
      transparent: true,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(0, 0, -index * 8);
    mesh.frustumCulled = false;
    videoOverlayGroup!.add(mesh);
  });

  videoOverlayRenderer.clear();
  videoOverlayRenderer.render(videoOverlayScene, videoOverlayCamera);
  videoOverlayLastKey = key;
}

function updateVideoOverlayVisibilityClassOnly() {
  mediaViewerEl.classList.toggle("overlay-enabled", activeMode === "video" && boolValue("#video-overlay-toggle"));
}

function inverseTransformMeshPoint(point: THREE.Vector3): [number, number, number] {
  return [-point.x, -point.y, point.z];
}

function approximateKps3dFromMesh(points: [number, number, number][]) {
  if (!points.length) return null;
  const box = new THREE.Box3();
  for (const point of points) {
    box.expandByPoint(transformMeshPoint(point));
  }
  if (box.isEmpty()) return null;
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  const h = Math.max(size.y, 0.001);
  const w = Math.max(size.x, 0.18);
  const z = center.z;
  const kps = Array.from({ length: 70 }, () => [0, 0, 0] as [number, number, number]);
  const set = (index: number, x: number, y: number, zed = z) => {
    kps[index] = inverseTransformMeshPoint(new THREE.Vector3(x, y, zed));
  };

  set(0, center.x, box.max.y, z);
  set(69, center.x, box.max.y - h * 0.2, z);
  set(5, center.x - w * 0.22, box.max.y - h * 0.27, z);
  set(6, center.x + w * 0.22, box.max.y - h * 0.27, z);
  set(7, center.x - w * 0.34, box.max.y - h * 0.45, z);
  set(8, center.x + w * 0.34, box.max.y - h * 0.45, z);
  set(62, center.x - w * 0.38, box.max.y - h * 0.62, z);
  set(41, center.x + w * 0.38, box.max.y - h * 0.62, z);
  set(9, center.x - w * 0.14, box.min.y + h * 0.48, z);
  set(10, center.x + w * 0.14, box.min.y + h * 0.48, z);
  set(11, center.x - w * 0.1, box.min.y + h * 0.25, z);
  set(12, center.x + w * 0.1, box.min.y + h * 0.25, z);
  set(13, center.x - w * 0.08, box.min.y + h * 0.04, z);
  set(14, center.x + w * 0.08, box.min.y + h * 0.04, z);
  return kps;
}

function videoPersonCenterX(person: VideoMeshPreview) {
  if (!person.bbox || !person.frameWidth) return person.personIndex;
  return ((person.bbox[0] + person.bbox[2]) * 0.5) / Math.max(1, person.frameWidth);
}

function sortedVideoPeople(people: VideoMeshPreview[]) {
  return [...people].sort((a, b) => (a.trackId ?? a.personIndex) - (b.trackId ?? b.personIndex));
}

function bboxArea(bbox: [number, number, number, number]) {
  return Math.max(0, bbox[2] - bbox[0]) * Math.max(0, bbox[3] - bbox[1]);
}

function bboxLooksValid(bbox?: [number, number, number, number] | null) {
  return !!bbox && bboxArea(bbox) > 64 && !(bbox[0] === 0 && bbox[1] === 0);
}

function bboxIou(a: [number, number, number, number], b: [number, number, number, number]) {
  const ix1 = Math.max(a[0], b[0]);
  const iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]);
  const iy2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  const union = bboxArea(a) + bboxArea(b) - inter;
  return union > 0 ? inter / union : 0;
}

function assignVideoTrackIds(frames: VideoMeshFrame[]) {
  const hasNativeTrackIds = frames.some((frame) =>
    frame.people.some((person) => typeof person.trackId === "number" && Number.isFinite(person.trackId)),
  );
  if (hasNativeTrackIds) {
    return frames.map((frame) => ({
      frameIndex: frame.frameIndex,
      people: frame.people.map((person) => ({
        ...person,
        trackId:
          typeof person.trackId === "number" && Number.isFinite(person.trackId)
            ? person.trackId
            : person.personIndex,
      })),
    }));
  }

  const tracks: { id: number; bbox: [number, number, number, number]; lastSeenFrame: number }[] = [];
  let nextTrackId = 0;
  const trackedFrames = frames.map((frame) => ({
    frameIndex: frame.frameIndex,
    people: frame.people.map((person) => ({ ...person })),
  }));

  for (const frame of trackedFrames) {
    const pairs: { personIndex: number; trackIndex: number; iou: number }[] = [];
    frame.people.forEach((person, personIndex) => {
      if (!bboxLooksValid(person.bbox)) return;
      tracks.forEach((track, trackIndex) => {
        if (frame.frameIndex - track.lastSeenFrame > 90) return;
        const iou = bboxIou(person.bbox!, track.bbox);
        if (iou >= 0.1) pairs.push({ personIndex, trackIndex, iou });
      });
    });
    pairs.sort((a, b) => b.iou - a.iou);

    const usedPeople = new Set<number>();
    const usedTracks = new Set<number>();
    for (const pair of pairs) {
      if (usedPeople.has(pair.personIndex) || usedTracks.has(pair.trackIndex)) continue;
      const person = frame.people[pair.personIndex];
      const track = tracks[pair.trackIndex];
      person.trackId = track.id;
      if (person.bbox) track.bbox = person.bbox;
      track.lastSeenFrame = frame.frameIndex;
      usedPeople.add(pair.personIndex);
      usedTracks.add(pair.trackIndex);
    }

    frame.people.forEach((person, personIndex) => {
      if (usedPeople.has(personIndex)) return;
      if (!bboxLooksValid(person.bbox)) {
        person.trackId = person.personIndex;
        return;
      }
      const track = { id: nextTrackId++, bbox: person.bbox!, lastSeenFrame: frame.frameIndex };
      tracks.push(track);
      person.trackId = track.id;
    });

    for (let index = tracks.length - 1; index >= 0; index -= 1) {
      if (frame.frameIndex - tracks[index].lastSeenFrame > 90) tracks.splice(index, 1);
    }
  }

  return trackedFrames;
}

function validPredCamT(person: VideoMeshPreview) {
  const cam = person.predCamT;
  return cam && cam.every((value) => Number.isFinite(value)) ? cam : null;
}

function updateMotionTrackReference(
  refs: Map<number, MotionTrackReference>,
  person: VideoMeshPreview,
  trackKey: number,
) {
  const existing = refs.get(trackKey);
  const next: MotionTrackReference = existing ?? {
    floorBottomY: 0,
    floorCenterY: 0,
    maxBoxHeight: 0.0001,
  };
  if (person.bbox && person.frameHeight) {
    const height = Math.max(1, person.frameHeight);
    const topY = person.bbox[1] / height;
    const bottomY = person.bbox[3] / height;
    const centerY = (topY + bottomY) * 0.5;
    const boxHeight = Math.max(0.0001, bottomY - topY);
    next.floorBottomY = existing ? Math.max(next.floorBottomY, bottomY) : bottomY;
    next.floorCenterY = existing ? Math.max(next.floorCenterY, centerY) : centerY;
    next.maxBoxHeight = existing ? Math.max(next.maxBoxHeight, boxHeight) : boxHeight;
  }
  const cam = validPredCamT(person);
  if (cam) {
    next.camFloorY = next.camFloorY === undefined ? cam[1] : Math.max(next.camFloorY, cam[1]);
    next.camBaseZ ??= cam[2];
  }
  if (!next.meshNorm && person.meshVertices.length) {
    next.meshNorm = meshNormalization(person.meshVertices);
  }
  refs.set(trackKey, next);
}

function rebuildMotionTrackReferences() {
  const refs = new Map<number, MotionTrackReference>();
  for (const frame of videoMeshFrames) {
    sortedVideoPeople(frame.people).forEach((person, trackIndex) => {
      updateMotionTrackReference(refs, person, person.trackId ?? trackIndex);
    });
  }
  motionTrackReferences = refs;
}

function bboxMotionOffset(
  person: VideoMeshPreview,
  trackKey: number,
  fallbackIndex: number,
  peopleCount: number,
): [number, number, number] | null {
  if (!person.bbox || !person.frameWidth || !person.frameHeight) return null;
  const width = Math.max(1, person.frameWidth);
  const height = Math.max(1, person.frameHeight);
  const aspect = width / height;
  const centerX = ((person.bbox[0] + person.bbox[2]) * 0.5) / width;
  const topY = person.bbox[1] / height;
  const bottomY = person.bbox[3] / height;
  const centerY = (topY + bottomY) * 0.5;
  const bboxReference = motionTrackReferences.get(trackKey) ?? {
    floorBottomY: bottomY,
    floorCenterY: centerY,
    maxBoxHeight: Math.max(0.0001, bottomY - topY),
  };
  const horizontalScale = Math.min(3.4, Math.max(1.6, aspect * 2.25));
  const fallbackSpread = peopleCount > 1 ? ((peopleCount - 1) * 0.5 - fallbackIndex) * 0.9 : 0;
  const x = Number.isFinite(centerX) ? (0.5 - centerX) * horizontalScale : fallbackSpread;
  const jumpFromBottom = (bboxReference.floorBottomY - bottomY) * 2.35;
  const jumpFromCenter = (bboxReference.floorCenterY - centerY) * 2.1;
  return [
    x,
    Math.max(0, jumpFromBottom, jumpFromCenter),
    0,
  ];
}

function videoMotionOffset(
  person: VideoMeshPreview,
  trackKey: number,
  fallbackIndex: number,
  peopleCount: number,
): [number, number, number] | null {
  const bboxOffset = bboxMotionOffset(person, trackKey, fallbackIndex, peopleCount);
  const cam = validPredCamT(person);
  const reference = motionTrackReferences.get(trackKey);
  if (cam) {
    const camX = -cam[0] * 0.45;
    const x = bboxOffset ? bboxOffset[0] * 0.9 + camX * 0.1 : camX;
    const y = Math.max(
      bboxOffset?.[1] ?? 0,
      ((reference?.camFloorY ?? cam[1]) - cam[1]) * 0.8,
      0,
    );
    return [
      x,
      y,
      (cam[2] - (reference?.camBaseZ ?? cam[2])) * 0.35,
    ];
  }
  return bboxOffset;
}

function videoFrameMotionOffsets(people: VideoMeshPreview[]) {
  const offsets = new Map<number, [number, number, number] | null>();
  const entries = people.map((person, index) => {
    const trackKey = person.trackId ?? index;
    const offset = videoMotionOffset(person, trackKey, index, people.length);
    offsets.set(trackKey, offset);
    return { person, index, trackKey, offset };
  });

  const separated = entries
    .filter((entry) => entry.offset && entry.person.bbox && entry.person.frameWidth)
    .sort((a, b) => videoPersonCenterX(a.person) - videoPersonCenterX(b.person));
  if (separated.length < 2) return offsets;

  const minGap = separated.length > 2 ? 0.68 : 0.82;
  const originalXs = separated.map((entry) => entry.offset![0]);
  const adjustedXs = [...originalXs];
  for (let index = 1; index < adjustedXs.length; index += 1) {
    const maxRightwardX = adjustedXs[index - 1] - minGap;
    if (adjustedXs[index] > maxRightwardX) adjustedXs[index] = maxRightwardX;
  }
  const originalMean = originalXs.reduce((sum, x) => sum + x, 0) / originalXs.length;
  const adjustedMean = adjustedXs.reduce((sum, x) => sum + x, 0) / adjustedXs.length;
  const shift = originalMean - adjustedMean;

  separated.forEach((entry, index) => {
    const offset = offsets.get(entry.trackKey);
    if (!offset) return;
    offsets.set(entry.trackKey, [adjustedXs[index] + shift, offset[1], offset[2]]);
  });
  return offsets;
}

async function renderMeshVertices(
  meshVertices: [number, number, number][],
  status: string,
  options: { preserveCamera?: boolean } = {},
) {
  await renderMeshPeople([{ meshVertices }], status, options);
}

async function renderMeshPeople(
  people: MeshRenderPerson[],
  status: string,
  options: { preserveCamera?: boolean; storeAs?: "image"; space?: "image" | "motion" } = {},
) {
  initMeshViewer();
  clearMeshPreview();
  lastMeshVertices = null;
  lastMeshTopology = null;

  const visiblePeople = people.filter((person) => person.meshVertices.length);
  if (!visiblePeople.length || !meshGroup) {
    setText("#mesh-state", "No mesh");
    return;
  }
  const group = meshGroup;

  try {
    const topology = await getMeshTopology();
    lastMeshVertices = visiblePeople[0].meshVertices;
    lastMeshTopology = topology;
    if (options.storeAs === "image") {
      lastImageMeshPeople = visiblePeople.map((person) => ({ ...person }));
      lastImageMeshStatus = status;
      lastImageMeshVertices = visiblePeople[0].meshVertices;
      lastImageMeshTopology = topology;
    }
    const imageCenterXs = visiblePeople.map((person, index) => person.imageCenterX ?? index);
    const minImageX = Math.min(...imageCenterXs);
    const maxImageX = Math.max(...imageCenterXs);
    const sourceSpread = maxImageX - minImageX;
    const layoutSpread = visiblePeople.length > 1 ? Math.min(0.9, Math.max(0.42, sourceSpread * 2.4)) : 0;
    const imageDepths = visiblePeople.map((person) => person.imageDepth);
    const validImageDepths = imageDepths.filter(
      (depth): depth is number => typeof depth === "number" && Number.isFinite(depth),
    );
    const minImageDepth = validImageDepths.length ? Math.min(...validImageDepths) : 0;
    const maxImageDepth = validImageDepths.length ? Math.max(...validImageDepths) : 0;
    const imageDepthSpread = maxImageDepth - minImageDepth;
    const layoutDepth =
      visiblePeople.length > 1 && imageDepthSpread > 0.02
        ? Math.min(1.35, Math.max(0.55, imageDepthSpread * 0.34))
        : 0;
    const useMotionSpace = options.space === "motion";
    if (useMotionSpace && !motionReference) {
      motionReference = meshNormalization(visiblePeople[0].meshVertices);
    }

    visiblePeople.forEach((person, personIndex) => {
      const personGroup = new THREE.Group();
      const normalizedX =
        visiblePeople.length > 1
          ? ((imageCenterXs[personIndex] - minImageX) / Math.max(sourceSpread, 0.0001) - 0.5) * layoutSpread
          : 0;
      const imageDepth = imageDepths[personIndex];
      const normalizedDepth =
        typeof imageDepth === "number" && Number.isFinite(imageDepth) && imageDepthSpread > 0.0001
          ? ((imageDepth - minImageDepth) / imageDepthSpread - 0.5) * layoutDepth
          : 0;
      personGroup.position.x = useMotionSpace ? (person.motionOffset?.[0] ?? 0) : -normalizedX;
      personGroup.position.y = useMotionSpace ? (person.motionOffset?.[1] ?? 0) : 0;
      personGroup.position.z = useMotionSpace ? (person.motionOffset?.[2] ?? 0) : normalizedDepth;

      const normalization =
        useMotionSpace && person.normalization
          ? person.normalization
          : useMotionSpace && person.motionOffset
            ? meshNormalization(person.meshVertices)
          : useMotionSpace && motionReference
            ? motionReference
            : meshNormalization(person.meshVertices);
      const geometry = new THREE.BufferGeometry();
      const positions = new Float32Array(person.meshVertices.length * 3);

      person.meshVertices.forEach((point, index) => {
        const transformed = transformMeshPoint(point);
        positions[index * 3] = transformed.x - normalization.centerX;
        positions[index * 3 + 1] = transformed.y - normalization.minY + gridY;
        positions[index * 3 + 2] = transformed.z - normalization.centerZ;
      });

      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      if (topology.vertexCount === person.meshVertices.length) {
        geometry.setIndex(topology.indices);
      }
      geometry.computeVertexNormals();

      const material = new THREE.MeshStandardMaterial({
        color: personIndex % 2 === 0 ? 0x25b8ab : 0xe0a12b,
        metalness: 0.05,
        roughness: 0.52,
        side: THREE.DoubleSide,
        wireframe: boolValue("#wireframe-toggle"),
      });
      const bodyMesh = new THREE.Mesh(geometry, material);
      bodyMesh.name = "mesh-body";
      bodyMesh.userData.meshBody = true;
      personGroup.add(bodyMesh);

      if (person.kps3d?.length) {
        personGroup.add(createMeshSkeleton(person.kps3d, normalization));
      }
      group.add(personGroup);
    });
    updateMeshSkeletonVisibility();
    updateMeshBodyVisibility();

    if (!options.preserveCamera) {
      fitMeshCamera();
    }
    setText("#mesh-state", status);
  } catch (error) {
    setText("#mesh-state", "Mesh failed");
    setOutput(String(error));
  }
}

function createMeshSkeleton(
  kps3d: [number, number, number][],
  normalization: { centerX: number; centerZ: number; minY: number },
) {
  const skeleton = new THREE.Group();
  skeleton.name = "mesh-skeleton";
  addSkeletonLines3d(skeleton, kps3d, mhrBodyEdges, normalization, 0x25b8ab);
  addSkeletonLines3d(skeleton, kps3d, mhrRightHandEdges, normalization, 0x6aa6ff);
  addSkeletonLines3d(skeleton, kps3d, mhrLeftHandEdges, normalization, 0xe85d75);
  return skeleton;
}

function addSkeletonLines3d(
  group: THREE.Group,
  points: [number, number, number][],
  edges: [number, number][],
  normalization: { centerX: number; centerZ: number; minY: number },
  color: number,
) {
  const positions: number[] = [];
  for (const [a, b] of edges) {
    const pa = points[a];
    const pb = points[b];
    if (!isValidMeshPoint(pa) || !isValidMeshPoint(pb)) continue;
    const va = transformMeshPoint(pa);
    const vb = transformMeshPoint(pb);
    positions.push(
      va.x - normalization.centerX,
      va.y - normalization.minY + gridY,
      va.z - normalization.centerZ,
      vb.x - normalization.centerX,
      vb.y - normalization.minY + gridY,
      vb.z - normalization.centerZ,
    );
  }
  if (!positions.length) return;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 });
  const lines = new THREE.LineSegments(geometry, material);
  lines.name = "mesh-skeleton";
  group.add(lines);
}

function isValidMeshPoint(point: [number, number, number] | undefined) {
  return !!point && point.every((value) => Number.isFinite(value)) && point.some((value) => Math.abs(value) > 1e-6);
}

function updateWireframe() {
  if (!meshGroup) return;
  meshGroup.traverse((child) => {
    const mesh = child as THREE.Mesh;
    const material = mesh.material;
    if (Array.isArray(material)) {
      material.forEach((entry) => {
        if ("wireframe" in entry) {
          (entry as THREE.MeshStandardMaterial).wireframe = boolValue("#wireframe-toggle");
        }
      });
    } else if (material && "wireframe" in material) {
      (material as THREE.MeshStandardMaterial).wireframe = boolValue("#wireframe-toggle");
    }
  });
}

function updateGridVisibility() {
  if (meshGrid) meshGrid.visible = boolValue("#grid-toggle");
}

function updateMeshSkeletonVisibility() {
  if (!meshGroup) return;
  meshGroup.traverse((child) => {
    if (child.name === "mesh-skeleton") {
      child.visible = boolValue("#mesh-skeleton-toggle");
    }
  });
}

function updateMeshBodyVisibility() {
  if (!meshGroup) return;
  meshGroup.traverse((child) => {
    if (child.name === "mesh-body" || child.userData.meshBody) {
      child.visible = boolValue("#mesh-body-toggle");
    }
  });
}

async function renderMesh(persons: PersonResult[]) {
  lastMeshPeople = persons;
  const meshPeople = persons
    .filter((candidate) => candidate.mesh_vertices?.length)
    .sort((a, b) => bboxCenterX(a) - bboxCenterX(b))
    .map((person) => ({
      meshVertices: person.mesh_vertices!,
      kps3d: person.kps_3d,
      imageCenterX: lastImageResult?.width ? bboxCenterX(person) / lastImageResult.width : undefined,
      imageDepth: imageDepthForPerson(person, lastImageResult?.width, lastImageResult?.height),
    }));
  if (!meshPeople.length) {
    lastImageMeshPeople = [];
    lastImageMeshVertices = null;
    lastImageMeshTopology = null;
    lastImageMeshStatus = "No mesh";
    await renderMeshVertices([], "No mesh");
    return;
  }
  const verts = meshPeople.reduce((sum, person) => sum + person.meshVertices.length, 0);
  await renderMeshPeople(
    meshPeople,
    `${meshPeople.length} people | ${verts.toLocaleString()} verts`,
    { storeAs: "image" },
  );
}

function bboxCenterX(person: PersonResult) {
  return (person.bbox[0] + person.bbox[2]) / 2;
}

function imageDepthForPerson(person: PersonResult, imageWidth = 0, imageHeight = 0) {
  const camZ = person.pred_cam_t?.[2];
  const validCamZ = typeof camZ === "number" && Number.isFinite(camZ) && camZ > 0 ? camZ : undefined;
  const bboxWidth = Math.max(1, person.bbox[2] - person.bbox[0]);
  const bboxHeight = Math.max(1, person.bbox[3] - person.bbox[1]);
  const frameMax = Math.max(1, imageWidth, imageHeight);
  const bboxScale = Math.max(bboxWidth, bboxHeight) / frameMax;
  const bboxDepth = bboxScale > 0 ? 1 / bboxScale : undefined;
  if (validCamZ !== undefined && bboxDepth !== undefined) {
    return validCamZ * 0.65 + bboxDepth * 0.35;
  }
  return validCamZ ?? bboxDepth;
}

function setRunStats(frames: string, people: string, fps = "-", time = "-") {
  setText("#video-frames", frames);
  setText("#video-people", people);
  setText("#video-fps", fps);
  setText("#video-time", time);
}

function updateMotionUi() {
  const controls = el<HTMLDivElement>("#motion-controls");
  const seek = el<HTMLInputElement>("#motion-seek");
  const playButton = el<HTMLButtonElement>("#motion-play");
  const hidden = activeMode !== "video" || videoMeshFrames.length === 0;
  controls.hidden = hidden;
  meshColumnEl.classList.toggle("no-motion", hidden);
  seek.max = String(Math.max(0, videoMeshFrames.length - 1));
  seek.value = String(Math.min(motionCursor, Math.max(0, videoMeshFrames.length - 1)));
  const nativeVideoPlaying = activeMode === "video" && !videoPreviewEl.paused && !videoPreviewEl.ended;
  playButton.textContent = motionPlaying || nativeVideoPlaying ? "Pause" : "Play";
  el<HTMLButtonElement>("#motion-loop").classList.toggle("active", motionLoop);
  setText(
    "#motion-frame-label",
    videoMeshFrames.length ? `${motionCursor + 1} / ${videoMeshFrames.length}` : "0 / 0",
  );
}

async function showMotionFrame(index: number, preserveCamera = true, syncVideo = true) {
  if (!videoMeshFrames.length) return;
  motionCursor = Math.min(Math.max(index, 0), videoMeshFrames.length - 1);
  const frame = videoMeshFrames[motionCursor];
  if (syncVideo && videoPreviewEl.src && lastVideoFps > 0) {
    const time = Math.max(0, frame.frameIndex / lastVideoFps);
    if (Number.isFinite(time) && Math.abs(videoPreviewEl.currentTime - time) > 0.04) {
      videoPreviewEl.currentTime = time;
    }
  }
  if (activeMode === "video") {
    rebuildMotionTrackReferences();
    const orderedPeople = sortedVideoPeople(frame.people);
    const frameOffsets = videoFrameMotionOffsets(orderedPeople);
    const people = orderedPeople.map((person, trackIndex) => ({
      meshVertices: person.meshVertices,
      kps3d: person.kps3d?.length ? person.kps3d : approximateKps3dFromMesh(person.meshVertices),
      imageCenterX:
        person.bbox && person.frameWidth
          ? ((person.bbox[0] + person.bbox[2]) * 0.5) / Math.max(1, person.frameWidth)
          : person.personIndex,
      motionOffset: frameOffsets.get(person.trackId ?? trackIndex) ?? null,
      normalization: motionTrackReferences.get(person.trackId ?? trackIndex)?.meshNorm,
    }));
    const verts = people.reduce((sum, person) => sum + person.meshVertices.length, 0);
    const displayPeople = Math.max(videoMaxPeopleSeen, people.length);
    const displayVerts = Math.max(verts, displayPeople * FSB_MESH_VERTEX_COUNT);
    await renderMeshPeople(
      people,
      `Motion f${frame.frameIndex + 1} | ${displayPeople} people | ${displayVerts.toLocaleString()} verts`,
      { preserveCamera, space: "motion" },
    );
    await renderVideoBodyOverlay(frame);
  }
  updateMotionUi();
}

function stopMotionPlayback() {
  motionPlaying = false;
  if (motionTimer) {
    window.clearTimeout(motionTimer);
    motionTimer = null;
  }
  updateMotionUi();
}

function scheduleMotionPlayback() {
  if (!motionPlaying || videoMeshFrames.length < 2) {
    stopMotionPlayback();
    return;
  }
  const speed = Number.parseFloat(value("#motion-speed")) || 1;
  const delay = Math.max(15, 1000 / Math.max(1, lastVideoFps) / speed);
  motionTimer = window.setTimeout(async () => {
    const next = motionCursor + 1;
    if (next >= videoMeshFrames.length && !motionLoop) {
      await showMotionFrame(videoMeshFrames.length - 1, true);
      stopMotionPlayback();
      return;
    }
    await showMotionFrame(next >= videoMeshFrames.length ? 0 : next, true);
    if (motionPlaying) scheduleMotionPlayback();
  }, delay);
}

function toggleMotionPlayback() {
  if (!videoMeshFrames.length) return;
  const nativeVideoPlaying = !videoPreviewEl.paused && !videoPreviewEl.ended;
  if (motionPlaying || nativeVideoPlaying) {
    videoPreviewEl.pause();
    stopMotionPlayback();
    return;
  }
  motionPlaying = !motionPlaying;
  if (motionPlaying) {
    videoPreviewEl.pause();
    scheduleMotionPlayback();
  } else {
    stopMotionPlayback();
  }
  updateMotionUi();
}

async function seekMotionTo(index: number) {
  const version = ++motionSeekVersion;
  await showMotionFrame(index, true);
  if (version === motionSeekVersion) updateMotionUi();
}

function seekMotionFromPointer(event: PointerEvent | MouseEvent) {
  if (!videoMeshFrames.length) return;
  const seek = el<HTMLInputElement>("#motion-seek");
  const rect = seek.getBoundingClientRect();
  const pct = rect.width > 0 ? (event.clientX - rect.left) / rect.width : 0;
  const index = Math.round(Math.min(1, Math.max(0, pct)) * Math.max(0, videoMeshFrames.length - 1));
  seek.value = String(index);
  void seekMotionTo(index);
}

function nearestMotionIndexForVideoTime(time: number) {
  if (!videoMeshFrames.length || !Number.isFinite(time)) return -1;
  const sourceFrame = time * Math.max(1, lastVideoFps);
  let bestIndex = 0;
  let bestDelta = Number.POSITIVE_INFINITY;
  for (let index = 0; index < videoMeshFrames.length; index += 1) {
    const delta = Math.abs(videoMeshFrames[index].frameIndex - sourceFrame);
    if (delta < bestDelta) {
      bestDelta = delta;
      bestIndex = index;
    }
  }
  return bestIndex;
}

function syncMotionToVideoPlayback() {
  if (activeMode !== "video" || !videoMeshFrames.length || motionPlaying || videoDrivenSyncPending) return;
  videoDrivenSyncPending = true;
  window.requestAnimationFrame(() => {
    videoDrivenSyncPending = false;
    const index = nearestMotionIndexForVideoTime(videoPreviewEl.currentTime);
    if (index >= 0 && index !== motionCursor) {
      void showMotionFrame(index, true, false);
    }
    updateMotionUi();
  });
}

function resetMotionFrames() {
  stopMotionPlayback();
  videoMeshFrames = [];
  videoMeshFrameMap = new Map<number, VideoMeshPreview[]>();
  motionReference = null;
  motionTrackReferences = new Map<number, MotionTrackReference>();
  motionCursor = 0;
  videoMaxPeopleSeen = 0;
  generatedMotionPath = "";
  generatedMotionVideoPath = "";
  clearVideoBodyOverlay();
  const extractButton = document.querySelector<HTMLButtonElement>("#extract-motion");
  if (extractButton) extractButton.disabled = true;
  updateMotionUi();
}

async function waitForMotionPreviewFrames(runToken: number, timeoutMs = 5000) {
  const deadline = performance.now() + timeoutMs;
  while (runToken === videoRunToken && !videoAbortRequested && videoMeshFrames.length === 0) {
    if (performance.now() >= deadline) break;
    await delay(50);
  }
}

function resetVideoRunUiForNewInput() {
  resetMotionFrames();
  setVideoRunning(false);
  setProgress("#video-progress-bar", 0, 1);
  setText("#video-progress-text", "No video running");
  setRunStats("0", "0", "-", "-");
  setOutput("");
}

function bindSteppers() {
  bindStepper("#frame-stride", "#frame-stride-down", "#frame-stride-up", 1);
  bindStepper("#max-frames", "#max-frames-down", "#max-frames-up", 10);
}

function bindStepper(inputId: string, downId: string, upId: string, step: number) {
  const input = el<HTMLInputElement>(inputId);
  const min = Number.parseInt(input.min || "0", 10);
  const setNext = (delta: number) => {
    const current = Number.parseInt(input.value || "0", 10) || 0;
    input.value = String(Math.max(min, current + delta));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  };
  el<HTMLButtonElement>(downId).addEventListener("click", () => setNext(-step));
  el<HTMLButtonElement>(upId).addEventListener("click", () => setNext(step));
}

async function copyOutput() {
  const text = lastOutputText;
  if (!text.trim()) return;
  await navigator.clipboard.writeText(text);
  showToast("Copied output");
}

async function downloadOutput() {
  const text = lastOutputText;
  if (!text.trim()) return;
  const extension = lastOutputKind === "json" ? "json" : lastOutputKind === "csv" ? "csv" : "txt";
  const defaultPath = suggestedSiblingPath(
    lastOutputBasePath || value("#image-path") || value("#video-path"),
    "_output",
    extension,
  );
  const selected = await saveDialog({
    defaultPath,
    filters: [{ name: extension.toUpperCase(), extensions: [extension] }],
  });
  if (!selected) return;
  const writtenPath = await invoke<string>("save_text_file", {
    request: { path: cleanPath(selected), content: text },
  });
  showToast(`Saved ${fileName(writtenPath)}`);
}

function resetMediaTransform(redraw = true) {
  mediaZoom = 1;
  mediaPanX = 0;
  mediaPanY = 0;
  setText("#media-zoom-state", "100%");
  if (redraw) drawCurrentImagePreview();
}

function clampMediaPan() {
  if (!lastImageResult) {
    mediaPanX = 0;
    mediaPanY = 0;
    return;
  }
  const baseScale = Math.min(mediaViewWidth / lastImageResult.width, mediaViewHeight / lastImageResult.height);
  const displayWidth = lastImageResult.width * baseScale * mediaZoom;
  const displayHeight = lastImageResult.height * baseScale * mediaZoom;
  const maxX = Math.max(0, (displayWidth - mediaViewWidth) / 2);
  const maxY = Math.max(0, (displayHeight - mediaViewHeight) / 2);
  mediaPanX = Math.min(maxX, Math.max(-maxX, mediaPanX));
  mediaPanY = Math.min(maxY, Math.max(-maxY, mediaPanY));
}

function setMediaZoom(nextZoom: number) {
  resizeMediaCanvas();
  mediaZoom = Math.min(6, Math.max(1, nextZoom));
  if (mediaZoom === 1) {
    mediaPanX = 0;
    mediaPanY = 0;
  }
  clampMediaPan();
  setText("#media-zoom-state", `${Math.round(mediaZoom * 100)}%`);
  drawCurrentImagePreview();
}

function bindMediaControls() {
  el<HTMLButtonElement>("#media-zoom-out").addEventListener("click", () => setMediaZoom(mediaZoom / 1.25));
  el<HTMLButtonElement>("#media-zoom-in").addEventListener("click", () => setMediaZoom(mediaZoom * 1.25));
  el<HTMLButtonElement>("#media-reset").addEventListener("click", () => resetMediaTransform(true));

  mediaViewerEl.addEventListener("wheel", (event) => {
    if (!lastImageResult || mediaViewerEl.classList.contains("video-mode")) return;
    event.preventDefault();
    setMediaZoom(mediaZoom * (event.deltaY < 0 ? 1.12 : 1 / 1.12));
  });

  canvas.addEventListener("pointerdown", (event) => {
    if (!lastImageResult || mediaZoom <= 1) return;
    mediaDragging = true;
    mediaDragStartX = event.clientX;
    mediaDragStartY = event.clientY;
    mediaDragOriginX = mediaPanX;
    mediaDragOriginY = mediaPanY;
    canvas.setPointerCapture(event.pointerId);
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!mediaDragging) return;
    mediaPanX = mediaDragOriginX + event.clientX - mediaDragStartX;
    mediaPanY = mediaDragOriginY + event.clientY - mediaDragStartY;
    clampMediaPan();
    drawCurrentImagePreview();
  });

  for (const name of ["pointerup", "pointercancel", "pointerleave"]) {
    canvas.addEventListener(name, () => {
      mediaDragging = false;
    });
  }
}

function applyTheme(theme: string) {
  currentTheme = theme === "light" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", currentTheme);
  localStorage.setItem("sam3dbody.theme", currentTheme);
  el<HTMLButtonElement>("#theme-dark").classList.toggle("active", currentTheme === "dark");
  el<HTMLButtonElement>("#theme-light").classList.toggle("active", currentTheme === "light");
  if (meshScene) meshScene.background = new THREE.Color(cssVar("--bg-inner", "#0d1013"));
  drawCurrentImagePreview();
  drawHardwareGraph();
}

function applyAccent(accent: string) {
  const allowed = ["teal", "blue", "green", "red", "yellow"];
  currentAccent = allowed.includes(accent) ? accent : "teal";
  document.documentElement.setAttribute("data-accent", currentAccent);
  localStorage.setItem("sam3dbody.accent", currentAccent);
  for (const button of document.querySelectorAll<HTMLButtonElement>("[data-accent-choice]")) {
    button.classList.toggle("active", button.dataset.accentChoice === currentAccent);
  }
  if (meshGrid) updateGridSize();
  drawCurrentImagePreview();
  drawHardwareGraph();
}

function applyUiScale(percent: number) {
  currentUiScale = Math.min(115, Math.max(90, percent || 100));
  const shouldResumeMotion = motionPlaying;
  document.documentElement.style.setProperty("--ui-scale", (currentUiScale / 100).toFixed(2));
  localStorage.setItem("sam3dbody.uiScale", String(currentUiScale));
  el<HTMLInputElement>("#ui-scale").value = String(currentUiScale);
  setText("#ui-scale-label", `${currentUiScale}%`);
  requestAnimationFrame(() => {
    resizeMeshViewer();
    drawCurrentImagePreview();
    drawHardwareGraph();
    videoOverlayLastKey = "";
    const frame = videoMeshFrames[motionCursor];
    if (frame) void renderVideoBodyOverlay(frame);
    if (shouldResumeMotion && !motionTimer) scheduleMotionPlayback();
    updateMotionUi();
  });
}

function initSettings() {
  const button = el<HTMLButtonElement>("#settings-button");
  const popover = el<HTMLDivElement>("#settings-popover");
  const setOpen = (open: boolean) => {
    popover.hidden = !open;
    button.classList.toggle("active", open);
  };
  button.addEventListener("click", () => setOpen(popover.hidden));
  document.addEventListener("pointerdown", (event) => {
    const target = event.target as Node;
    if (!popover.hidden && !popover.contains(target) && !button.contains(target)) setOpen(false);
  });
  el<HTMLButtonElement>("#theme-dark").addEventListener("click", () => applyTheme("dark"));
  el<HTMLButtonElement>("#theme-light").addEventListener("click", () => applyTheme("light"));
  for (const accentButton of document.querySelectorAll<HTMLButtonElement>("[data-accent-choice]")) {
    accentButton.addEventListener("click", () => applyAccent(accentButton.dataset.accentChoice || "teal"));
  }
  el<HTMLInputElement>("#ui-scale").addEventListener("input", (event) => {
    applyUiScale(Number.parseInt((event.target as HTMLInputElement).value, 10));
  });
  applyAccent(currentAccent);
  applyTheme(currentTheme);
  applyUiScale(currentUiScale);
}

function updateToolbarForMode() {
  for (const control of document.querySelectorAll<HTMLElement>(".image-only")) {
    control.hidden = activeMode !== "image";
  }
  for (const control of document.querySelectorAll<HTMLElement>(".video-only")) {
    control.hidden = activeMode !== "video";
  }
  updateVideoOverlayVisibility();
}

function imageResultPayload(result: ProcessResponse, sourcePath: string): ImageResultFile {
  return {
    kind: "sam3dbody.imageResult",
    version: 1,
    sourcePath,
    width: result.width,
    height: result.height,
    persons: result.persons,
  };
}

function normalizeImageResultPayload(payload: unknown): ImageResultFile {
  const candidate = Array.isArray(payload)
    ? { width: mediaViewWidth, height: mediaViewHeight, persons: payload }
    : payload;
  if (!candidate || typeof candidate !== "object") {
    throw new Error("JSON is not a SAM3DBody image result");
  }
  const result = candidate as Partial<ImageResultFile>;
  if (!Array.isArray(result.persons)) {
    throw new Error("JSON does not contain a persons array");
  }
  const width = Number(result.width) || Math.ceil(Math.max(1, ...result.persons.map((p) => Number(p?.bbox?.[2]) || 1)));
  const height = Number(result.height) || Math.ceil(Math.max(1, ...result.persons.map((p) => Number(p?.bbox?.[3]) || 1)));
  return {
    kind: result.kind || "sam3dbody.imageResult",
    version: result.version || 1,
    sourcePath: result.sourcePath || "",
    width,
    height,
    persons: result.persons as PersonResult[],
  };
}

async function importImageJson(path: string) {
  activateMode("image");
  setValue("#image-path", cleanPath(path));
  if (!value("#obj-path")) setValue("#obj-path", suggestedSiblingPath(path, "_mesh", imageExportExtension()));
  setText("#image-state", "Loading JSON");

  try {
    const text = await invoke<string>("read_text_file", { path: cleanPath(path) });
    const payload = normalizeImageResultPayload(JSON.parse(text));
    const hasMesh = payload.persons.some((person) => Array.isArray(person.mesh_vertices) && person.mesh_vertices.length);
    if (!hasMesh) {
      showToast("JSON has no mesh vertices", "warning");
      setText("#image-state", "JSON incomplete");
      setOutput("This JSON cannot recreate the 3D view because it does not contain mesh_vertices. New JSON exported from this version will include them.");
      return;
    }

    const imagePath = payload.sourcePath && !isJsonPath(payload.sourcePath) ? cleanPath(payload.sourcePath) : "";
    lastImageResult = payload;
    lastImagePath = imagePath;
    lastImageElement = imagePath ? await loadPreviewImage(imagePath) : null;
    lastImageFailed = !!imagePath && !lastImageElement;
    drawCurrentImagePreview();
    await renderMesh(payload.persons);
    setText("#image-state", "JSON loaded");
    setText("#people-count", String(payload.persons.length));
    setRunStats("1", String(payload.persons.length), "-", "-");
    setText("#image-progress-text", `Loaded JSON | ${payload.persons.length} people`);
    setOutput(text, "json", path);
    showToast("JSON loaded");
  } catch (error) {
    setText("#image-state", "JSON failed");
    setOutput(String(error));
    showToast("Could not load JSON", "warning");
  }
}

async function restoreImageMesh() {
  if (lastImageMeshPeople.length) {
    await renderMeshPeople(lastImageMeshPeople, lastImageMeshStatus, { preserveCamera: true });
    return;
  }
  if (activeMode === "image") {
    clearMeshPreview();
    setText("#mesh-state", lastImageResult?.persons.length ? "No mesh" : "Mesh idle");
  }
}

function activateMode(mode: "image" | "video") {
  activeMode = mode;
  const image = mode === "image";
  el("#image-tab").classList.toggle("active", image);
  el("#video-tab").classList.toggle("active", !image);
  el("#image-mode").classList.toggle("active", image);
  el("#video-mode").classList.toggle("active", !image);
  setMediaMode(mode);
  updateToolbarForMode();
  updateMotionUi();
  if (image) {
    void restoreImageMesh();
  } else if (videoMeshFrames.length) {
    void showMotionFrame(motionCursor, true);
  } else {
    clearMeshPreview();
    setText("#mesh-state", "Video mesh idle");
  }
}

async function loadEngine() {
  const button = el<HTMLButtonElement>("#load-engine");
  button.disabled = true;
  setText("#engine-chip", "Loading");
  setOutput("");

  try {
    const response = await invoke<LoadResponse>("load_engine", {
      request: {
        libraryPath: value("#library-path"),
        onnxDir: value("#onnx-dir"),
        ggufPath: value("#gguf-path"),
        yoloPath: value("#yolo-path"),
        backboneName: value("#backbone-name") || "backbone.onnx",
        cudaDevice: intValue("#cuda-device"),
        useTrtEp: false,
        useFp16: boolValue("#use-fp16"),
        skipBodyModel: boolValue("#skip-body-model"),
        maxPersons: intValue("#max-persons"),
        personThresh: 0.5,
        personNmsIou: 0.45,
      },
    });
    setEngineLoaded(response.loaded);
    setText("#abi-version", String(response.result_version));
    if (response.loaded) {
      saveUserPaths();
      showToast("Engine loaded");
    }
  } catch (error) {
    setEngineLoaded(false);
    setText("#engine-chip", "Load failed");
    setText("#dll-state", "Failed");
    setOutput(String(error));
  } finally {
    button.disabled = false;
  }
}

async function unloadEngine() {
  const button = el<HTMLButtonElement>("#unload-engine");
  button.disabled = true;
  try {
    await invoke("unload_engine");
    setEngineLoaded(false);
    setText("#abi-version", "-");
    setText("#mesh-state", lastMeshVertices?.length ? "Preview retained" : "Mesh idle");
    setOutput("Engine unloaded.");
    showToast("Engine unloaded");
  } catch (error) {
    setOutput(String(error));
    button.disabled = !engineLoaded;
  }
}

async function refreshModelPaths() {
  const button = el<HTMLButtonElement>("#refresh-models");
  button.disabled = true;
  setText("#model-chip", "Checking");
  setText("#model-state", "Checking");
  setText("#model-progress-text", "Searching for local model files");
  setProgress("#model-progress-bar", 0, 1);
  setOutput("");

  try {
    const models = await invoke<ModelPaths | null>("default_model_paths");
    if (models) {
      applyModelPaths(models, true);
      setOutput(`Models ready:\n${models.onnxDir}`);
    } else {
      setText("#model-chip", "Models missing");
      setText("#model-state", "Missing");
      setText("#model-progress-text", "No local model bundle found");
      setOutput("No local model bundle was found. Put the extracted files in an onnx folder beside the app, or set ONNX/GGUF/YOLO manually.");
      setProgress("#model-progress-bar", 0, 1);
    }
  } catch (error) {
    setText("#model-chip", "Check failed");
    setText("#model-state", "Failed");
    setText("#model-progress-text", String(error));
    setOutput(String(error));
  } finally {
    button.disabled = false;
  }
}

type MediaKind = "image" | "video";

const imageExtensions = ["jpg", "jpeg", "png", "webp", "bmp", "json"];
const videoExtensions = ["mp4", "mov", "mkv", "avi", "webm", "m4v"];

function mediaKindFromPath(path: string): MediaKind | null {
  const ext = path.split(".").pop()?.toLowerCase();
  if (!ext) return null;
  if (imageExtensions.includes(ext)) return "image";
  if (videoExtensions.includes(ext)) return "video";
  return null;
}

function applyMediaPath(kind: MediaKind, path: string) {
  path = cleanPath(path);
  if (kind === "image") {
    activateMode("image");
    setValue("#image-path", path);
    if (!value("#obj-path")) setValue("#obj-path", suggestedSiblingPath(path, "_mesh", imageExportExtension()) || defaultObjPath(path));
    if (isJsonPath(path)) {
      void importImageJson(path);
      return;
    }
    setText("#image-state", "Ready");
    void previewImageOnly(path);
  } else {
    const previousVideo = cleanPath(value("#video-path"));
    if (previousVideo && previousVideo !== path) {
      resetVideoRunUiForNewInput();
      clearMeshPreview();
      setText("#mesh-state", "Video mesh idle");
    }
    activateMode("video");
    setValue("#video-path", path);
    if (!value("#bvh-path")) setValue("#bvh-path", defaultBvhPath(path));
    setVideoPreviewSource(path);
    setText("#video-state", "Ready");
  }
  lastOutputBasePath = path;
  setOutput(path);
}

function handleVideoPathEdited() {
  const nextPath = cleanPath(value("#video-path"));
  if (!nextPath || !generatedMotionVideoPath || nextPath === cleanPath(generatedMotionVideoPath)) return;
  resetVideoRunUiForNewInput();
  clearMeshPreview();
  setText("#mesh-state", "Video mesh idle");
  setText("#video-state", "Ready");
  setText("#video-progress-text", "Video changed; generate motion again");
}

async function browseMedia(kind: MediaKind) {
  const selected = await openDialog({
    multiple: false,
    filters: [
      {
        name: kind === "image" ? "Images" : "Videos",
        extensions: kind === "image" ? imageExtensions : videoExtensions,
      },
    ],
  });
  const path = Array.isArray(selected) ? selected[0] : selected;
  if (typeof path === "string" && path.length > 0) {
    applyMediaPath(kind, cleanPath(path));
  }
}

async function browseLibraryPath() {
  const selected = await openDialog({
    multiple: false,
    filters: [{ name: "Engine DLL", extensions: ["dll"] }],
  });
  const path = Array.isArray(selected) ? selected[0] : selected;
  if (typeof path === "string" && path.length > 0) {
    setValue("#library-path", cleanPath(path));
    saveUserPath("#library-path");
    setText("#dll-state", "Custom");
    setOutput(cleanPath(path));
  }
}

type ModelBrowseKind = "onnxDir" | "gguf" | "yolo" | "backbone" | "lbs" | "bvhTemplate";

async function browseModelPath(kind: ModelBrowseKind) {
  const selected = await openDialog({
    multiple: false,
    directory: kind === "onnxDir",
    filters:
      kind === "onnxDir"
        ? undefined
        : [
            {
              name: modelBrowseTitle(kind),
              extensions: modelBrowseExtensions(kind),
            },
          ],
  });
  const rawPath = Array.isArray(selected) ? selected[0] : selected;
  const path = typeof rawPath === "string" ? cleanPath(rawPath) : "";
  if (!path) return;

  if (kind === "onnxDir") {
    await applyBrowsedOnnxDir(path);
    return;
  }

  const target = modelBrowseTarget(kind);
  setValue(target, kind === "backbone" ? path : path);
  if (!value("#onnx-dir") && ["gguf", "yolo", "backbone", "lbs"].includes(kind)) {
    const dir = parentDir(path);
    if (dir) setValue("#onnx-dir", dir);
  }
  if (kind === "backbone" && value("#onnx-dir")) {
    const dir = parentDir(path);
    if (dir === value("#onnx-dir")) setValue("#backbone-name", fileName(path));
  }
  saveUserPaths();
  setOutput(path);
}

async function applyBrowsedOnnxDir(path: string) {
  setValue("#onnx-dir", path);
  setText("#model-chip", "Checking");
  setText("#model-state", "Checking");
  setText("#model-progress-text", "Checking selected model folder");
  setProgress("#model-progress-bar", 0, 1);

  try {
    const models = await invoke<ModelPaths>("model_paths_for_dir", { path });
    setValue("#onnx-dir", models.onnxDir);
    setValue("#gguf-path", models.ggufPath);
    setValue("#yolo-path", models.yoloPath);
    setValue("#backbone-name", models.backboneName);
    setValue("#lbs-path", models.lbsPath);
    if (models.bvhTemplatePath) setValue("#bvh-template-path", models.bvhTemplatePath);
    applyModelPaths(models, true);
    setOutput(`Models selected:\n${models.onnxDir}`);
  } catch (error) {
    setText("#model-chip", "Models incomplete");
    setText("#model-state", "Missing");
    setText("#model-progress-text", String(error));
    setOutput(`${String(error)}\n\nChoose the folder that directly contains pipeline.gguf, yolo.onnx, decoder.onnx, backbone.onnx, backbone.onnx.data, body_model.lbs, correctives.bin, and keypoint_mapping.bin.`);
    setProgress("#model-progress-bar", 0, 1);
  }
}

function modelBrowseTitle(kind: ModelBrowseKind) {
  switch (kind) {
    case "gguf":
      return "GGUF";
    case "yolo":
      return "YOLO";
    case "backbone":
      return "Backbone";
    case "lbs":
      return "Body Model";
    case "bvhTemplate":
      return "BVH Template";
    default:
      return "Model Folder";
  }
}

function modelBrowseExtensions(kind: ModelBrowseKind) {
  switch (kind) {
    case "gguf":
      return ["gguf"];
    case "yolo":
    case "backbone":
      return ["onnx"];
    case "lbs":
      return ["lbs"];
    case "bvhTemplate":
      return ["bvh"];
    default:
      return [];
  }
}

function modelBrowseTarget(kind: ModelBrowseKind) {
  switch (kind) {
    case "gguf":
      return "#gguf-path";
    case "yolo":
      return "#yolo-path";
    case "backbone":
      return "#backbone-name";
    case "lbs":
      return "#lbs-path";
    case "bvhTemplate":
      return "#bvh-template-path";
    default:
      return "#onnx-dir";
  }
}

function bindDropzone(id: string, expectedKind: MediaKind) {
  const zone = el<HTMLDivElement>(id);
  const setActive = (active: boolean) => zone.classList.toggle("drag-active", active);

  for (const name of ["dragenter", "dragover"]) {
    zone.addEventListener(name, (event) => {
      event.preventDefault();
      event.stopPropagation();
      setActive(true);
    });
  }

  for (const name of ["dragleave", "dragend"]) {
    zone.addEventListener(name, (event) => {
      event.preventDefault();
      event.stopPropagation();
      setActive(false);
    });
  }

  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    event.stopPropagation();
    setActive(false);

    const file = event.dataTransfer?.files?.[0] as (File & { path?: string }) | undefined;
    if (!file?.path) {
      setOutput("Drop path was hidden by WebView. Use Browse to select the file path.");
      return;
    }

    const droppedPath = cleanPath(file.path);
    const actualKind = mediaKindFromPath(droppedPath);
    if (actualKind && actualKind !== expectedKind) {
      setOutput(`That looks like a ${actualKind} file. Drop it on the ${actualKind} panel instead.`);
      return;
    }
    applyMediaPath(expectedKind, droppedPath);
  });
}

function setDropzoneActive(kind: MediaKind | null, active: boolean) {
  for (const zoneKind of ["image", "video"] as MediaKind[]) {
    el<HTMLDivElement>(`#${zoneKind}-dropzone`).classList.toggle(
      "drag-active",
      active && (!kind || kind === zoneKind),
    );
  }
}

function clearDropzoneActive() {
  setDropzoneActive(null, false);
}

async function openExternalUrl(url: string, label = "link") {
  try {
    await invoke("open_external_url", { url });
    showToast(`Opening ${label}`);
  } catch (error) {
    setOutput(String(error));
    showToast("Could not open browser", "warning");
  }
}

function bindExternalLinks() {
  document.querySelectorAll<HTMLAnchorElement>(".link-grid a").forEach((anchor) => {
    anchor.addEventListener("click", (event) => {
      event.preventDefault();
      void openExternalUrl(anchor.href, anchor.textContent?.trim() || "model link");
    });
  });

  document.querySelectorAll<HTMLElement>("[data-url]").forEach((button) => {
    button.addEventListener("click", () => {
      const url = button.dataset.url;
      if (url) void openExternalUrl(url, button.textContent?.trim() || "link");
    });
  });
}

async function processImage() {
  if (!requireEngineLoaded()) return;
  const button = el<HTMLButtonElement>("#process-image");
  button.disabled = true;
  beginImageProgress();
  setText("#image-state", "Running");
  setOutput("");
  const objPath = value("#obj-path") || defaultObjPath(value("#image-path"));
  if (objPath && !value("#obj-path")) setValue("#obj-path", objPath);

  let ok = false;
  try {
    const result = await invoke<ProcessResponse>("process_image", {
      request: {
        imagePath: value("#image-path"),
        maxResults: 16,
      },
    });
    await drawResults(result, value("#image-path"));
    await renderMesh(result.persons);
    setText("#image-state", "Complete");
    setText("#people-count", String(result.persons.length));
    const elapsed = ((performance.now() - imageStartedAt) / 1000).toFixed(1);
    setRunStats("1", String(result.persons.length), "-", `${elapsed}s`);
    const payload = imageResultPayload(result, value("#image-path"));
    setOutput(JSON.stringify(payload, null, 2), "json", value("#image-path"));
    setText("#image-progress-text", `Complete | ${result.persons.length} people`);
    ok = true;
  } catch (error) {
    setText("#image-state", "Failed");
    setText("#image-progress-text", "Failed");
    setOutput(String(error));
  } finally {
    finishImageProgress(ok);
    button.disabled = false;
  }
}

function imageExportExtension() {
  return value("#image-export-format") === "glb" ? "glb" : "obj";
}

function updateImageExportFormatUi() {
  const extension = imageExportExtension();
  const button = el<HTMLButtonElement>("#export-obj");
  button.textContent = `Export ${extension.toUpperCase()}`;
  const current = value("#obj-path");
  if (current && /\.(obj|glb)$/i.test(current)) {
    setValue("#obj-path", `${stripExtension(current)}.${extension}`);
  }
}

function imageExportLayout(people: MeshRenderPerson[]) {
  const imageCenterXs = people.map((person, index) => person.imageCenterX ?? index);
  const minImageX = Math.min(...imageCenterXs);
  const maxImageX = Math.max(...imageCenterXs);
  const sourceSpread = maxImageX - minImageX;
  const layoutSpread = people.length > 1 ? Math.min(0.9, Math.max(0.42, sourceSpread * 2.4)) : 0;
  const imageDepths = people.map((person) => person.imageDepth);
  const validImageDepths = imageDepths.filter(
    (depth): depth is number => typeof depth === "number" && Number.isFinite(depth),
  );
  const minImageDepth = validImageDepths.length ? Math.min(...validImageDepths) : 0;
  const maxImageDepth = validImageDepths.length ? Math.max(...validImageDepths) : 0;
  const imageDepthSpread = maxImageDepth - minImageDepth;
  const layoutDepth =
    people.length > 1 && imageDepthSpread > 0.02
      ? Math.min(1.35, Math.max(0.55, imageDepthSpread * 0.34))
      : 0;
  return people.map((person, index) => {
    const normalizedX =
      people.length > 1
        ? ((imageCenterXs[index] - minImageX) / Math.max(sourceSpread, 0.0001) - 0.5) * layoutSpread
        : 0;
    const imageDepth = imageDepths[index];
    const normalizedDepth =
      typeof imageDepth === "number" && Number.isFinite(imageDepth) && imageDepthSpread > 0.0001
        ? ((imageDepth - minImageDepth) / imageDepthSpread - 0.5) * layoutDepth
        : 0;
    return {
      person,
      offsetX: -normalizedX,
      offsetZ: normalizedDepth,
      normalization: meshNormalization(person.meshVertices),
    };
  });
}

function transformedExportPoint(
  point: [number, number, number],
  normalization: { centerX: number; centerZ: number; minY: number },
  offsetX: number,
  offsetZ: number,
) {
  const transformed = transformMeshPoint(point);
  return [
    transformed.x - normalization.centerX + offsetX,
    transformed.y - normalization.minY,
    transformed.z - normalization.centerZ + offsetZ,
  ] as [number, number, number];
}

async function buildImageObjText() {
  const people = lastImageMeshPeople.filter((person) => person.meshVertices.length);
  if (!people.length) throw new Error("Run image inference with mesh output before exporting.");
  const topology = lastImageMeshTopology ?? (await getMeshTopology());
  const lines = ["# SAM3DBody mesh export"];
  let vertexOffset = 0;
  for (const [personIndex, entry] of imageExportLayout(people).entries()) {
    lines.push(`o person_${personIndex + 1}`);
    for (const point of entry.person.meshVertices) {
      const [x, y, z] = transformedExportPoint(point, entry.normalization, entry.offsetX, entry.offsetZ);
      lines.push(`v ${x} ${y} ${z}`);
    }
    for (let i = 0; i < topology.indices.length; i += 3) {
      lines.push(
        `f ${topology.indices[i] + 1 + vertexOffset} ${topology.indices[i + 1] + 1 + vertexOffset} ${topology.indices[i + 2] + 1 + vertexOffset}`,
      );
    }
    vertexOffset += entry.person.meshVertices.length;
  }
  return lines.join("\n");
}

async function buildImageGlb() {
  const people = lastImageMeshPeople.filter((person) => person.meshVertices.length);
  if (!people.length) throw new Error("Run image inference with mesh output before exporting.");
  const topology = lastImageMeshTopology ?? (await getMeshTopology());
  const group = new THREE.Group();
  group.name = "SAM3DBody_Image_Result";
  for (const [personIndex, entry] of imageExportLayout(people).entries()) {
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(entry.person.meshVertices.length * 3);
    entry.person.meshVertices.forEach((point, index) => {
      const [x, y, z] = transformedExportPoint(point, entry.normalization, entry.offsetX, entry.offsetZ);
      positions[index * 3] = x;
      positions[index * 3 + 1] = y;
      positions[index * 3 + 2] = z;
    });
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    if (topology.vertexCount === entry.person.meshVertices.length) geometry.setIndex(topology.indices);
    geometry.computeVertexNormals();
    const material = new THREE.MeshStandardMaterial({
      color: personIndex % 2 === 0 ? 0x25b8ab : 0xe0a12b,
      roughness: 0.52,
      metalness: 0.05,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = `person_${personIndex + 1}`;
    group.add(mesh);
  }
  const exporter = new GLTFExporter();
  try {
    return await new Promise<ArrayBuffer>((resolve, reject) => {
      exporter.parse(
        group,
        (result) => {
          if (result instanceof ArrayBuffer) resolve(result);
          else resolve(new TextEncoder().encode(JSON.stringify(result)).buffer);
        },
        (error) => reject(error),
        { binary: true },
      );
    });
  } finally {
    disposeMeshObject(group);
  }
}

function arrayBufferToBase64(buffer: ArrayBuffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

async function exportObj() {
  const extension = imageExportExtension();
  const defaultPath =
    value("#obj-path") ||
    suggestedSiblingPath(value("#image-path"), "_mesh", extension) ||
    defaultObjPath(value("#image-path"));

  if (!lastImageMeshPeople.some((person) => person.meshVertices.length)) {
    setOutput("Run image inference with mesh output before exporting OBJ.");
    return;
  }

  try {
    const selected = await saveDialog({
      defaultPath,
      filters: [{ name: extension === "glb" ? "Binary glTF" : "Wavefront OBJ", extensions: [extension] }],
    });
    if (!selected) return;
    const outputPath = cleanPath(selected);
    setValue("#obj-path", outputPath);
    const writtenPath =
      extension === "glb"
        ? await invoke<string>("save_binary_file", {
            request: { path: outputPath, base64: arrayBufferToBase64(await buildImageGlb()) },
          })
        : await invoke<string>("save_text_file", {
            request: { path: outputPath, content: await buildImageObjText() },
          });
    setOutput(`${extension.toUpperCase()} written:\n${writtenPath}`, "text", value("#image-path"));
    showToast(`Saved ${fileName(writtenPath)}`);
  } catch (error) {
    setOutput(String(error));
  }
}

async function exportImageBvh() {
  if (!requireEngineLoaded()) return;
  const imagePath = cleanPath(value("#image-path"));
  if (!imagePath || isJsonPath(imagePath)) {
    showToast("Image BVH needs the source image path", "warning");
    setOutput("Static image BVH is exported by rerunning the native image through the BVH writer. A JSON-only result does not contain the full pose channels needed for BVH.");
    return;
  }

  const selected = await saveDialog({
    defaultPath: suggestedSiblingPath(imagePath, "_static_motion", "bvh"),
    filters: [{ name: "BVH motion", extensions: ["bvh"] }],
  });
  if (!selected) return;

  const outputPath = cleanPath(selected);
  setText("#image-state", "Exporting BVH");
  setText("#image-progress-text", "Exporting one-frame BVH");
  try {
    const result = await invoke<ImageBvhResponse>("process_image_bvh", {
      request: {
        imagePath,
        bvhPath: outputPath,
        bvhTemplatePath: value("#bvh-template-path"),
        lbsPath: value("#lbs-path"),
        maxResults: Math.max(1, intValue("#max-persons") || 16),
      },
    });
    setText("#image-state", "Complete");
    setOutput(
      result.paths.length
        ? `Static BVH exported:\n${result.paths.join("\n")}\n\nPeople: ${result.people}`
        : "No people were detected, so no BVH file was written.",
      "text",
      imagePath,
    );
    showToast(
      result.paths.length
        ? `Saved ${result.paths.length} BVH ${result.paths.length === 1 ? "file" : "files"}`
        : "No people detected",
      result.paths.length ? "success" : "warning",
    );
  } catch (error) {
    setText("#image-state", "BVH failed");
    setOutput(String(error));
    showToast("Image BVH failed", "warning");
  }
}

async function chooseBvhOutput() {
  const defaultPath = value("#bvh-path") || suggestedSiblingPath(value("#video-path"), "_motion", "bvh");
  const selected = await saveDialog({
    defaultPath,
    filters: [{ name: "BVH motion", extensions: ["bvh"] }],
  });
  if (selected) setValue("#bvh-path", cleanPath(selected));
}

function setVideoRunning(running: boolean) {
  el<HTMLButtonElement>("#process-video").disabled = running;
  el<HTMLButtonElement>("#extract-motion").disabled = running || !generatedMotionPath;
  const abortButton = el<HTMLButtonElement>("#abort-video");
  abortButton.hidden = !running;
  abortButton.disabled = !running || videoAbortRequested;
}

function abortVideoExport() {
  if (videoAbortRequested) return;
  videoAbortRequested = true;
  stopMotionPlayback();
  const abortButton = el<HTMLButtonElement>("#abort-video");
  abortButton.disabled = true;
  setText("#video-state", "Aborting");
  setText("#video-progress-text", "Abort requested; finishing the current native frame");
  showToast("Abort requested", "warning");
}

async function processVideo() {
  if (!requireEngineLoaded()) return;
  const videoPath = cleanPath(value("#video-path"));
  if (!videoPath) {
    showToast("Choose a video first", "warning");
    return;
  }
  if (!value("#bvh-path")) setValue("#bvh-path", defaultBvhPath(videoPath));
  const csvPath = value("#csv-path");

  const runToken = ++videoRunToken;
  videoAbortRequested = false;
  resetMotionFrames();
  activateMode("video");
  if (videoPath) {
    setVideoPreviewSource(videoPath);
  }
  activeWork += 1;
  setVideoRunning(true);
  setText("#video-state", "Running");
  setText("#video-progress-text", "Generating motion preview");
  setProgress("#video-progress-bar", 0, 1);
  setOutput("");

  try {
    const result = await invoke<VideoResponse>("process_video", {
      request: {
        videoPath,
        bvhPath: "",
        bvhTemplatePath: value("#bvh-template-path"),
        lbsPath: value("#lbs-path"),
        csvPath,
        maxFrames: intValue("#max-frames"),
        frameStride: intValue("#frame-stride") || 1,
        maxResults: Math.max(1, intValue("#max-persons") || 16),
        butterworth: boolValue("#butterworth-toggle"),
        bwCutoff: floatValue("#bw-cutoff") || 6,
        butterworthRootRotation: boolValue("#root-rotation-filter"),
        rotClampDeg: floatValue("#rot-clamp-deg"),
      },
    });
    if (runToken !== videoRunToken || videoAbortRequested) {
      setText("#video-state", "Aborted");
      setText("#video-progress-text", "Aborted by user");
      setOutput("Video export was aborted in the UI. If the native worker had already written a partial BVH, choose a new output path before running again.");
      return;
    }
    if (result.framesProcessed <= 0) {
      setText("#video-state", "No frames");
      setText("#video-progress-text", "No video frames were processed");
      setRunStats("0", "0", "-", "-");
      setOutput(
        `No video frames were processed.\n\nVideo path:\n${videoPath}\n\nCheck that FFmpeg/OpenCV can read the file, then try again.`,
        "text",
        videoPath,
      );
      showToast("No video frames processed", "warning");
      return;
    }
    if (result.personsTotal <= 0) {
      setText("#video-state", "No people");
      setText("#video-progress-text", "No people detected in processed frames");
      setRunStats(String(result.framesProcessed), "0", result.sourceFps > 0 ? result.sourceFps.toFixed(1) : "-", `${(result.totalMs / 1000).toFixed(1)}s`);
      setOutput(
        `Video processed, but no people were detected.\n\nFrames processed: ${result.framesProcessed}`,
        "text",
        videoPath,
      );
      showToast("No people detected", "warning");
      return;
    }
    if ((result.meshPreviews ?? 0) <= 0) {
      setText("#video-state", "No 3D preview");
      setText("#video-progress-text", "Native video path did not emit mesh preview frames");
      setOutput(
        `Motion generation finished, but no 3D preview meshes were emitted.\n\nThis usually means the engine was loaded with Skip mesh enabled or the loaded DLL does not expose video mesh preview callbacks.\n\nFrames processed: ${result.framesProcessed}\nPeople detected: ${result.personsTotal}`,
        "text",
        videoPath,
      );
      showToast("No 3D motion preview", "warning");
      return;
    }
    await waitForMotionPreviewFrames(runToken);
    if (runToken !== videoRunToken || videoAbortRequested) {
      setText("#video-state", "Aborted");
      setText("#video-progress-text", "Aborted by user");
      return;
    }
    if (!videoMeshFrames.length) {
      setText("#video-state", "Preview delayed");
      setText("#video-progress-text", "Motion data finished but preview frames did not reach the UI");
      setOutput(
        `Motion generation finished, but the UI did not receive preview frames in time.\n\nNative preview meshes emitted: ${result.meshPreviews}\nFrames processed: ${result.framesProcessed}\nPeople detected: ${result.personsTotal}`,
        "text",
        videoPath,
      );
      showToast("Motion preview did not arrive", "warning");
      return;
    }
    lastVideoFps = result.sourceFps > 0 ? result.sourceFps : lastVideoFps;
    const averagePeople =
      result.framesProcessed > 0 ? Math.round(result.personsTotal / result.framesProcessed) : 0;
    const peopleEstimate = Math.max(videoMaxPeopleSeen, averagePeople);
    generatedMotionPath = result.bvhPath;
    generatedMotionVideoPath = videoPath;
    setText("#video-state", "Complete");
    setRunStats(
      String(result.framesProcessed),
      String(peopleEstimate),
      result.sourceFps > 0 ? result.sourceFps.toFixed(1) : "-",
      `${(result.totalMs / 1000).toFixed(1)}s`,
    );
    setOutput(
      `Motion generated:\n${result.bvhPath}` +
        `\n\nClick Extract Motion to save BVH output.` +
        (result.csvPath ? `\n\nDiagnostic CSV written:\n${result.csvPath}` : "") +
        `\n\nFrames processed: ${result.framesProcessed}` +
        `\nPer-frame detections: ${result.personsTotal}` +
        `\nPreview meshes: ${result.meshPreviews}` +
        `\nPeople shown in UI: ${peopleEstimate}`,
      "text",
      videoPath,
    );
    if (videoMeshFrames.length) {
      await showMotionFrame(Math.min(motionCursor, videoMeshFrames.length - 1), true);
    }
    el<HTMLButtonElement>("#extract-motion").disabled = false;
    showToast("Motion generated");
  } catch (error) {
    if (videoAbortRequested) {
      setText("#video-state", "Aborted");
      setText("#video-progress-text", "Aborted by user");
    } else {
      setText("#video-state", "Failed");
      setOutput(String(error));
      showToast("Motion generation failed", "warning");
    }
  } finally {
    activeWork = Math.max(0, activeWork - 1);
    if (runToken === videoRunToken) {
      videoAbortRequested = false;
      setVideoRunning(false);
    }
  }
}

async function extractMotion() {
  if (!generatedMotionPath) {
    showToast("Generate motion first", "warning");
    return;
  }
  const defaultPath =
    value("#bvh-path") || suggestedSiblingPath(generatedMotionVideoPath || value("#video-path"), "_motion", "bvh");
  const selected = await saveDialog({
    defaultPath,
    filters: [{ name: "BVH motion", extensions: ["bvh"] }],
  });
  if (!selected) return;
  const outputPath = cleanPath(selected);
  const result = await invoke<CopyMotionOutputsResponse>("copy_motion_outputs", {
    request: { fromPath: generatedMotionPath, toPath: outputPath },
  });
  setValue("#bvh-path", outputPath);
  setOutput(`BVH extracted:\n${result.paths.join("\n")}`, "text", generatedMotionVideoPath || value("#video-path"));
  showToast(`Saved ${result.paths.length} BVH ${result.paths.length === 1 ? "file" : "files"}`);
}

listen<VideoProgress>("video-progress", (event) => {
  if (videoAbortRequested) return;
  const p = event.payload;
  videoMaxPeopleSeen = Math.max(videoMaxPeopleSeen, p.persons);
  setProgress("#video-progress-bar", p.frameIndex + 1, p.totalFrames || p.frameIndex + 1);
  setText("#video-progress-text", `Frame ${p.frameIndex + 1} | ${p.persons} people | ${p.frameMs.toFixed(0)} ms`);
});

listen<VideoMeshPreview>("video-mesh-preview", async (event) => {
  if (videoAbortRequested) return;
  const p = event.payload;
  const people = videoMeshFrameMap.get(p.frameIndex) ?? [];
  const existingIndex = people.findIndex((person) => person.personIndex === p.personIndex);
  if (existingIndex >= 0) {
    people[existingIndex] = p;
  } else {
    people.push(p);
  }
  people.sort((a, b) => a.personIndex - b.personIndex);
  videoMeshFrameMap.set(p.frameIndex, people);
  videoMeshFrames = assignVideoTrackIds([...videoMeshFrameMap.entries()]
    .sort(([a], [b]) => a - b)
    .map(([frameIndex, framePeople]) => ({ frameIndex, people: framePeople })));
  motionCursor = Math.max(0, videoMeshFrames.findIndex((frame) => frame.frameIndex === p.frameIndex));
  updateMotionUi();
  await showMotionFrame(motionCursor, videoMeshFrames.length > 1);
});

listen("tauri://drag-enter", () => {
  setDropzoneActive(activeMode, true);
});

listen("tauri://drag-over", () => {
  setDropzoneActive(activeMode, true);
});

listen("tauri://drag-leave", clearDropzoneActive);

listen<{ paths?: string[] }>("tauri://drag-drop", (event) => {
  clearDropzoneActive();
  const path = event.payload.paths?.[0] ? cleanPath(event.payload.paths[0]) : "";
  if (!path) return;
  const kind = mediaKindFromPath(path);
  if (kind) {
    applyMediaPath(kind, path);
  }
});

el("#image-tab").addEventListener("click", () => activateMode("image"));
el("#video-tab").addEventListener("click", () => activateMode("video"));
el("#load-engine").addEventListener("click", loadEngine);
el("#unload-engine").addEventListener("click", unloadEngine);
el("#refresh-models").addEventListener("click", refreshModelPaths);
el("#browse-library").addEventListener("click", browseLibraryPath);
el("#browse-onnx-dir").addEventListener("click", () => browseModelPath("onnxDir"));
el("#browse-gguf").addEventListener("click", () => browseModelPath("gguf"));
el("#browse-yolo").addEventListener("click", () => browseModelPath("yolo"));
el("#browse-backbone").addEventListener("click", () => browseModelPath("backbone"));
el("#browse-bvh-output").addEventListener("click", chooseBvhOutput);
el("#browse-bvh-template").addEventListener("click", () => browseModelPath("bvhTemplate"));
el("#browse-lbs").addEventListener("click", () => browseModelPath("lbs"));
el("#browse-image").addEventListener("click", () => browseMedia("image"));
el("#browse-video").addEventListener("click", () => browseMedia("video"));
el("#process-image").addEventListener("click", processImage);
el("#export-obj").addEventListener("click", exportObj);
el("#export-image-bvh").addEventListener("click", exportImageBvh);
el("#process-video").addEventListener("click", processVideo);
el("#extract-motion").addEventListener("click", extractMotion);
el("#abort-video").addEventListener("click", abortVideoExport);
el("#copy-output").addEventListener("click", copyOutput);
el("#download-output").addEventListener("click", downloadOutput);
el("#wireframe-toggle").addEventListener("change", updateWireframe);
el("#grid-toggle").addEventListener("change", updateGridVisibility);
el("#grid-size").addEventListener("input", updateGridSize);
el("#mesh-skeleton-toggle").addEventListener("change", updateMeshSkeletonVisibility);
el("#mesh-body-toggle").addEventListener("change", updateMeshBodyVisibility);
el("#image-export-format").addEventListener("change", updateImageExportFormatUi);
el("#image-bbox-toggle").addEventListener("change", drawCurrentImagePreview);
el("#image-landmarks-toggle").addEventListener("change", drawCurrentImagePreview);
el("#image-skeleton-toggle").addEventListener("change", drawCurrentImagePreview);
el("#video-overlay-toggle").addEventListener("change", updateVideoOverlayVisibility);
el<HTMLInputElement>("#video-path").addEventListener("change", handleVideoPathEdited);
el<HTMLInputElement>("#video-path").addEventListener("input", handleVideoPathEdited);
videoPreviewEl.addEventListener("play", (event) => {
  if (event.isTrusted && motionPlaying) stopMotionPlayback();
  syncMotionToVideoPlayback();
  updateMotionUi();
});
videoPreviewEl.addEventListener("timeupdate", syncMotionToVideoPlayback);
videoPreviewEl.addEventListener("seeked", syncMotionToVideoPlayback);
videoPreviewEl.addEventListener("pause", updateMotionUi);
videoPreviewEl.addEventListener("ended", updateMotionUi);
videoPreviewEl.addEventListener("loadedmetadata", () => {
  const frame = videoMeshFrames[motionCursor];
  if (frame) void renderVideoBodyOverlay(frame);
});
for (const id of userPathFields) {
  el<HTMLInputElement>(id).addEventListener("change", () => saveUserPath(id));
}
el("#reset-view").addEventListener("click", fitMeshCamera);
el("#motion-play").addEventListener("click", toggleMotionPlayback);
el("#motion-loop").addEventListener("click", () => {
  motionLoop = !motionLoop;
  updateMotionUi();
});
el("#motion-seek").addEventListener("pointerdown", (event) => {
  stopMotionPlayback();
  const target = event.currentTarget as HTMLInputElement;
  target.setPointerCapture(event.pointerId);
  seekMotionFromPointer(event);
});
el("#motion-seek").addEventListener("pointermove", (event) => {
  if (event.buttons === 1) seekMotionFromPointer(event);
});
el("#motion-seek").addEventListener("click", (event) => seekMotionFromPointer(event));
el("#motion-seek").addEventListener("input", (event) => {
  stopMotionPlayback();
  void seekMotionTo(Number.parseInt((event.target as HTMLInputElement).value, 10) || 0);
});
el("#motion-seek").addEventListener("change", (event) => {
  void seekMotionTo(Number.parseInt((event.target as HTMLInputElement).value, 10) || 0);
});
el("#motion-speed").addEventListener("change", () => {
  if (motionPlaying) {
    if (motionTimer) window.clearTimeout(motionTimer);
    scheduleMotionPlayback();
  }
});

bindDropzone("#image-dropzone", "image");
bindDropzone("#video-dropzone", "video");
bindExternalLinks();
document.addEventListener("contextmenu", (event) => event.preventDefault());

initSettings();
fillDefaultPaths();
initMeshViewer();
initCollapsibles();
initHardwarePollRate();
bindSteppers();
bindMediaControls();
updateToolbarForMode();
updateImageExportFormatUi();
updateGridSize();
updateMotionUi();
pollHardware();
window.addEventListener("resize", () => {
  drawHardwareGraph();
  drawCurrentImagePreview();
  resizeVideoBodyOverlay();
  const frame = videoMeshFrames[motionCursor];
  if (frame) void renderVideoBodyOverlay(frame);
});
