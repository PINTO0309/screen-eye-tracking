const { app, BrowserWindow, ipcMain, screen } = require("electron");
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");
const { startExternalApiServer } = require("./externalApi.cjs");
const { GazeStateStore } = require("./externalStore.cjs");

const repoRoot = path.resolve(__dirname, "..");
const defaultCalibrationFile = path.join(repoRoot, ".gaze_calibration.json");
const liteRtWasmDir = path.resolve(path.dirname(require.resolve("@litertjs/core")), "..", "wasm");
const ortWasmDir = path.dirname(require.resolve("onnxruntime-web/wasm"));
let mainWindow = null;
let backendProcess = null;
let selectedDisplay = null;
let rendererConfig = null;
let rendererReady = false;
let debugOverlay = false;
let shapeOverlay = false;
let externalApi = null;
const pendingMessages = [];
const gazeStateStore = new GazeStateStore({
  readCalibrationFile: (filePath) => {
    try {
      return fs.readFileSync(filePath, "utf8");
    } catch (error) {
      if (error && error.code === "ENOENT") {
        return null;
      }
      throw error;
    }
  }
});

function appendGpuSwitches() {
  app.commandLine.appendSwitch("ignore-gpu-blocklist");
  app.commandLine.appendSwitch("enable-zero-copy");
  app.commandLine.appendSwitch("disable-gpu-sandbox");
  app.commandLine.appendSwitch("enable-unsafe-webgpu");
  app.commandLine.appendSwitch("enable-webgpu-developer-features");
  const features =
    process.platform === "linux"
      ? "Vulkan,WebGPU,WebGPUService"
      : process.platform === "darwin"
        ? "Metal,WebGPU,WebGPUService"
        : "WebGPU,WebGPUService";
  app.commandLine.appendSwitch("enable-features", features);
  app.commandLine.appendSwitch("use-webgpu-adapter", "default");
  app.commandLine.appendSwitch("disable-features", "UseSkiaRenderer,UseChromeOSDirectVideoDecoder");
}

appendGpuSwitches();

function cliArgs() {
  const args = process.argv.slice(2);
  return args.filter((arg) => arg !== ".");
}

function readOption(args, name, fallback) {
  const index = args.indexOf(name);
  if (index >= 0 && index + 1 < args.length) {
    return args[index + 1];
  }
  const prefix = `${name}=`;
  const match = args.find((arg) => arg.startsWith(prefix));
  return match ? match.slice(prefix.length) : fallback;
}

function hasFlag(args, name) {
  return args.includes(name);
}

function readNumberOption(args, name, fallback) {
  const value = Number.parseFloat(readOption(args, name, String(fallback)));
  return Number.isFinite(value) ? value : fallback;
}

function readPositiveNumberOption(args, name, fallback) {
  const value = readNumberOption(args, name, fallback);
  return value > 0 ? value : fallback;
}

function readPortOption(args, name, fallback) {
  const value = Number.parseInt(readOption(args, name, String(fallback)), 10);
  return value > 0 && value <= 65535 ? value : fallback;
}

function readCameraFovOption(args) {
  const value = readNumberOption(args, "--camera-fov", 90);
  return value > 0 && value < 180 ? value : 90;
}

const defaultCameraResolution = { name: "VGA", width: 640, height: 480 };
const rejectedCameraResolutionNames = new Set(["2mp", "4mp", "8mp"]);
const cameraResolutionPresets = new Map();
for (const [name, width, height, aliases] of [
  ["QQVGA", 160, 120, ["QQVGA"]],
  ["QVGA", 320, 240, ["QVGA"]],
  ["VGA", 640, 480, ["VGA"]],
  ["SVGA", 800, 600, ["SVGA"]],
  ["XGA", 1024, 768, ["XGA"]],
  ["HD", 1280, 720, ["HD", "720p"]],
  ["SXGA", 1280, 1024, ["SXGA"]],
  ["UXGA", 1600, 1200, ["UXGA"]],
  ["Full HD", 1920, 1080, ["Full HD", "1080p"]],
  ["3MP", 2048, 1536, ["3MP"]],
  ["QHD", 2560, 1440, ["QHD", "WQHD", "1440p"]],
  ["5MP", 2592, 1944, ["5MP"]],
  ["6MP", 3072, 2048, ["6MP"]],
  ["4K UHD", 3840, 2160, ["4K UHD"]],
  ["DCI 4K", 4096, 2160, ["DCI 4K"]],
  ["12MP", 4000, 3000, ["12MP"]],
  ["5K", 5120, 2880, ["5K"]],
  ["6K", 6144, 3456, ["6K"]],
  ["8K UHD", 7680, 4320, ["8K UHD"]],
  ["12K", 12288, 6480, ["12K"]]
]) {
  const resolution = { name, width, height };
  for (const alias of aliases) {
    cameraResolutionPresets.set(normalizeCameraResolutionName(alias), resolution);
  }
}

function normalizeCameraResolutionName(value) {
  return Array.from(value.toLowerCase()).filter((ch) => /[a-z0-9]/.test(ch)).join("");
}

function parseCameraResolution(value) {
  const raw = String(value).trim();
  const sizeMatch = raw.match(/^(\d+)\s*[xX×]\s*(\d+)$/);
  if (sizeMatch) {
    const width = Number.parseInt(sizeMatch[1], 10);
    const height = Number.parseInt(sizeMatch[2], 10);
    if (width <= 0 || height <= 0) {
      throw new Error("camera resolution width and height must be positive integers");
    }
    return { width, height };
  }
  const normalized = normalizeCameraResolutionName(raw);
  if (rejectedCameraResolutionNames.has(normalized)) {
    throw new Error(`camera resolution alias is not accepted; use WIDTHxHEIGHT instead: ${value}`);
  }
  const preset = cameraResolutionPresets.get(normalized);
  if (!preset) {
    throw new Error(`unknown camera resolution: ${value}`);
  }
  return preset;
}

function readCameraResolutionOption(args) {
  return parseCameraResolution(readOption(args, "--camera-resolution", defaultCameraResolution.name));
}

function resolveRuntime(args) {
  const runtime = readOption(args, "--runtime", "python");
  return ["python", "onnxweb", "litert"].includes(runtime) ? runtime : "python";
}

function resolveRepoPath(value) {
  const raw = value || "";
  return path.resolve(repoRoot, raw);
}

function useDevServer() {
  return Boolean(process.env.VITE_DEV_SERVER_URL) || (!app.isPackaged && process.env.NODE_ENV !== "production");
}

function rendererStaticDir() {
  return useDevServer() ? path.join(repoRoot, "public") : path.join(repoRoot, "dist");
}

function rendererModelDir() {
  return path.join(rendererStaticDir(), "models");
}

function copyFileForRenderer(sourcePath, relativeDir) {
  const source = path.resolve(sourcePath);
  const targetDir = path.join(rendererStaticDir(), relativeDir);
  const target = path.join(targetDir, path.basename(source));
  try {
    fs.mkdirSync(targetDir, { recursive: true });
    if (source !== target) {
      fs.copyFileSync(source, target);
    }
  } catch (error) {
    console.error(`Failed to copy renderer asset ${source} -> ${target}: ${error.message}`);
  }
  const prefix = useDevServer() ? "/" : "";
  return `${prefix}${relativeDir}/${encodeURIComponent(path.basename(source))}`;
}

function copyModelForRenderer(modelPath) {
  return copyFileForRenderer(modelPath, "models");
}

function copyRuntimeAssetsForRenderer(sourceDir, relativeDir, predicate) {
  const names = fs.readdirSync(sourceDir).filter(predicate).sort();
  for (const name of names) {
    copyFileForRenderer(path.join(sourceDir, name), relativeDir);
  }
  return `${useDevServer() ? "/" : ""}${relativeDir}/`;
}

function copyOnnxRuntimeAssetsForRenderer() {
  return copyRuntimeAssetsForRenderer(
    ortWasmDir,
    "runtime-assets/onnxruntime-web",
    (name) => name.startsWith("ort-wasm-simd-threaded.") && (name.endsWith(".wasm") || name.endsWith(".mjs"))
  );
}

function copyLiteRtRuntimeAssetsForRenderer() {
  return copyRuntimeAssetsForRenderer(
    liteRtWasmDir,
    "runtime-assets/litert",
    (name) => name.startsWith("litert_wasm_") && (name.endsWith(".wasm") || name.endsWith(".js"))
  );
}

function backendArgs(args, displayBounds) {
  const passthrough = [];
  const options = [
    "--detector",
    "--backend",
    "--camera",
    "--camera-resolution",
    "--camera-fov",
    "--score-threshold",
    "--display-size-inch",
    "--calibration-file",
    "--detector-model",
    "--retinaface-model",
    "--deim-model",
    "--gaze-model",
    "--smoothing-alpha",
    "--smoothing-alpha-y",
    "--preview-fps",
    "--camera-screen-x",
    "--camera-screen-y",
    "--eye-position-weight-x",
    "--eye-position-weight-y",
    "--retinaface-head-face-ratio",
    "--gaze-projection-mode"
  ];
  for (const option of options) {
    const value = readOption(args, option, null);
    if (value !== null) {
      passthrough.push(option, value);
    }
  }
  if (hasFlag(args, "--hide-preview")) {
    passthrough.push("--hide-preview");
  }
  if (hasFlag(args, "--no-flip-x")) {
    passthrough.push("--no-flip-x");
  }
  if (hasFlag(args, "--no-flip-y")) {
    passthrough.push("--no-flip-y");
  }
  passthrough.push("--display-width", String(displayBounds.width));
  passthrough.push("--display-height", String(displayBounds.height));
  return passthrough;
}

function webInferenceConfig(args, displayBounds, runtime) {
  const detector = readOption(args, "--detector", "retinaface");
  const cameraResolution = readCameraResolutionOption(args);
  const calibrationFile = resolveRepoPath(readOption(args, "--calibration-file", defaultCalibrationFile));
  const retinafaceModel = resolveRepoPath(
    readOption(
      args,
      "--detector-model",
      readOption(
        args,
        "--retinaface-model",
        runtime === "litert"
          ? "public/models/retinaface_mbn025_wo_postprocess_480x640_float32.tflite"
          : "public/models/retinaface_mbn025_with_postprocess_480x640_max1000_th0.70.onnx"
      )
    )
  );
  const gazeModel = resolveRepoPath(
    readOption(
      args,
      "--gaze-model",
      runtime === "litert" ? "public/models/gaze_1x3x160x160_float32.tflite" : "public/models/gaze_1x3x160x160.onnx"
    )
  );
  return {
    runtime,
    detector,
    backend: readOption(args, "--backend", "tensorrt"),
    camera: readOption(args, "--camera", "0"),
    cameraResolutionName: cameraResolution.name,
    cameraWidth: cameraResolution.width,
    cameraHeight: cameraResolution.height,
    cameraFov: readCameraFovOption(args),
    scoreThreshold: readNumberOption(args, "--score-threshold", 0.5),
    displaySizeInch: readPositiveNumberOption(args, "--display-size-inch", 31.5),
    displayWidth: displayBounds.width,
    displayHeight: displayBounds.height,
    calibrationFile,
    retinafaceModel,
    gazeModel,
    retinafaceModelUrl: copyModelForRenderer(retinafaceModel),
    gazeModelUrl: copyModelForRenderer(gazeModel),
    onnxWasmBaseUrl: copyOnnxRuntimeAssetsForRenderer(),
    liteRtWasmBaseUrl: copyLiteRtRuntimeAssetsForRenderer(),
    smoothingAlpha: readNumberOption(args, "--smoothing-alpha", 0.65),
    smoothingAlphaY: readNumberOption(args, "--smoothing-alpha-y", 0.45),
    previewFps: readNumberOption(args, "--preview-fps", 8.0),
    hidePreview: hasFlag(args, "--hide-preview"),
    flipX: !hasFlag(args, "--no-flip-x"),
    flipY: !hasFlag(args, "--no-flip-y"),
    cameraScreenX: readNumberOption(args, "--camera-screen-x", 0.5),
    cameraScreenY: readNumberOption(args, "--camera-screen-y", 0),
    eyePositionWeightX: readNumberOption(args, "--eye-position-weight-x", 1),
    eyePositionWeightY: readNumberOption(args, "--eye-position-weight-y", 0.25),
    retinafaceHeadFaceRatio: readNumberOption(args, "--retinaface-head-face-ratio", 1.545),
    gazeProjectionMode: readOption(args, "--gaze-projection-mode", "legacy")
  };
}

function externalStoreInitOptions(args) {
  const cameraResolution = readCameraResolutionOption(args);
  return {
    backend: readOption(args, "--backend", "tensorrt"),
    detector: readOption(args, "--detector", "retinaface"),
    camera: readOption(args, "--camera", "0"),
    cameraResolutionName: cameraResolution.name,
    cameraWidth: cameraResolution.width,
    cameraHeight: cameraResolution.height,
    cameraFov: readCameraFovOption(args),
    displaySizeInch: readPositiveNumberOption(args, "--display-size-inch", 31.5),
    calibrationFile: resolveRepoPath(readOption(args, "--calibration-file", defaultCalibrationFile)),
    eyePositionWeightX: readNumberOption(args, "--eye-position-weight-x", 1),
    eyePositionWeightY: readNumberOption(args, "--eye-position-weight-y", 0.25)
  };
}

function maybeStartExternalApi(args) {
  if (!hasFlag(args, "--external-api")) {
    return;
  }
  const host = readOption(args, "--external-api-host", "127.0.0.1");
  const port = readPortOption(args, "--external-api-port", 47892);
  externalApi = startExternalApiServer({
    store: gazeStateStore,
    host,
    port,
    logger: console
  });
}

function stopExternalApi() {
  if (!externalApi) {
    return;
  }
  externalApi.close();
  externalApi = null;
}

function sendToRenderer(payload) {
  gazeStateStore.updateFromBackendMessage(payload);
  if (payload && payload.type === "status" && (payload.level === "warning" || payload.level === "error")) {
    const log = payload.level === "error" ? console.error : console.warn;
    log(`[renderer-status:${payload.level}] ${payload.message}`);
  }
  if (mainWindow && !mainWindow.isDestroyed() && rendererReady) {
    mainWindow.webContents.send("backend-message", payload);
  } else {
    pendingMessages.push(payload);
  }
}

function shouldSuppressRendererConsole(message) {
  return (
    message.includes("VerifyEachNodeIsAssignedToAnEp") ||
    message.includes("Some nodes were not assigned to the preferred execution providers") ||
    message.includes("Rerunning with verbose output on a non-minimal build will show node assignments")
  );
}

function applyPassiveOverlayMode() {
  if (!mainWindow || mainWindow.isDestroyed() || debugOverlay) {
    return;
  }
  mainWindow.setFocusable(false);
  mainWindow.setIgnoreMouseEvents(true);
}

function sanitizeOverlayRegions(regions) {
  if (!Array.isArray(regions)) {
    return [];
  }
  return regions
    .slice(0, 16)
    .map((region) => ({
      x: Math.max(0, Math.floor(Number(region.x) || 0)),
      y: Math.max(0, Math.floor(Number(region.y) || 0)),
      width: Math.max(0, Math.ceil(Number(region.width) || 0)),
      height: Math.max(0, Math.ceil(Number(region.height) || 0))
    }))
    .filter((region) => region.width > 0 && region.height > 0);
}

function createWindow() {
  const args = cliArgs();
  const runtime = resolveRuntime(args);
  const displays = screen.getAllDisplays().sort((a, b) => a.bounds.x - b.bounds.x || a.bounds.y - b.bounds.y);
  const displayIndexArg = readOption(args, "--display-index", null);
  const primaryDisplay = screen.getPrimaryDisplay();
  const primaryDisplayIndex = Math.max(
    0,
    displays.findIndex((display) => display.id === primaryDisplay.id)
  );
  const requestedDisplayIndex =
    displayIndexArg === null ? primaryDisplayIndex : Number.parseInt(displayIndexArg, 10);
  selectedDisplay = displays[requestedDisplayIndex] || displays[0];
  const invalidDisplay = !displays[requestedDisplayIndex];
  const bounds = selectedDisplay.bounds;
  debugOverlay = hasFlag(args, "--debug-overlay");
  shapeOverlay = hasFlag(args, "--shape-overlay");
  const cameraScreenX = Number.parseFloat(readOption(args, "--camera-screen-x", "0.5"));
  const cameraScreenY = Number.parseFloat(readOption(args, "--camera-screen-y", "0.0"));

  rendererConfig = {
    calibrate: hasFlag(args, "--calibrate"),
    runtime,
    displayIndex: invalidDisplay ? 0 : requestedDisplayIndex,
    requestedDisplayIndex,
    displayCount: displays.length,
    displayBounds: bounds,
    invalidDisplay,
    cameraScreenX: Number.isFinite(cameraScreenX) ? Math.min(1, Math.max(0, cameraScreenX)) : 0.5,
    cameraScreenY: Number.isFinite(cameraScreenY) ? Math.min(1, Math.max(0, cameraScreenY)) : 0,
    webInference: runtime === "python" ? null : webInferenceConfig(args, bounds, runtime)
  };
  gazeStateStore.initialize(rendererConfig, externalStoreInitOptions(args));
  maybeStartExternalApi(args);

  mainWindow = new BrowserWindow({
    x: bounds.x,
    y: bounds.y,
    width: bounds.width,
    height: bounds.height,
    frame: debugOverlay,
    transparent: !debugOverlay,
    backgroundColor: debugOverlay ? "#0f172a" : undefined,
    resizable: false,
    movable: false,
    alwaysOnTop: true,
    skipTaskbar: !debugOverlay,
    focusable: debugOverlay,
    fullscreenable: false,
    hasShadow: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });
  mainWindow.webContents.session.setPermissionRequestHandler((_webContents, permission, callback) => {
    callback(permission === "media");
  });
  mainWindow.setAlwaysOnTop(true, "screen-saver");
  applyPassiveOverlayMode();
  console.log(
    `Overlay display ${rendererConfig.displayIndex + 1}/${displays.length}: x=${bounds.x}, y=${bounds.y}, width=${bounds.width}, height=${bounds.height}`
  );
  console.log(
    displays
      .map(
        (display, index) =>
          `Display ${index}: id=${display.id}, primary=${display.id === primaryDisplay.id}, x=${display.bounds.x}, y=${display.bounds.y}, width=${display.bounds.width}, height=${display.bounds.height}`
      )
      .join("\n")
  );

  mainWindow.webContents.on("did-fail-load", (_event, errorCode, errorDescription, validatedURL) => {
    console.error(`Renderer failed to load ${validatedURL}: ${errorCode} ${errorDescription}`);
    sendToRenderer({
      type: "status",
      level: "error",
      message: `Renderer failed to load: ${errorDescription}`
    });
  });
  mainWindow.webContents.on("did-finish-load", () => {
    console.log("Renderer loaded");
    applyPassiveOverlayMode();
  });
  mainWindow.once("ready-to-show", applyPassiveOverlayMode);
  mainWindow.webContents.on("render-process-gone", (_event, details) => {
    console.error(`Renderer process gone: ${details.reason}`);
  });
  mainWindow.webContents.on("console-message", (_event, levelOrDetails, messageArg, lineArg, sourceIdArg) => {
    const details =
      typeof levelOrDetails === "object" && levelOrDetails !== null
        ? levelOrDetails
        : {
            level: levelOrDetails,
            message: messageArg,
            lineNumber: lineArg,
            sourceId: sourceIdArg
          };
    const level = Number(details.level ?? 0);
    const message = String(details.message ?? "");
    if (shouldSuppressRendererConsole(message)) {
      return;
    }
    const line = details.lineNumber ?? details.line ?? 0;
    const sourceId = details.sourceId ?? "";
    const prefix = sourceId ? `${sourceId}:${line}` : `renderer:${line}`;
    if (level >= 3) {
      console.error(`[renderer-console] ${prefix} ${message}`);
    } else if (level === 2) {
      console.warn(`[renderer-console] ${prefix} ${message}`);
    } else {
      console.log(`[renderer-console] ${prefix} ${message}`);
    }
  });

  if (process.env.VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL);
  } else if (!app.isPackaged && process.env.NODE_ENV !== "production") {
    mainWindow.loadURL("http://127.0.0.1:5173");
  } else {
    mainWindow.loadFile(path.join(repoRoot, "dist", "index.html"));
  }
  if (debugOverlay) {
    mainWindow.webContents.openDevTools({ mode: "detach" });
  }

  if (invalidDisplay) {
    sendToRenderer({
      type: "status",
      level: "warning",
      message: `Display index ${requestedDisplayIndex} is not available; using display 0`
    });
  }

  if (runtime === "python") {
    startBackend(args, bounds);
  } else {
    sendToRenderer({
      type: "status",
      level: "info",
      message: `Using ${runtime} runtime in Electron renderer`,
      runtime
    });
    if (readOption(args, "--backend", null) !== null) {
      sendToRenderer({
        type: "status",
        level: "warning",
        message: `--backend is ignored by ${runtime} runtime`,
        runtime
      });
    }
  }
}

function startBackend(args, displayBounds) {
  const childArgs = ["run", "python", "-m", "screen_eye_tracking.backend", ...backendArgs(args, displayBounds)];
  backendProcess = spawn("uv", childArgs, {
    cwd: repoRoot,
    stdio: ["pipe", "pipe", "pipe"],
    env: process.env
  });

  let stdoutBuffer = "";
  backendProcess.stdout.on("data", (chunk) => {
    stdoutBuffer += chunk.toString("utf8");
    const lines = stdoutBuffer.split(/\r?\n/);
    stdoutBuffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) {
        continue;
      }
      try {
        sendToRenderer(JSON.parse(line));
      } catch (error) {
        sendToRenderer({ type: "status", level: "warning", message: line });
      }
    }
  });

  backendProcess.stderr.on("data", (chunk) => {
    sendToRenderer({ type: "status", level: "warning", message: chunk.toString("utf8").trim() });
  });

  backendProcess.on("exit", (code) => {
    sendToRenderer({ type: "status", level: code === 0 ? "info" : "error", message: `Python backend exited: ${code}` });
  });
}

ipcMain.handle("get-config", () => rendererConfig);

ipcMain.handle("read-calibration", async (_event, filePath) => {
  const resolved = resolveRepoPath(filePath || defaultCalibrationFile);
  try {
    const text = await fs.promises.readFile(resolved, "utf8");
    return { ok: true, text };
  } catch (error) {
    if (error && error.code === "ENOENT") {
      return { ok: true, text: null };
    }
    return { ok: false, error: error.message };
  }
});

ipcMain.handle("write-calibration", async (_event, filePath, payload) => {
  const resolved = resolveRepoPath(filePath || defaultCalibrationFile);
  try {
    await fs.promises.writeFile(resolved, JSON.stringify(payload, null, 2), "utf8");
    gazeStateStore.updateFromBackendMessage({ type: "calibration", status: "saved", path: resolved });
    return { ok: true, path: resolved };
  } catch (error) {
    return { ok: false, error: error.message };
  }
});

ipcMain.on("publish-backend-message", (_event, payload) => {
  gazeStateStore.updateFromBackendMessage(payload);
});

ipcMain.on("renderer-ready", () => {
  rendererReady = true;
  while (pendingMessages.length > 0) {
    sendToRenderer(pendingMessages.shift());
  }
});

ipcMain.on("calibration-capture", (_event, target) => {
  if (!backendProcess || !backendProcess.stdin.writable) {
    sendToRenderer({ type: "calibration", status: "backend_unavailable" });
    return;
  }
  backendProcess.stdin.write(`${JSON.stringify({ type: "calibration_capture", target })}\n`);
});

ipcMain.on("overlay-regions", (_event, regions) => {
  if (!mainWindow || mainWindow.isDestroyed() || debugOverlay || !shapeOverlay) {
    return;
  }
  const rects = sanitizeOverlayRegions(regions);
  if ((process.platform === "linux" || process.platform === "win32") && typeof mainWindow.setShape === "function") {
    try {
      mainWindow.setShape(rects);
    } catch (error) {
      console.error(`Failed to set overlay shape: ${error.message}`);
    }
  }
  applyPassiveOverlayMode();
});

app.whenReady().then(() => {
  createWindow();
});

app.on("window-all-closed", () => {
  stopExternalApi();
  if (backendProcess) {
    backendProcess.kill();
    backendProcess = null;
  }
  app.quit();
});

app.on("before-quit", () => {
  stopExternalApi();
  if (backendProcess) {
    backendProcess.kill();
    backendProcess = null;
  }
});
