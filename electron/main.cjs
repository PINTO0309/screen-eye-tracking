const { app, BrowserWindow, ipcMain, screen } = require("electron");
const { spawn } = require("child_process");
const path = require("path");

const repoRoot = path.resolve(__dirname, "..");
let mainWindow = null;
let backendProcess = null;
let selectedDisplay = null;
let rendererConfig = null;
let rendererReady = false;
let debugOverlay = false;
let shapeOverlay = false;
const pendingMessages = [];

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

function backendArgs(args, displayBounds) {
  const passthrough = [];
  const options = [
    "--detector",
    "--backend",
    "--camera",
    "--score-threshold",
    "--display-size-inch",
    "--calibration-file",
    "--detector-model",
    "--retinaface-model",
    "--deim-model",
    "--gaze-model",
    "--smoothing-alpha",
    "--preview-fps",
    "--camera-screen-x",
    "--camera-screen-y",
    "--eye-position-weight-x",
    "--eye-position-weight-y",
    "--retinaface-head-face-ratio"
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

function sendToRenderer(payload) {
  if (mainWindow && !mainWindow.isDestroyed() && rendererReady) {
    mainWindow.webContents.send("backend-message", payload);
  } else {
    pendingMessages.push(payload);
  }
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
    displayIndex: invalidDisplay ? 0 : requestedDisplayIndex,
    requestedDisplayIndex,
    displayCount: displays.length,
    displayBounds: bounds,
    invalidDisplay,
    cameraScreenX: Number.isFinite(cameraScreenX) ? Math.min(1, Math.max(0, cameraScreenX)) : 0.5,
    cameraScreenY: Number.isFinite(cameraScreenY) ? Math.min(1, Math.max(0, cameraScreenY)) : 0
  };

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

  startBackend(args, bounds);
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

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  if (backendProcess) {
    backendProcess.kill();
    backendProcess = null;
  }
  app.quit();
});

app.on("before-quit", () => {
  if (backendProcess) {
    backendProcess.kill();
    backendProcess = null;
  }
});
