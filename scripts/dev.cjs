const { spawn } = require("child_process");
const http = require("http");
const net = require("net");
const path = require("path");

const repoRoot = path.resolve(__dirname, "..");
const args = process.argv.slice(2);
const binSuffix = process.platform === "win32" ? ".cmd" : "";
const viteBin = path.join(repoRoot, "node_modules", ".bin", `vite${binSuffix}`);
const electronBin = path.join(repoRoot, "node_modules", ".bin", `electron${binSuffix}`);
const devServerUrl = "http://127.0.0.1:5173";
const useShell = process.platform === "win32";

let vite = null;
let electronStarted = false;

function isPortAvailable(port) {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.once("error", () => resolve(false));
    server.once("listening", () => {
      server.close(() => resolve(true));
    });
    server.listen(port, "127.0.0.1");
  });
}

function waitForVite(attempt = 0) {
  const request = http.get(devServerUrl, (response) => {
    response.resume();
    startElectron();
  });
  request.on("error", () => {
    if (attempt > 120) {
      console.error("Vite dev server did not start.");
      vite?.kill();
      process.exit(1);
      return;
    }
    setTimeout(() => waitForVite(attempt + 1), 250);
  });
}

function startElectron() {
  if (electronStarted) {
    return;
  }
  electronStarted = true;
  const electron = spawn(electronBin, [".", ...args], {
    cwd: repoRoot,
    stdio: "inherit",
    shell: useShell,
    env: {
      ...process.env,
      VITE_DEV_SERVER_URL: devServerUrl
    }
  });
  electron.on("error", (error) => {
    console.error(`Failed to start Electron: ${error.message}`);
    vite?.kill();
    process.exit(1);
  });
  electron.on("exit", (code) => {
    vite?.kill();
    process.exit(code || 0);
  });
}

process.on("SIGINT", () => {
  vite?.kill();
  process.exit(130);
});

async function main() {
  const available = await isPortAvailable(5173);
  if (!available) {
    console.error("Port 5173 is already in use. Stop the existing dev server before running `pnpm dev`.");
    process.exit(1);
  }

  vite = spawn(viteBin, ["--host", "127.0.0.1", "--port", "5173", "--strictPort"], {
    cwd: repoRoot,
    stdio: "inherit",
    shell: useShell
  });

  vite.on("error", (error) => {
    console.error(`Failed to start Vite: ${error.message}`);
    process.exit(1);
  });

  vite.on("exit", (code) => {
    if (!electronStarted) {
      console.error(`Vite exited before Electron started: ${code}`);
      process.exit(code || 1);
    }
  });

  waitForVite();
}

main();
