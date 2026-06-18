const { spawnSync } = require("child_process");
const path = require("path");

const repoRoot = path.resolve(__dirname, "..");
const args = process.argv.slice(2);
const binSuffix = process.platform === "win32" ? ".cmd" : "";
const electronBin = path.join(repoRoot, "node_modules", ".bin", `electron${binSuffix}`);

const build = spawnSync("pnpm", ["run", "build"], {
  cwd: repoRoot,
  stdio: "inherit",
  shell: process.platform === "win32"
});

if (build.status !== 0) {
  process.exit(build.status || 1);
}

const electron = spawnSync(electronBin, [".", ...args], {
  cwd: repoRoot,
  stdio: "inherit",
  env: {
    ...process.env,
    NODE_ENV: "production"
  },
  shell: process.platform === "win32"
});

process.exit(electron.status || 0);
