const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("gazeBridge", {
  getConfig: () => ipcRenderer.invoke("get-config"),
  readCalibration: (path) => ipcRenderer.invoke("read-calibration", path),
  writeCalibration: (path, payload) => ipcRenderer.invoke("write-calibration", path, payload),
  ready: () => ipcRenderer.send("renderer-ready"),
  setOverlayRegions: (regions) => ipcRenderer.send("overlay-regions", regions),
  onBackendMessage: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("backend-message", listener);
    return () => ipcRenderer.removeListener("backend-message", listener);
  },
  sendCalibrationCapture: (target) => ipcRenderer.send("calibration-capture", target)
});
