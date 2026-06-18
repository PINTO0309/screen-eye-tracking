const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("gazeBridge", {
  getConfig: () => ipcRenderer.invoke("get-config"),
  ready: () => ipcRenderer.send("renderer-ready"),
  setOverlayRegions: (regions) => ipcRenderer.send("overlay-regions", regions),
  onBackendMessage: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("backend-message", listener);
    return () => ipcRenderer.removeListener("backend-message", listener);
  },
  sendCalibrationCapture: (target) => ipcRenderer.send("calibration-capture", target)
});
