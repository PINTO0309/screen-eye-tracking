const http = require("http");
const { WebSocketServer } = require("ws");
const { SCHEMA_VERSION } = require("./externalStore.cjs");

const SNAPSHOT_SLICES = new Set(["gaze", "display", "camera", "calibration", "runtime", "models", "status"]);

function sendJson(response, statusCode, payload) {
  const body = JSON.stringify(payload);
  response.writeHead(statusCode, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
    "content-length": Buffer.byteLength(body)
  });
  response.end(body);
}

function createRequestHandler(store) {
  return (request, response) => {
    if (request.method !== "GET") {
      sendJson(response, 405, { error: "method_not_allowed" });
      return;
    }

    const url = new URL(request.url || "/", "http://127.0.0.1");
    if (url.pathname === "/health") {
      const snapshot = store.getSnapshot();
      sendJson(response, 200, {
        ok: true,
        schema_version: SCHEMA_VERSION,
        started_at: snapshot.started_at
      });
      return;
    }

    if (url.pathname === "/snapshot") {
      sendJson(response, 200, store.getSnapshot());
      return;
    }

    const sliceMatch = url.pathname.match(/^\/snapshot\/([a-z_]+)$/u);
    if (sliceMatch && SNAPSHOT_SLICES.has(sliceMatch[1])) {
      sendJson(response, 200, store.getSlice(sliceMatch[1]));
      return;
    }

    sendJson(response, 404, { error: "not_found" });
  };
}

function sendSocketJson(socket, payload) {
  if (socket.readyState === socket.OPEN) {
    socket.send(JSON.stringify(payload));
  }
}

function startExternalApiServer({ store, host, port, logger = console }) {
  const server = http.createServer(createRequestHandler(store));
  const wss = new WebSocketServer({ noServer: true });

  wss.on("connection", (socket) => {
    sendSocketJson(socket, {
      type: "snapshot",
      snapshot: store.getSnapshot()
    });
    const unsubscribe = store.subscribe((event) => sendSocketJson(socket, event));
    socket.on("close", unsubscribe);
    socket.on("error", unsubscribe);
  });

  server.on("upgrade", (request, socket, head) => {
    const url = new URL(request.url || "/", `http://${host}`);
    if (url.pathname !== "/events") {
      socket.destroy();
      return;
    }
    wss.handleUpgrade(request, socket, head, (ws) => {
      wss.emit("connection", ws, request);
    });
  });

  server.on("error", (error) => {
    logger.error(`External API failed: ${error.message}`);
  });

  server.listen(port, host, () => {
    const address = server.address();
    const actualPort = typeof address === "object" && address ? address.port : port;
    logger.log(`External API listening on http://${host}:${actualPort}`);
  });

  return {
    server,
    wss,
    close: () => {
      wss.close();
      server.close();
    }
  };
}

module.exports = {
  SNAPSHOT_SLICES,
  createRequestHandler,
  startExternalApiServer
};
