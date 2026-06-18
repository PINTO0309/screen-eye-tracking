import type { BackendMessage } from "./global";

export function logStatus(payload: Extract<BackendMessage, { type: "status" }>, detail?: unknown): void {
  if (payload.level === "error") {
    console.error(payload.message, detail ?? "");
  } else if (payload.level === "warning") {
    console.warn(payload.message, detail ?? "");
  }
}
