import { describe, expect, it } from "vitest";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { createLipClickController, DOUBLE_CLICK_EFFECT_MS, LOST_RELEASE_MS } = require("./lipClickController.cjs");

function createHarness() {
  const calls = [];
  const effects = [];
  let currentNow = 1000;
  const controller = createLipClickController({
    mouse: {
      move: async (point) => calls.push(["move", point]),
      press: async () => calls.push(["press"]),
      release: async () => calls.push(["release"])
    },
    getDisplayBounds: () => ({ x: 10, y: 20, width: 1000, height: 500 }),
    logger: { warn: () => undefined },
    now: () => currentNow,
    emitEffect: (event) => effects.push(event)
  });
  controller.setEnabled(true);
  const gaze = (mouthOpen, extra = {}) => ({
    type: "gaze",
    x_norm: 0.25,
    y_norm: 0.5,
    mouth_detected: true,
    mouth_open: mouthOpen,
    ...extra
  });
  return {
    calls,
    effects,
    controller,
    gaze,
    advance: (ms) => {
      currentNow += ms;
    }
  };
}

describe("lip click controller", () => {
  it("clicks on open and does not drag while open", async () => {
    const { calls, controller, gaze } = createHarness();

    controller.update(gaze(true));
    controller.update(gaze(true, { x_norm: 0.5 }));
    controller.update(gaze(false, { x_norm: 0.75 }));
    await controller.idle();

    expect(calls).toEqual([
      ["move", { x: 260, y: 270 }],
      ["press"],
      ["release"]
    ]);
  });

  it("keeps double-click behavior as two press/release pairs", async () => {
    const { calls, controller, gaze } = createHarness();

    controller.update(gaze(true));
    controller.update(gaze(false));
    controller.update(gaze(true));
    controller.update(gaze(false));
    await controller.idle();

    expect(calls.map((call) => call[0])).toEqual(["move", "press", "release", "move", "press", "release"]);
  });

  it("emits a single-click effect immediately on the open transition", async () => {
    const { controller, effects, gaze } = createHarness();

    controller.update(gaze(true, { x_norm: 0.75 }));
    await controller.idle();

    expect(effects).toEqual([{ type: "lip_click_effect", effect: "single", x: 760, y: 270 }]);
  });

  it("emits a double-click effect immediately on open-close-open", async () => {
    const { controller, effects, gaze, advance } = createHarness();

    controller.update(gaze(true));
    controller.update(gaze(false));
    await controller.idle();
    advance(DOUBLE_CLICK_EFFECT_MS - 1);
    controller.update(gaze(true, { x_norm: 0.75 }));
    await controller.idle();

    expect(effects).toEqual([
      { type: "lip_click_effect", effect: "single", x: 260, y: 270 },
      { type: "lip_click_effect", effect: "double", x: 760, y: 270 }
    ]);
    advance(1);
    expect(effects).toHaveLength(2);
  });

  it("resets the open state after mouth detection is lost for the timeout", async () => {
    const { calls, controller, gaze, advance } = createHarness();

    controller.update(gaze(true));
    controller.update(gaze(false, { mouth_detected: false, mouth_open: false }));
    advance(LOST_RELEASE_MS);
    controller.update(gaze(false, { mouth_detected: false, mouth_open: false }));
    controller.update(gaze(true, { x_norm: 0.75 }));
    await controller.idle();

    expect(calls.map((call) => call[0])).toEqual(["move", "press", "release", "move", "press", "release"]);
    expect(controller.isPressed()).toBe(false);
  });

  it("resets the open state immediately when calibration becomes active", async () => {
    const { calls, controller, gaze } = createHarness();

    controller.update(gaze(true));
    controller.setCalibrationActive(true);
    controller.setCalibrationActive(false);
    controller.update(gaze(true, { x_norm: 0.75 }));
    await controller.idle();

    expect(calls.map((call) => call[0])).toEqual(["move", "press", "release", "move", "press", "release"]);
    expect(controller.isPressed()).toBe(false);
  });
});
