const LOST_RELEASE_MS = 500;
const DOUBLE_CLICK_EFFECT_MS = 500;

function createLipClickController({
  mouse,
  getDisplayBounds,
  logger = console,
  now = () => Date.now(),
  emitEffect = () => undefined
}) {
  let enabled = false;
  let calibrationActive = false;
  let mouthOpenActive = false;
  let invalidSince = null;
  let lastPressAt = null;
  let pending = Promise.resolve();

  function setEnabled(value) {
    enabled = Boolean(value);
    if (!enabled) {
      releaseNow();
    }
  }

  function setCalibrationActive(value) {
    calibrationActive = Boolean(value);
    if (calibrationActive) {
      releaseNow();
    }
  }

  function update(payload) {
    if (!enabled || calibrationActive || !payload || payload.type !== "gaze") {
      if (mouthOpenActive && (calibrationActive || !enabled)) {
        releaseNow();
      }
      return;
    }
    const point = gazePoint(payload);
    if (!point) {
      markInvalid();
      return;
    }
    if (payload.mouth_detected !== true || typeof payload.mouth_open !== "boolean") {
      markInvalid();
      return;
    }
    invalidSince = null;
    if (payload.mouth_open) {
      const shouldClick = !mouthOpenActive;
      if (shouldClick) {
        mouthOpenActive = true;
        registerPressEffect(point);
        enqueue(async () => {
          await mouse.move(point);
          await mouse.press();
          await mouse.release();
        });
      }
    } else if (mouthOpenActive) {
      mouthOpenActive = false;
    }
  }

  function releaseNow() {
    invalidSince = null;
    lastPressAt = null;
    mouthOpenActive = false;
  }

  function isPressed() {
    return false;
  }

  function registerPressEffect(point) {
    const current = now();
    if (lastPressAt !== null && current - lastPressAt <= DOUBLE_CLICK_EFFECT_MS) {
      lastPressAt = null;
      emitEffect({ type: "lip_click_effect", effect: "double", x: point.x, y: point.y });
      return;
    }
    lastPressAt = current;
    emitEffect({ type: "lip_click_effect", effect: "single", x: point.x, y: point.y });
  }

  function gazePoint(payload) {
    const xNorm = Number(payload.x_norm);
    const yNorm = Number(payload.y_norm);
    if (!Number.isFinite(xNorm) || !Number.isFinite(yNorm)) {
      return null;
    }
    const bounds = getDisplayBounds();
    if (!bounds || !Number.isFinite(bounds.x) || !Number.isFinite(bounds.y) || !Number.isFinite(bounds.width) || !Number.isFinite(bounds.height)) {
      return null;
    }
    return {
      x: Math.round(bounds.x + Math.max(0, Math.min(1, xNorm)) * bounds.width),
      y: Math.round(bounds.y + Math.max(0, Math.min(1, yNorm)) * bounds.height)
    };
  }

  function markInvalid() {
    if (!mouthOpenActive) {
      invalidSince = null;
      return;
    }
    const current = now();
    if (invalidSince === null) {
      invalidSince = current;
      return;
    }
    if (current - invalidSince >= LOST_RELEASE_MS) {
      releaseNow();
    }
  }

  function enqueue(action) {
    pending = pending.then(action, action).catch((error) => {
      logger.warn?.(`Lip motion mouse action failed: ${error instanceof Error ? error.message : String(error)}`);
    });
  }

  return {
    setEnabled,
    setCalibrationActive,
    update,
    releaseNow,
    isPressed,
    idle: () => pending
  };
}

module.exports = { createLipClickController, DOUBLE_CLICK_EFFECT_MS, LOST_RELEASE_MS };
