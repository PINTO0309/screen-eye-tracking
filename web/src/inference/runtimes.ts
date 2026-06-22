import type { WebAccelerator, WebInferenceConfig } from "../global";
import {
  createGazeInput,
  createGazeInputNhwc,
  createRetinaFaceInput,
  createRetinaFaceInputNhwc,
  createYoloInput,
  createYoloInputNhwc,
  estimateGazeFromModelOutput,
  type Detection,
  type GazeEstimate,
  type GazeCrop,
  parseRetinaFaceOutput,
  parseRetinaFaceRawOutput,
  parseYoloOutput,
  YOLO_INPUT_HEIGHT,
  YOLO_INPUT_WIDTH
} from "./core";

export interface RuntimeModels {
  readonly accelerator: WebAccelerator;
  readonly detectorProviders: string[];
  readonly gazeProviders: string[];
  detect(frame: ImageData): Promise<{ head: Detection | null; eyes: Detection[] }>;
  estimate(frame: ImageData, head: Detection, eyes: Detection[]): Promise<GazeEstimate>;
  dispose(): void;
}

let onnxSessionCreateQueue = Promise.resolve();
let liteRtLoadPromise: Promise<unknown> | null = null;

export async function createRuntimeModels(config: WebInferenceConfig, accelerator: WebAccelerator): Promise<RuntimeModels> {
  if (config.runtime === "onnxweb") {
    return createOnnxRuntimeModels(config, accelerator);
  }
  return createLiteRtModels(config, accelerator);
}

async function createOnnxRuntimeModels(config: WebInferenceConfig, accelerator: WebAccelerator): Promise<RuntimeModels> {
  const ort = await import("onnxruntime-web/webgpu");
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = absoluteAssetUrl(config.onnxWasmBaseUrl);
  ort.env.logLevel = "error";
  const providers = [accelerator];
  const detector = await createOnnxSessionQueued(ort, config.detectorModelUrl, providers);
  let gaze: Awaited<ReturnType<typeof ort.InferenceSession.create>>;
  try {
    if (accelerator === "webgpu") {
      await delay(50);
    }
    gaze = await createOnnxSessionQueued(ort, config.gazeModelUrl, providers);
  } catch (error) {
    void detector.release();
    throw error;
  }
  const detectorInputName = config.detector === "yolo" ? detector.inputNames[0] ?? "images" : "input";
  const detectorOutputName = config.detector === "yolo" ? detector.outputNames[0] ?? "output0" : "batchno_classid_score_x1y1x2y2_landms";
  const gazeInputName = "input";
  const gazeOutputName = "output";
  return {
    accelerator,
    detectorProviders: providers,
    gazeProviders: providers,
    async detect(frame) {
      const input =
        config.detector === "yolo"
          ? new ort.Tensor("float32", createYoloInput(frame), [1, 3, YOLO_INPUT_HEIGHT, YOLO_INPUT_WIDTH])
          : new ort.Tensor("float32", createRetinaFaceInput(frame), [1, 3, 480, 640]);
      const results = await detector.run({ [detectorInputName]: input });
      const output = results[detectorOutputName] ?? results[detector.outputNames[0]];
      if (config.detector === "yolo") {
        return parseYoloOutput(output.data as Float32Array, config.scoreThreshold, config.cameraWidth, config.cameraHeight);
      }
      return parseRetinaFaceOutput(output.data as Float32Array, config.scoreThreshold, config.cameraWidth, config.cameraHeight);
    },
    async estimate(frame, head, eyes) {
      const crop = createGazeInput(frame, head, eyes);
      const input = new ort.Tensor("float32", crop.input, [1, 3, 160, 160]);
      const results = await gaze.run({ [gazeInputName]: input });
      const output = results[gazeOutputName] ?? results[gaze.outputNames[0]];
      return estimateFromOutput(output.data as Float32Array, crop);
    },
    dispose() {
      void detector.release();
      void gaze.release();
    }
  };
}

function createOnnxSessionQueued(
  ort: typeof import("onnxruntime-web/webgpu"),
  modelUrl: string,
  providers: string[]
): Promise<Awaited<ReturnType<typeof ort.InferenceSession.create>>> {
  const create = onnxSessionCreateQueue.then(() =>
    ort.InferenceSession.create(modelUrl, { executionProviders: providers, logSeverityLevel: 3 })
  );
  onnxSessionCreateQueue = create.then(
    () => undefined,
    () => undefined
  );
  return create;
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function absoluteAssetUrl(url: string): string {
  return new URL(url, window.location.href).toString();
}

async function createLiteRtModels(config: WebInferenceConfig, accelerator: WebAccelerator): Promise<RuntimeModels> {
  const { Tensor, loadAndCompile, loadLiteRt } = await import("@litertjs/core");
  await ensureLiteRtLoaded(loadLiteRt, absoluteAssetUrl(config.liteRtWasmBaseUrl));
  const detector = await loadAndCompile(config.detectorModelUrl, { accelerator });
  let gaze: Awaited<ReturnType<typeof loadAndCompile>>;
  try {
    gaze = await loadAndCompile(config.gazeModelUrl, { accelerator });
  } catch (error) {
    detector.delete();
    throw error;
  }
  const detectorInputShape = Array.from(detector.getInputDetails()[0].shape);
  const yoloOutputIndex = config.detector === "yolo" ? yoloLiteRtOutputIndex(detector.getOutputDetails()) : -1;
  const retinaFaceOutputs =
    config.detector === "retinaface"
      ? {
          loc: liteRtOutputIndex(detector.getOutputDetails(), "loc", [1, 12600, 4]),
          confLogits: liteRtOutputIndex(detector.getOutputDetails(), "conf_logits", [1, 12600, 2]),
          landms: liteRtOutputIndex(detector.getOutputDetails(), "landms", [1, 12600, 10])
        }
      : null;
  const gazeInputShape = Array.from(gaze.getInputDetails()[0].shape);
  validateLiteRtInputShape(
    config.detector === "yolo" ? "YOLO" : "RetinaFace",
    detectorInputShape,
    config.detector === "yolo"
      ? [[1, YOLO_INPUT_HEIGHT, YOLO_INPUT_WIDTH, 3]]
      : [
          [1, 3, 480, 640],
          [1, 480, 640, 3]
        ]
  );
  validateLiteRtInputShape("Gaze", gazeInputShape, [
    [1, 3, 160, 160],
    [1, 160, 160, 3]
  ]);
  return {
    accelerator,
    detectorProviders: [detector.isFullyAccelerated ? accelerator : "wasm"],
    gazeProviders: [gaze.isFullyAccelerated ? accelerator : "wasm"],
    async detect(frame) {
      const inputData =
        config.detector === "yolo"
          ? createYoloInputNhwc(frame)
          : isShape(detectorInputShape, [1, 480, 640, 3])
            ? createRetinaFaceInputNhwc(frame)
            : createRetinaFaceInput(frame);
      const input = new Tensor(inputData, detectorInputShape);
      const outputs = (await detector.run(input)) as InstanceType<typeof Tensor>[];
      input.delete();
      try {
        if (config.detector === "yolo") {
          const output0 = await liteRtTensorData(outputs[yoloOutputIndex]);
          try {
            return parseYoloOutput(output0.data, config.scoreThreshold, config.cameraWidth, config.cameraHeight);
          } finally {
            output0.delete();
          }
        }
        if (retinaFaceOutputs === null) {
          throw new Error(`Unsupported LiteRT detector: ${config.detector}`);
        }
        const loc = await liteRtTensorData(outputs[retinaFaceOutputs.loc]);
        const confLogits = await liteRtTensorData(outputs[retinaFaceOutputs.confLogits]);
        const landms = await liteRtTensorData(outputs[retinaFaceOutputs.landms]);
        try {
          return parseRetinaFaceRawOutput(
            loc.data,
            confLogits.data,
            landms.data,
            config.scoreThreshold,
            config.cameraWidth,
            config.cameraHeight
          );
        } finally {
          loc.delete();
          confLogits.delete();
          landms.delete();
        }
      } finally {
        for (const tensor of outputs) {
          tensor.delete();
        }
      }
    },
    async estimate(frame, head, eyes) {
      const crop = isShape(gazeInputShape, [1, 160, 160, 3])
        ? createGazeInputNhwc(frame, head, eyes)
        : createGazeInput(frame, head, eyes);
      const input = new Tensor(crop.input, gazeInputShape);
      const outputs = (await gaze.run(input)) as InstanceType<typeof Tensor>[];
      input.delete();
      try {
        const source = outputs[0];
        const output = source.accelerator === "wasm" ? source : await source.moveTo("wasm");
        try {
          return estimateFromOutput(output.toTypedArray() as Float32Array, crop);
        } finally {
          if (output !== source) {
            output.delete();
          }
        }
      } finally {
        for (const tensor of outputs) {
          tensor.delete();
        }
      }
    },
    dispose() {
      detector.delete();
      gaze.delete();
    }
  };
}

function liteRtOutputIndex(
  outputs: readonly { readonly name: string; readonly index: number; readonly shape: Int32Array }[],
  name: string,
  expectedShape: readonly number[]
): number {
  const detail = outputs.find((output) => output.name === name);
  if (!detail) {
    throw new Error(`RetinaFace LiteRT output ${name} was not found`);
  }
  const shape = Array.from(detail.shape);
  if (!isShape(shape, expectedShape)) {
    throw new Error(
      `RetinaFace LiteRT output ${name} shape ${JSON.stringify(shape)} is not supported; expected ${JSON.stringify(
        expectedShape
      )}`
    );
  }
  return detail.index;
}

function yoloLiteRtOutputIndex(outputs: readonly { readonly name: string; readonly index: number; readonly shape: Int32Array }[]): number {
  const detail =
    outputs.find((output) => output.name === "output0") ??
    outputs.find((output) => isShape(Array.from(output.shape), [1, 4 + 28, 6300]));
  if (!detail) {
    throw new Error("YOLO LiteRT output0 was not found");
  }
  const shape = Array.from(detail.shape);
  if (!isShape(shape, [1, 4 + 28, 6300])) {
    throw new Error(`YOLO LiteRT output0 shape ${JSON.stringify(shape)} is not supported; expected [1,32,6300]`);
  }
  return detail.index;
}

async function liteRtTensorData<T extends { accelerator: string; moveTo(accelerator: "wasm"): Promise<T>; toTypedArray(): ArrayLike<number>; delete(): void }>(
  source: T
): Promise<{ data: ArrayLike<number>; delete(): void }> {
  const output = source.accelerator === "wasm" ? source : await source.moveTo("wasm");
  return {
    data: output.toTypedArray(),
    delete() {
      if (output !== source) {
        output.delete();
      }
    }
  };
}

function isShape(actual: readonly number[], expected: readonly number[]): boolean {
  return actual.length === expected.length && actual.every((value, index) => value === expected[index]);
}

function validateLiteRtInputShape(modelName: string, actual: readonly number[], expectedShapes: readonly number[][]): void {
  if (!expectedShapes.some((expected) => isShape(actual, expected))) {
    throw new Error(
      `${modelName} LiteRT input shape ${JSON.stringify(actual)} is not supported; expected one of ${expectedShapes
        .map((shape) => JSON.stringify(shape))
        .join(", ")}`
    );
  }
}

function ensureLiteRtLoaded(
  loadLiteRt: (path: string, options?: { threads?: boolean }) => Promise<unknown>,
  wasmBaseUrl: string
): Promise<unknown> {
  if (liteRtLoadPromise === null) {
    liteRtLoadPromise = loadLiteRt(wasmBaseUrl, { threads: false }).catch((error) => {
      liteRtLoadPromise = null;
      throw error;
    });
  }
  return liteRtLoadPromise;
}

function estimateFromOutput(output: Float32Array, crop: GazeCrop): GazeEstimate {
  return estimateGazeFromModelOutput(output, crop);
}
