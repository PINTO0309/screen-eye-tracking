import type { WebAccelerator, WebInferenceConfig } from "../global";
import {
  createGazeInput,
  createGazeInputNhwc,
  createRetinaFaceInput,
  createRetinaFaceInputNhwc,
  estimateGazeFromModelOutput,
  type Detection,
  type GazeEstimate,
  type GazeCrop,
  parseRetinaFaceOutput
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
  ort.env.wasm.wasmPaths = config.onnxWasmBaseUrl;
  ort.env.logLevel = "error";
  const providers = [accelerator];
  const detector = await createOnnxSessionQueued(ort, config.retinafaceModelUrl, providers);
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
  const detectorInputName = "input";
  const detectorOutputName = "batchno_classid_score_x1y1x2y2_landms";
  const gazeInputName = "input";
  const gazeOutputName = "output";
  return {
    accelerator,
    detectorProviders: providers,
    gazeProviders: providers,
    async detect(frame) {
      const input = new ort.Tensor("float32", createRetinaFaceInput(frame), [1, 3, 480, 640]);
      const results = await detector.run({ [detectorInputName]: input });
      const output = results[detectorOutputName] ?? results[detector.outputNames[0]];
      return parseRetinaFaceOutput(output.data as Float32Array, config.scoreThreshold);
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

async function createLiteRtModels(config: WebInferenceConfig, accelerator: WebAccelerator): Promise<RuntimeModels> {
  const { Tensor, loadAndCompile, loadLiteRt } = await import("@litertjs/core");
  await ensureLiteRtLoaded(loadLiteRt, config.liteRtWasmBaseUrl);
  const detector = await loadAndCompile(config.retinafaceModelUrl, { accelerator });
  let gaze: Awaited<ReturnType<typeof loadAndCompile>>;
  try {
    gaze = await loadAndCompile(config.gazeModelUrl, { accelerator });
  } catch (error) {
    detector.delete();
    throw error;
  }
  const detectorInputShape = Array.from(detector.getInputDetails()[0].shape);
  const gazeInputShape = Array.from(gaze.getInputDetails()[0].shape);
  validateLiteRtInputShape("RetinaFace", detectorInputShape, [
    [1, 3, 480, 640],
    [1, 480, 640, 3]
  ]);
  validateLiteRtInputShape("Gaze", gazeInputShape, [
    [1, 3, 160, 160],
    [1, 160, 160, 3]
  ]);
  return {
    accelerator,
    detectorProviders: [detector.isFullyAccelerated ? accelerator : "wasm"],
    gazeProviders: [gaze.isFullyAccelerated ? accelerator : "wasm"],
    async detect(frame) {
      const inputData = isShape(detectorInputShape, [1, 480, 640, 3])
        ? createRetinaFaceInputNhwc(frame)
        : createRetinaFaceInput(frame);
      const input = new Tensor(inputData, detectorInputShape);
      const outputs = (await detector.run(input)) as InstanceType<typeof Tensor>[];
      input.delete();
      try {
        const source = outputs[0];
        const output = source.accelerator === "wasm" ? source : await source.moveTo("wasm");
        try {
          return parseRetinaFaceOutput(output.toTypedArray() as Float32Array, config.scoreThreshold);
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
