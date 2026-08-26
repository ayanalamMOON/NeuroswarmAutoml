"""
NeuroSwarm-AutoML Async Inference Server & Gradio Web Interface.

Serves exported TorchScript (.pt) and ONNX (.onnx) model binaries via a low-latency
FastAPI REST engine with lifespan context management and an embedded Gradio web app.
"""

import argparse
from contextlib import asynccontextmanager
import io
import logging
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from pydantic import BaseModel
import uvicorn

# Dynamically link cuDNN 9 & cuBLAS 12 DLLs on Windows before importing onnxruntime
if os.name == "nt":
    venv_base = sys.prefix
    dll_paths = [
        os.path.join(venv_base, "Lib", "site-packages", "nvidia", "cudnn", "lib"),
        os.path.join(venv_base, "Lib", "site-packages", "nvidia", "cublas", "lib"),
        os.path.join(venv_base, "Lib", "site-packages", "torch", "lib"),
    ]
    for p in dll_paths:
        if os.path.exists(p):
            try:
                os.add_dll_directory(p)
            except AttributeError:
                pass
            os.environ["PATH"] = p + os.path.pathsep + os.environ.get("PATH", "")

try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:
    HAS_ORT = False

import gradio as gr

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("neuroswarm.server")

# Default Class Labels
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class HealthResponse(BaseModel):
    status: str
    model_path: str
    model_format: str
    device: str
    num_classes: int


class ClassificationResponse(BaseModel):
    top_prediction: str
    confidence: float
    probabilities: Dict[str, float]


class ModelInferenceEngine:
    """Unified inference container supporting TorchScript and ONNX Runtime execution."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        class_names: Optional[List[str]] = None,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at: {self.model_path.resolve()}"
            )

        self.class_names = class_names or CIFAR10_CLASSES
        self.num_classes = len(self.class_names)
        self.device_str = (
            "cuda" if device.lower() == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.model_format = self.model_path.suffix.lower()

        # Input Normalization Pipeline (32x32 ImageNet/CIFAR RGB standard)
        self.transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        self._load_model()

    def _load_model(self) -> None:
        """Loads target model into memory with active CUDA/CPU execution provider."""
        if self.model_format == ".pt":
            logger.info(
                f"Loading TorchScript model from {self.model_path} onto {self.device_str.upper()}..."
            )
            self.model = torch.jit.load(
                str(self.model_path), map_location=self.device_str
            )
            self.model.eval()
            logger.info("TorchScript model loaded successfully.")

        elif self.model_format == ".onnx":
            if not HAS_ORT:
                raise RuntimeError("onnxruntime is required to serve .onnx models.")
            logger.info(f"Loading ONNX model from {self.model_path}...")

            available_providers = ort.get_available_providers()
            if (
                self.device_str == "cuda"
                and "CUDAExecutionProvider" in available_providers
            ):
                providers = [
                    (
                        "CUDAExecutionProvider",
                        {
                            "device_id": 0,
                            "arena_extend_strategy": "kNextPowerOfTwo",
                            "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                            "cudnn_conv_algo_search": "EXHAUSTIVE",
                        },
                    ),
                    "CPUExecutionProvider",
                ]
            else:
                providers = ["CPUExecutionProvider"]

            self.ort_session = ort.InferenceSession(
                str(self.model_path), providers=providers
            )
            active_provider = self.ort_session.get_providers()[0]
            logger.info(
                f"ONNX Session loaded successfully with provider: {active_provider}"
            )

        else:
            raise ValueError(
                f"Unsupported model extension '{self.model_format}'. Use .pt or .onnx."
            )

    def preprocess_image(self, img: Image.Image) -> torch.Tensor:
        """Converts PIL Image to normalized BCHW PyTorch tensor."""
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img).unsqueeze(0)

    def predict(self, img: Image.Image) -> Dict[str, float]:
        """Executes forward pass and returns class probability distribution."""
        tensor_input = self.preprocess_image(img)

        if self.model_format == ".pt":
            tensor_input = tensor_input.to(self.device_str)
            with torch.no_grad():
                logits = self.model(tensor_input)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        elif self.model_format == ".onnx":
            numpy_input = tensor_input.numpy()
            input_name = self.ort_session.get_inputs()[0].name
            logits = self.ort_session.run(None, {input_name: numpy_input})[0]
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = (exp_logits / np.sum(exp_logits, axis=1, keepdims=True)).squeeze(0)

        # Map predictions to class labels with dynamic index fallback
        prob_dict = {
            (self.class_names[i] if i < len(self.class_names) else f"class_{i}"): float(
                probs[i]
            )
            for i in range(len(probs))
        }
        return dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))


engine: Optional[ModelInferenceEngine] = None


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Modern FastAPI Lifespan Handler replacing deprecated on_event handlers."""
    global engine
    model_path = os.getenv("MODEL_PATH", "./outputs_cifar100_cuda/winner_4f5974bd.onnx")
    device = os.getenv("DEVICE", "cuda")

    if Path(model_path).exists() and engine is None:
        engine = ModelInferenceEngine(model_path=model_path, device=device)
    yield


# Instantiate FastAPI Application
app = FastAPI(
    title="NeuroSwarm-AutoML Inference Server",
    description="High-performance async REST API and Gradio Web UI for deployed AutoML model candidates.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint confirming engine readiness and active hardware."""
    if engine is None:
        raise HTTPException(
            status_code=503, detail="Model inference engine is not initialized."
        )
    return HealthResponse(
        status="HEALTHY",
        model_path=str(engine.model_path),
        model_format=engine.model_format,
        device=engine.device_str,
        num_classes=engine.num_classes,
    )


@app.post("/predict", response_model=ClassificationResponse)
async def predict_image(file: UploadFile = File(...)):
    """Receives an uploaded image file and returns softmax class probabilities."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model engine not loaded.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be an image.",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        prob_dict = engine.predict(image)

        top_class = next(iter(prob_dict))
        top_confidence = prob_dict[top_class]

        return ClassificationResponse(
            top_prediction=top_class,
            confidence=top_confidence,
            probabilities=prob_dict,
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Image processing failed: {str(e)}"
        )


def gradio_predict(img: Image.Image) -> Dict[str, float]:
    """Inference function callback for Gradio UI."""
    if engine is None or img is None:
        return {}
    return engine.predict(img)


def create_gradio_ui() -> gr.Blocks:
    """Constructs the Gradio interactive web UI layout."""
    with gr.Blocks(title="NeuroSwarm-AutoML Classifier") as demo:
        gr.Markdown("# 🛸 NeuroSwarm-AutoML Model Classifier")
        gr.Markdown(
            "Upload an image to evaluate top-k predictions from your deployed neural architecture."
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Input Image (32x32 RGB)")
                submit_btn = gr.Button("Classify Image", variant="primary")
            with gr.Column():
                label_output = gr.Label(
                    num_top_classes=5, label="Prediction Probabilities"
                )

        submit_btn.click(fn=gradio_predict, inputs=image_input, outputs=label_output)
        image_input.change(fn=gradio_predict, inputs=image_input, outputs=label_output)

    return demo


# Mount Gradio app onto FastAPI under /ui
demo_ui = create_gradio_ui()
app = gr.mount_gradio_app(app, demo_ui, path="/ui")


def parse_args():
    parser = argparse.ArgumentParser(
        description="NeuroSwarm-AutoML FastAPI + Gradio Server"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./outputs_cifar100_cuda/winner_4f5974bd.onnx",
        help="Path to .onnx or .pt model binary",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference hardware device",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Binding host address"
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port number")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Initialize engine prior to launching uvicorn worker
    if Path(args.model).exists():
        engine = ModelInferenceEngine(model_path=args.model, device=args.device)
    else:
        logger.error(f"Specified model path does not exist: {args.model}")

    print("=" * 65)
    print(">>> NeuroSwarm-AutoML Server Launching...")
    print(f"REST API Endpoint:   http://{args.host}:{args.port}/predict")
    print(f"Gradio Web UI:      http://{args.host}:{args.port}/ui")
    print(f"API Documentation:  http://{args.host}:{args.port}/docs")
    print("=" * 65)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
