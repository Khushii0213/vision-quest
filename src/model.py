"""Pretrained MobileNetV3-Small (ImageNet-1K) classifier.

Pipeline: PIL RGB image -> weights.transforms() (resize 256, centre-crop 224, scale to [0,1],
normalise with ImageNet mean/std) -> model in eval() + inference_mode() -> softmax -> top-k.
There is no fallback: if the model cannot load, ModelUnavailableError is raised and the UI says so.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from PIL import Image

from . import config  # noqa: F401  (sets TORCH_HOME before torch is used)


class ModelUnavailableError(RuntimeError):
    """PyTorch/torchvision is missing or the pretrained weights could not be obtained."""


class InferenceError(RuntimeError):
    """The forward pass failed."""


@dataclass(frozen=True)
class Prediction:
    class_index: int
    label: str
    confidence: float  # softmax probability, 0..1


class Classifier:
    def __init__(self, torch_module, model, preprocess, categories: list[str]):
        self._torch, self._model, self._preprocess, self._categories = torch_module, model, preprocess, categories
        self._lock = threading.Lock()

    @classmethod
    def load(cls) -> "Classifier":
        try:
            import torch
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        except ImportError as exc:
            raise ModelUnavailableError("PyTorch/torchvision is not installed. Run: pip install -r requirements.txt") from exc
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        try:
            model = mobilenet_v3_small(weights=weights)  # downloads ~10 MB once, then cached
        except (OSError, RuntimeError) as exc:
            raise ModelUnavailableError(
                "The pretrained MobileNetV3-Small weights could not be downloaded. Connect to the internet "
                f"for the first launch and try again. Details: {exc}") from exc
        model.eval()
        return cls(torch, model, weights.transforms(), list(weights.meta["categories"]))

    def predict(self, image: Image.Image, top_k: int = 5) -> list[Prediction]:
        torch = self._torch
        try:
            batch = self._preprocess(image.convert("RGB")).unsqueeze(0)
            with self._lock, torch.inference_mode():
                probs = torch.softmax(self._model(batch), dim=1)[0]
                conf, idx = torch.topk(probs, k=top_k)
        except RuntimeError as exc:
            raise InferenceError(f"Inference failed: {exc}") from exc
        return [Prediction(int(i), self._categories[int(i)], float(c)) for c, i in zip(conf.tolist(), idx.tolist())]
