"""Safe image loading and resizing helpers."""
from __future__ import annotations

import io

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 40_000_000


class InvalidImageError(ValueError):
    """The bytes are not a usable image."""


def load_image(data: bytes) -> Image.Image:
    if not data:
        raise InvalidImageError("The file is empty.")
    if len(data) > MAX_BYTES:
        raise InvalidImageError("The file is larger than 20 MB.")
    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > MAX_PIXELS:
            raise InvalidImageError("The image has more than 40 megapixels.")
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("This file is not a readable image (use JPG, PNG, WebP or BMP).") from exc
    return ImageOps.exif_transpose(img).convert("RGB")


def fit_longest_side(img: Image.Image, max_side: int) -> Image.Image:
    """Downscale (never upscale) so the longest side is at most ``max_side``."""
    if max(img.size) <= max_side:
        return img
    scale = max_side / max(img.size)
    return img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.Resampling.LANCZOS)


def upscale_square(img: Image.Image, size: int) -> Image.Image:
    return img.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
