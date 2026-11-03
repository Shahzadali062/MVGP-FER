"""Image preprocessing in the order described in the paper."""

from PIL import Image

from .grayscale import to_grayscale
from .normalize import normalize_image, replicate_channels
from .resize import resize_image


def preprocess_image(image: Image.Image):
    gray = resize_image(to_grayscale(image))
    rgb = replicate_channels(gray)
    return normalize_image(rgb), rgb
