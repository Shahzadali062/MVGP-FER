"""Resize an image without changing its channel count."""

from PIL import Image

SIZE = (224, 224)


def resize_image(image: Image.Image) -> Image.Image:
    return image.resize(SIZE, Image.Resampling.BILINEAR)
