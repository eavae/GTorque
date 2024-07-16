import requests
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from typing import List


def get_image(url_or_base64: str) -> Image.Image:
    if url_or_base64.startswith("data:"):
        image_content = url_or_base64.split(",")[1]
        image_content = base64.b64decode(image_content)
        image = Image.open(BytesIO(image_content))
        return image

    if url_or_base64.startswith("http"):
        response = requests.get(url_or_base64)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image

    return Image.open(url_or_base64)


def get_image_resampling(name: str):
    if name == "nearest":
        return Image.NEAREST
    elif name == "bilinear":
        return Image.BILINEAR
    elif name == "bicubic":
        return Image.BICUBIC
    elif name == "lanczos":
        return Image.LANCZOS
    else:
        raise NotImplementedError


def normalize_with(image: np.ndarray, mean: List[float], std: List[float]):
    assert image.shape[-1] == len(mean) == len(std)
    image /= 255.0
    image -= np.array(mean)
    image /= np.array(std)
    return image
