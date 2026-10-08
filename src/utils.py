from pathlib import Path
import numpy as np
from skimage import img_as_float, io, color, data

def load_image(path=None):
    if path is None:
        image = data.camera()
        source = "skimage.data.camera()"
    else:
        image = io.imread(path)
        source = str(path)

    image = img_as_float(image)

    if image.ndim == 3:
        if image.shape[-1] == 4:
            image = image[..., :3]
        image = color.rgb2gray(image)

    if image.ndim != 2:
        raise ValueError("Input must be a 2-D grayscale image or a standard RGB/RGBA image.")

    return np.clip(image, 0.0, 1.0), source

def save_image(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(image, 0.0, 1.0)
    io.imsave(path, (image * 255.0 + 0.5).astype(np.uint8))

def normalize_for_display(image):
    image = np.asarray(image, dtype=np.float64)
    lo, hi = np.percentile(image, [1, 99])
    if hi <= lo:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)
