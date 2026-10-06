import numpy as np
from skimage import io, transform as sk_transform


def pil_loader(path: str, convert_rgb=True) -> np.ndarray:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # with open(path, 'rb') as f:
    #     img = Image.open(f)
    img = io.imread(path)
    img = sk_transform.resize(img, (512, 512), anti_aliasing=False)
    # image to 0-1 range
    img = img / 255.0
    return img.astype(np.float32)
    # img = Image.open(path)
    # return img.convert('RGB') if convert_rgb else img
