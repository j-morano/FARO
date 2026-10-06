import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import dilation, disk
from skimage.measure import label
import lzma
import yaml
import matplotlib.pyplot as plt



def remove_white_borders(img, show=False):
    img = np.pad(img, 10, constant_values=255)
    bimg = img.copy()
    # Pad image with 255
    bimg = gaussian_filter(bimg, sigma=1)
    bimg = (bimg > 253).astype(int)
    selem = disk(10)
    try:
        bimg = dilation(bimg, selem)
    except IndexError as e:
        print('Error in binary_dilation')
        raise e
    # Get connected components
    label_img = label(bimg, background=0)
    assert isinstance(label_img, np.ndarray)
    # Plot the regions in different colors
    if show:
        plt.imshow(img, cmap='gray')
        plt.imshow(label_img, cmap='nipy_spectral')
        plt.show()
    for v in np.unique(label_img):
        num_pixels = np.sum(label_img == v)
        # print(v, num_pixels)
        if num_pixels > 10000 and v != 0:
            img[label_img == v] = 0
    img = img[10:-10, 10:-10]
    # print(img.shape)
    if show:
        plt.imshow(img, cmap='gray')
        plt.show()
    return img



def get_layer_maps_from_yml(yml_path):
    with lzma.open(yml_path, 'rt') as f:
        data = yaml.safe_load(f)

    width = data['info']['width']
    height = data['info']['height']
    raw_layers = data['layers']

    # 1. Pre-allocate the mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # 2. Vectorized Y-indices (the "column")
    y_indices = np.arange(height).reshape(-1, 1)

    # 3. Process layers
    for i, layer_dict in enumerate(raw_layers):
        layer_name = list(layer_dict.keys())[0]
        val_str = layer_dict[layer_name][0]['values']

        # np.fromstring is very CPU-efficient
        coords = np.fromstring(val_str, sep=',', dtype=int)

        # BROADCASTING: This is the CPU-fastest part.
        # It creates the boolean mask on the fly in the CPU registers.
        mask[y_indices >= coords] = i + 1

    return mask


def get_thicknesses_from_yml(yml_path):
    with lzma.open(yml_path, 'rt') as f:
        # Optimization: Use C-based loader if available for 10x YAML speed
        try:
            data = yaml.load(f, Loader=yaml.CSafeLoader)
        except AttributeError:
            data = yaml.safe_load(f)

    height = data['info']['height']
    raw_layers = data['layers']

    # 1. Extract all boundary lines into a list of 1D arrays
    # Each entry in 'boundaries' is an array of shape (width,)
    boundaries = []
    for layer_dict in raw_layers:
        # Get the first value list from the dict
        val_str = next(iter(layer_dict.values()))[0]['values']
        boundaries.append(np.fromstring(val_str, sep=',', dtype=int))

    # 2. Add the final bottom boundary (the bottom of the image)
    # This allows us to calculate the thickness of the last layer
    bottom_boundary = np.full_like(boundaries[0], height)
    boundaries.append(bottom_boundary)

    # 3. Calculate mean and std for each layer by subtracting boundaries
    # Layer 1 thickness = boundary[1] - boundary[0]
    # Layer 2 thickness = boundary[2] - boundary[1], etc.
    thickness_stats = {}

    # We only care about layers 1 through 11
    # raw_layers[0] is usually the top-most boundary
    for i in range(1, 12):
        # Subtract current boundary from the one below it to get pixel height
        col_thicknesses = boundaries[i] - boundaries[i-1]

        # Calculate stats in pixel units
        # (Scaling to um should happen in your main process function)
        mean_px = col_thicknesses.mean()
        std_px = col_thicknesses.std()

        thickness_stats[i] = (mean_px, std_px)

    return thickness_stats, data['info']
