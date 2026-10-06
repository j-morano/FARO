from typing import Union, Optional, Tuple
from pathlib import Path
import xml.etree.ElementTree as ET
import shutil

import numpy as np
import matplotlib.pyplot as plt

from pyoptimadb.connection import OptimaConnection


NDArray = np.ndarray



def get_optima_connection() -> OptimaConnection:
    return OptimaConnection(
        db="",
        user="",
        pwd=""
    )


def get_metadata(
    file_set_id: str,
    optima_connection: Optional[OptimaConnection]=None,
    verbose: bool=False,
) -> dict:
    """Gets metadata for a given file_set_id."""
    if optima_connection is None:
        optima_connection = get_optima_connection()
    assert optima_connection is not None
    scan_metadata = optima_connection.services.scan.load_by_filesetid(file_set_id)
    if verbose:
        print(scan_metadata)
    return vars(scan_metadata)

def get_oct_and_fundus_fns(
    file_set_id: str,
    optima_connection: Optional[OptimaConnection]=None,
    verbose: bool=False,
) -> Tuple[Path, Path]:
    """Gets OCT and fundus filenames for a given file_set_id."""
    if optima_connection is None:
        optima_connection = get_optima_connection()
    assert optima_connection is not None
    scan_metadata = optima_connection.services.scan.load_by_filesetid(file_set_id)
    if verbose:
        print(scan_metadata)
    scan_full_path = optima_connection.services.data._get_scan_base_url(scan_metadata)
    return (
        Path(scan_full_path) / 'bscan.dcm',
        Path(scan_full_path) / 'fundus.dcm'
    )


def get_oct_fn_and_metadata(
    file_set_id: str,
    optima_connection: Optional[OptimaConnection]=None,
    verbose: bool=False,
) -> Tuple[Path, dict]:
    """Gets OCT and fundus filenames for a given file_set_id."""
    if optima_connection is None:
        optima_connection = get_optima_connection()
    scan_metadata = optima_connection.services.scan.load_by_filesetid(file_set_id)
    if verbose:
        print(scan_metadata)
    scan_full_path = optima_connection.services.data._get_scan_base_url(scan_metadata)
    return Path(scan_full_path) / 'bscan.dcm', vars(scan_metadata)


def get_layers_from_xml(fn: Union[str, Path]) -> NDArray:
    with open(fn, 'r') as f:
        xml = f.read()

    root = ET.fromstring(xml)

    layers = root.findall('surface')
    layers_seg = []
    for layer in layers:
        # name = layer.find('name')
        bscans = layer.findall('bscan')
        seg = []
        for bscan in bscans:
            bscan_seg = []
            for z in bscan.findall('z'):
                bscan_seg.append(int(z.text))  # type: ignore
            seg.append(bscan_seg)
        layers_seg.append(seg)
    layers_seg = np.array(layers_seg).transpose(1, 2, 0)

    return layers_seg


def get_layer_maps(bscan_height: int, layers):
    """Get layer maps for a given shape and layers.
    Args:
        bscan_height: B-scan height.
        layers: NDArray in the format (num_bscans, bscan_width, num_layers)
            [e.g., (128, 512, 12)]. Every layer is a 2D array indicating the
            position of the layer within the B-scan.
    Returns:
        3D array with the shape (num_bscans, bscan_height, bscan_width)
        where every B-scan has the layers as a semantic segmentation 2D
        map.
    """
    num_bscans, bscan_width, _ = layers.shape
    masks = np.zeros(shape=(num_bscans, bscan_height, bscan_width), dtype=int)

    xx, yy, zz = np.meshgrid(
        np.arange(0, num_bscans, 1, dtype=int),
        np.arange(0, bscan_height, 1, dtype=int),
        np.arange(0, bscan_width, 1, dtype=int),
        indexing='ij',
    )
    for i in range(layers.shape[2]):
        layer = layers[:, :, i]
        masks[yy >= layer[xx, zz]] = i + 1

    return masks.astype(np.uint8)


def remove_tmp():
    print('Removing tmp contents')
    for fn in Path('./__tmp').iterdir():
        if not 'debug_layer_segmentation' in fn.stem and fn.is_dir():
            shutil.rmtree(fn)


def print_progress(current, total):
    percent = int(round(current / len(total) * 100))
    print('=' * 107)
    print('#' * percent, ' ' * (100 - percent), f'| {percent}%')
    print('=' * 107)


def debug_layers(oct_pa, layer_map, file_set_id, debug_path):
    if len(oct_pa.shape) == 3:
        oct_pa = oct_pa.transpose(1, 2, 0)
    plt.imshow(oct_pa, cmap='gray')
    alpha=0.3
    plt.imshow(layer_map, alpha=alpha, cmap='Set3') # , cmap='tab20')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(
        debug_path / f'{file_set_id}_layers.jpg',
        bbox_inches='tight',
        pad_inches=0,
        dpi=150,
    )
    plt.close()


