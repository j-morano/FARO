import argparse
from pathlib import Path
import subprocess
import multiprocessing

import numpy as np
import pydicom
from skimage import io
from skimage.transform import resize

import utils



def get_fake_dicom():
    ds = pydicom.dcmread('./_data/bscan.dcm')
    ds.SamplesPerPixel = 1
    ds.NumberOfFrames = 3
    # Change (0022, 0031)  Ophthalmic Frame Location Sequence  49 item(s)
    middle = len(ds[0x0022, 0x0031].value)//2
    ds[0x0022, 0x0031].value = [
        ds[0x0022, 0x0031].value[middle-1],
        ds[0x0022, 0x0031].value[middle],
        ds[0x0022, 0x0031].value[middle+1],
    ]
    ds.BitsStored = 8
    ds.BitsAllocated = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    return ds


class LayerGetter():
    def __init__(self, args):
        self.args = args
        self.ds = get_fake_dicom()

    def save_fake_dicom(self, image_data, output_file):
        self.ds.Rows, self.ds.Columns = image_data.shape[1], image_data.shape[2]
        self.ds.PixelData = image_data.tobytes()
        self.ds.save_as(output_file)

    def get_layers(self, paths):
        for pi, path in enumerate(paths):
            utils.print_progress(pi, paths)
            fn = Path(path)
            if fn.suffix == '.npz':
                bscan = np.load(fn)
            elif fn.suffix == '.npy':
                bscan = np.load(fn)
            elif fn.suffix in ['.png', '.jpg', '.jpeg']:
                bscan = io.imread(fn)
            else:
                raise ValueError('Unrecognized file type:', fn)
            # print(bscan.shape)

            if self.args.resize:
                bscan = resize(bscan, (512, 512), preserve_range=True).astype(np.uint8)

            if len(bscan.shape) == 2:
                bscan = np.stack([bscan,] * 3, axis=0)
            elif len(bscan.shape) == 3:
                if bscan.shape[-1] > 1:
                    bscan = bscan.transpose(2, 0, 1)

            current_path = Path('./__tmp') / fn.stem
            layers_path = Path('./__tmp') / fn.stem / 'layers'

            # Save the B-scan as dicom in the temporary directory
            current_path.mkdir(parents=False, exist_ok=True)
            layers_path.mkdir(parents=False, exist_ok=True)
            current_fn = current_path / 'bscan.dcm'

            self.save_fake_dicom(bscan, current_fn)

            subprocess.run([args.script, current_path, layers_path])

            layers_fn = layers_path / 'iowa_layer_raw/lres.xml'
            layers_seg = utils.get_layers_from_xml(layers_fn)

            # Run a different process to remove the directory
            subprocess.Popen(["rm", "-r", current_path])

            layer_maps = utils.get_layer_maps(bscan.shape[1], layers_seg)

            # Remap the layers to 0-255, so the layer maps are visible
            #   in the images.
            # 255 (max) // 11 (num layers) = 23
            layer_map = layer_maps[1] * 23
            # print(layer_map.shape, np.unique(layer_map))
            io.imsave(
                Path(self.args.save_path) / fn.with_suffix('.png').name,
                layer_map,
            )
            if not self.args.dont_debug:
                utils.debug_layers(bscan, layer_map, path.stem, Path(self.args.debug_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', type=str)
    parser.add_argument('--save-path', type=str)
    parser.add_argument('--debug-path', type=str, default='./__tmp/00005_debug_layer_segmentation')
    parser.add_argument('--script', type=str, default='./_scripts/iowa/iowa_layer_csv.sh')
    parser.add_argument('--nproc', type=int, default=None)
    parser.add_argument('--dont-debug', action='store_true')
    parser.add_argument('--print-sample', action='store_true')
    parser.add_argument('--resize', action='store_true')
    args = parser.parse_args()

    if args.print_sample:
        # Print all details of the example DICOM file
        example_fn = Path('./_data/bscan.dcm')
        example_dcm = pydicom.dcmread(example_fn)
        print(example_dcm)
        example_dcm.SamplesPerPixel = 1
        pa = example_dcm.pixel_array
        print(pa.shape)
        exit(0)

    # utils.remove_tmp()

    base_path = Path(args.base_path)
    save_path = Path(args.save_path)
    debug_path = Path(args.debug_path)
    save_path.mkdir(parents=False, exist_ok=True)
    debug_path.mkdir(parents=False, exist_ok=True)

    ids = []
    for fn in base_path.iterdir():
        if fn.is_file():
            ids.append(fn.stem)
    done_ids = []
    for fn in save_path.iterdir():
        if fn.is_file():
            done_ids.append(fn.stem)
    ids = list(set(ids) - set(done_ids))
    paths = []
    for fn in base_path.iterdir():
        if fn.is_file() and fn.stem in ids:
            paths.append(fn)

    print('Total files:', len(paths))

    if args.nproc:
         n_processes = args.nproc
    else:
        # As many as available CPUs minus 4, to avoid overloading the
        #   system.
        n_processes = multiprocessing.cpu_count() - 4
    print('Using', n_processes, 'processes')

    layer_getter = LayerGetter(args)

    paths_splits = np.array_split(paths, n_processes)
    with multiprocessing.Pool(n_processes) as pool:
        pool.map(layer_getter.get_layers, paths_splits)

