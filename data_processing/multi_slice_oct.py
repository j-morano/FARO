from pathlib import Path
import json
from multiprocessing import Pool

import numpy as np
from skimage import io



base_path = Path('.')
oct_path = base_path / 'oct'

metadata = 'vibes_metadata.json'
with open(metadata, 'r') as f:
    info = json.load(f)

# Print the first element of the dict to verify loading
print(list(info.items())[0:5])

vendor_mapping = {
    'Heidelberg Engineering': 'Spectralis',
    'Cirrus': 'Cirrus',
}

bscan_path = Path('bscan')
bscan_path.mkdir(exist_ok=True)

max_slices = 49

def split_to_slices(fns):
    for fn in fns:
        fn = Path(fn)
        data = np.load(fn)['data']
        total = data.shape[0]
        if total > 50:
            start = 1
            end = total - 2
        else:
            start = 0
            end = total - 1
        num_slices = min(total, max_slices)
        selected_indices = np.linspace(start, end, num_slices, dtype=int)
        vendor = vendor_mapping[info[fn.stem]['Vendor']]
        laterality = info[fn.stem]['Position']
        print(
            f"Processing {fn.stem}\n"
            f"   total slices={total}, selected slices={num_slices}\n"
            f"   start={start}, end={end}\n"
            f"   vendor={vendor}, laterality={laterality}"
        )
        tgt_dir = bscan_path / fn.stem
        tgt_dir.mkdir(exist_ok=True)
        for i in selected_indices:
            name = f"{vendor}__{laterality}__{i:03d}-{total:03d}.png"
            slice_fn = tgt_dir / name
            io.imsave(slice_fn, data[i])

n_proc = 44

fns = list(oct_path.iterdir())
chunks = np.array_split(fns, n_proc)  # type: ignore

with Pool(n_proc) as p:
    p.map(split_to_slices, chunks)


print("Done!")

