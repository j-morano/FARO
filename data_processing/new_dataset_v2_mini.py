from pathlib import Path
import json
import gzip
import shutil

bscanlayermap_path = Path('/mnt/Data/FullVIBES-v2/v2_bscanlayermap')

files = []
for first_dir in bscanlayermap_path.iterdir():
    for second_dir in first_dir.iterdir():
        for fsid_dir in second_dir.iterdir():
            fsid = fsid_dir.name
            for file_path in fsid_dir.iterdir():
                # Append only two last levels of the path
                subpath = str(file_path.relative_to(bscanlayermap_path))
                files.append(subpath)

save_fn = '/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/cfgs/v2_minifiles.json.gz'
with gzip.open(save_fn, 'wt') as f:
    json.dump(files, f)

