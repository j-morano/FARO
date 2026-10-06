import json

with open('./__info/to_segment_vibes.json', 'r') as f:
    info = json.load(f)

with open('./__info/vibes_metadata.json', 'r') as f:
    metadata = json.load(f)

new_to_segment = []
for fsid in info:
    if fsid in metadata.keys():
        new_to_segment.append(fsid)

with open('./__info/correct_to_segment_vibes.json', 'w') as f:
    json.dump(new_to_segment, f)
