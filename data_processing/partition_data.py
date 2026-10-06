import json

with open('./__info/vibes_metadata.json', 'r') as f:
    info = list(json.load(f).keys())

info_0 = info[:len(info)//2]
info_1 = info[len(info)//2:]

with open('./__info/partition_0_vibes.json', 'w') as f:
    json.dump(info_0, f)

with open('./__info/partition_1_vibes.json', 'w') as f:
    json.dump(info_1, f)

with open('./__info/partition_all_vibes.json', 'w') as f:
    json.dump(info, f)

