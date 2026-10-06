import json
import gzip
import pickle



fn = '/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/cfgs/v3_all_metadata.pkl.gz'


with gzip.open(fn, 'rb') as f:
    data = pickle.load(f)


# Get first item of data and print
item = list(data.items())[0]
print(item)
