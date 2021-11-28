
import sys
from collections import defaultdict
import compress_pickle

in_fname = sys.argv[1]
in_fname_base = in_fname.rsplit('.', 1)[0]

syndb = compress_pickle.load(in_fname)
filtered_db = defaultdict(lambda: defaultdict(list))

n = 0
n_filtered = 0
n_total = 0

threshold = 10

for grc, grc_syns in syndb.items():
    print(grc)
    for pc, syns in grc_syns.items():
        for s in syns:
            n_total += 1
            if s['major_axis_length'] <= threshold:
                n_filtered += 1
                continue
            filtered_db[grc][pc].append(s)
            n += 1

print(f'Filtered {n_filtered} ({n_filtered/n_total*100}%), out of {n_total}')

fout = f'{in_fname_base}_filtered_{threshold}.gz'

compress_pickle.dump((
    dict(filtered_db)
    ), fout)


