
import sys
from collections import defaultdict
import importlib
sys.path.insert(0, '/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc')

def to_ng(coord):
    return [
        int(coord[0]/4),
        int(coord[1]/4),
        int(coord[2]/40),
        ]

'''Load data'''
import compress_pickle

f0_threshold = 50
f1_threshold = 30

if len(sys.argv) >= 3:
    f0_threshold = int(sys.argv[1])
    f1_threshold = int(sys.argv[2])

fname0 = f'/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/gen_db/grc_axons/gen_201223_setup01_syndb_threshold_10_coalesced.gz'
fname1 = f'/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/gen_db/grc_axons/gen_201223_setup01_syndb_threshold_15_coalesced.gz'
# fname1 = f'/n/groups/htem/Segmentation/shared-nondev/cb2_segmentation/analysis_mf_grc/gen_db/grc_axons/gen_201221_setup22_syndb_threshold_25_coalesced.gz'

pfs_pc0 = compress_pickle.load(fname0)
pfs_pc1 = compress_pickle.load(fname1)
fname0 = fname0.split('/')[-1]
fname1 = fname1.split('/')[-1]

pfs_pc0_new = defaultdict(lambda: defaultdict(list))
for k in pfs_pc0:
    pfs_pc0_new[k] = pfs_pc0[k]
pfs_pc1_new = defaultdict(lambda: defaultdict(list))
for k in pfs_pc1:
    pfs_pc1_new[k] = pfs_pc1[k]

pfs_pc0 = pfs_pc0_new
pfs_pc1 = pfs_pc1_new

grcs = sorted([k for k in set(pfs_pc0.keys()) | set(pfs_pc1.keys())])

mode = 'false-negative'
# mode = 'false-positive'
# threshold = 1
threshold = .01

print_diff_only = True

num_merged0 = 0
num_merged1 = 0

for grc in grcs:
    pcs = set(pfs_pc0[grc].keys()) | set(pfs_pc1[grc].keys())
    for pc in pcs:
        locs0 = set([loc[1] for loc in pfs_pc0[grc][pc]])
        locs1 = set([loc[1] for loc in pfs_pc1[grc][pc]])
        num_merged0 += len(locs0)
        num_merged1 += len(locs1)

        diff = locs1.symmetric_difference(locs0)

        if len(diff):
            print(f'{grc}')
            for loc in diff:
                if loc in locs0:
                    print(f'{fname0}: {loc}')
                if loc in locs1:
                    print(f'{fname1}: {loc}')
            print()

            # for loc in locs1:
                # if print_diff_only and loc in locs0: continue
                # point0, point1 = loc
                # print(f'{to_ng(point0)} {to_ng(point1)}')
                # print(loc)
            # print()

print(f'{fname0} merged {num_merged0} pairs')
print(f'{fname1} merged {num_merged1} pairs')
diff = num_merged1-num_merged0
diff_pct = diff / num_merged0 * 100
print(f'n = +{diff}')
print(f'% = +{diff_pct}')


    # diff = len(locs0) - len(locs1)
    # max_len = max(len(locs0), len(locs1))
    # min_len = min(len(locs0), len(locs1))

    # diff_norm = (len(locs0) - len(locs1)) / max(len(locs0), len(locs1))

    # if mode == 'false-negative':
    #     score = -diff_norm
    # elif mode == 'false-positive':
    #     score = diff/min_len
    # else:
    #     assert False

    # if score >= threshold:
    #     print()
    #     print(f'{k}')
    #     for loc in locs0:
    #         if print_diff_only and loc in locs1: continue
    #         point0, point1 = loc
    #         print(f'{to_ng(point0)} {to_ng(point1)}')
    #     print()
    #     for loc in locs1:
    #         if print_diff_only and loc in locs0: continue
    #         point0, point1 = loc
    #         print(f'{to_ng(point0)} {to_ng(point1)}')
    #     print()


