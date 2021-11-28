import collections
from collections import defaultdict
import sys
import json
import random
from jsmin import jsmin
from io import StringIO
import numpy as np
import copy
import os

script_n = os.path.basename(__file__).split('.')[0]
script_n = script_n.split('_', 1)[1]

def to_ng(loc):
    return (int(loc[0]/4), int(loc[1]/4), int(loc[2]/40))

'''Load data'''
import compress_pickle
fname = 'gen_210518_setup01_v2_syndb_threshold_20_coalesced.gz'
grc_mfs_locs = compress_pickle.load(fname)

mfs_locs = defaultdict(list)
for grc in grc_mfs_locs:
    for mf in grc_mfs_locs[grc]:
        for syn in grc_mfs_locs[grc][mf]:
            mfs_locs[mf].append(syn['syn_loc0'])

# print(mfs_locs[mf]); asdf
asdff = (172644, 113468, 89)
asdfff = (137580, 101824, 369)

# white list for big boutons
whitelist = set([
    (172644, 113468, 89),
    (163520, 98364, 83),
    (113008, 109372, 1154),
    (70424, 116512, 71),
    (186536, 100020, 130),
    (86780, 110184, 81),
    (177992, 108528, 1164),
    (127368, 101716, 1143),
    (155036, 103252, 71),
    (97884, 104152, 1160),
    (109476, 104808, 76),
    (82936, 122484, 76),
    (113532, 104660, 1150),
    (78904, 115540, 1158),
    (190684, 91276, 1015),
    (160500, 99828, 1165),
    (109020, 115476, 74),
    (93516, 101476, 858),
    (126728, 104988, 86),
    (173456, 106376, 71),
    (197436, 95688, 898),
    (122752, 110608, 85),
    (122192, 119344, 70),
    (122396, 118840, 83),
    (204868, 103452, 145),
    (94212, 107860, 1137),
    (92360, 105844, 1162),
    (84704, 115452, 119),
    (54036, 105484, 394),
    (110624, 105800, 70),
    (170512, 99132, 107),
    (71200, 114308, 1123),
    (106588, 98692, 1160),
    (70164, 107908, 1015),
    (144772, 106812, 105),
    (asdff),
    (asdff),
    (asdff),
])


blacklist = set([
    (137580, 101824, 369),
    (127384, 115252, 746),
    (155268, 99276, 918),
    (182000, 91966, 716),
    (119828, 107400, 312),
    (171384, 94244, 573),
    (asdfff),
    (asdfff),
    (asdfff),
    (asdfff),
    (asdfff),
    (asdfff),
])

'''Cluster and extract locations of MF boutons'''
from sklearn.cluster import DBSCAN

mfs_bouton_locs = {}

'''if a bouton location has less than this many synapses then it won't be considered in order to reduce false positives'''
# bouton_synapse_threshold = 6  # safe for determining big bouton locations
bouton_synapse_threshold = 2
bouton_synapse_threshold = 3
bouton_synapse_threshold = 4  # 4 is a bit iffy, since it has some semi big boutons
bouton_synapse_threshold = 5
# bouton_synapse_threshold = 6  # this threshold has quite a bit of FPs

for mf in mfs_locs:
    dbscan = DBSCAN(eps=8000, min_samples=2)  # max dist set to 8um
    # dbscan = DBSCAN(eps=10000, min_samples=2)  # max dist set to 8um
    dbscan.fit(mfs_locs[mf])
    loc_by_label = defaultdict(list)
    for loc, label in zip(mfs_locs[mf], dbscan.labels_):
        loc_by_label[label].append(loc)
    mf_bouton_locs = []
    for label in loc_by_label:
        if len(loc_by_label[label]) <= bouton_synapse_threshold:
            whitelisted = False
            for loc in loc_by_label[label]:
                if to_ng(loc) in whitelist:
                    whitelisted = True
            if not whitelisted:
                if len(loc_by_label[label]) >= 2:
                    print(f'Ignoring {mf} due to insufficient synapses')
                    for loc in loc_by_label[label]:
                        print(to_ng(loc))
                    continue
        sum = [0, 0, 0]
        for loc in loc_by_label[label]:
            sum = [sum[0]+loc[0], sum[1]+loc[1], sum[2]+loc[2]]
        center = [
            int(sum[0]/len(loc_by_label[label])),
            int(sum[1]/len(loc_by_label[label])),
            int(sum[2]/len(loc_by_label[label])),
            ]
        mf_bouton_locs.append(center)
    mfs_bouton_locs[mf] = mf_bouton_locs
    # print(mf_bouton_locs)
    # for loc in mf_bouton_locs:
        # print([int(loc[0]/4), int(loc[1]/4), int(loc[2]/40)])

mfs_bouton_count = defaultdict(list)
for mf in mfs_bouton_locs:
    mfs_bouton_count[len(mfs_bouton_locs[mf])].append(mf)

for count in sorted(mfs_bouton_count.keys()):
    print(f'{count}: {mfs_bouton_count[count]}')

'''save mfs_bouton_locs'''
import compress_pickle
compress_pickle.dump((
    mfs_bouton_locs
    ), f"{script_n}.gz")


asdf

for loc in mfs_bouton_locs['mf_431']:
    print([int(loc[0]/4), int(loc[1]/4), int(loc[2]/40)])

for loc in mfs_locs['mf_41']:
    print([int(loc[0]/4), int(loc[1]/4), int(loc[2]/40)])
