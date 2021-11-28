import collections
from collections import defaultdict
import sys
import json
import random
from jsmin import jsmin
from io import StringIO
import numpy as np
import copy

'''Load data'''
import compress_pickle
fname = 'mf_grc_weights/grc-mf-locs-setup22-setup09-201114.gz'
grc_mfs_locs, grc_mfs_adj_counts = compress_pickle.load(fname)

mfs_locs = defaultdict(list)
# mfs_synapse_count = defaultdict(int)

for grc in grc_mfs_locs:
    for mf in grc_mfs_locs[grc]:
        mfs_locs[mf].extend(grc_mfs_locs[grc][mf])
        # mfs_synapse_count[mf] += grc_mfs_adj_counts[grc][mf]

# print(mfs_locs)

'''Cluster and extract locations of MF boutons'''
from sklearn.cluster import DBSCAN

mfs_bouton_locs = {}

'''if a bouton location has less than this many synapses then it won't be considered in order to reduce false positives'''
bouton_synapse_threshold = 3

for mf in mfs_locs:
    dbscan = DBSCAN(eps=8000)  # max dist set to 8um
    dbscan.fit(mfs_locs[mf])
    # dbscan.labels_
    loc_by_label = defaultdict(list)
    for loc, label in zip(mfs_locs[mf], dbscan.labels_):
        loc_by_label[label].append(loc)
    mf_bouton_locs = []
    for label in loc_by_label:
        if len(loc_by_label[label]) < bouton_synapse_threshold:
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
    ), "mfs_bouton_locs_210407.gz")


asdf

for loc in mfs_bouton_locs['mf_431']:
    print([int(loc[0]/4), int(loc[1]/4), int(loc[2]/40)])

for loc in mfs_locs['mf_41']:
    print([int(loc[0]/4), int(loc[1]/4), int(loc[2]/40)])
