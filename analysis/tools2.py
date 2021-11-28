import collections
import sys
# import json
import random
# from jsmin import jsmin
# from io import StringIO
# import numpy as np
from collections import defaultdict

def to_ng_coord(coord):
    return (
        int(coord[0]/4),
        int(coord[1]/4),
        int(coord[2]/40),
        )

def compute_mf_share(nids, mf_grc_db):
    count = defaultdict(lambda: defaultdict(list))
    grcs = nids
    for i in range(len(grcs)):
        grc_i = grcs[i]
        if grc_i in mf_grc_db:
            i_set = set(mf_grc_db[grc_i].keys())
            for j in range(len(grcs)):
                if i == j:
                    continue
                grc_j = grcs[j]
                if grc_j in mf_grc_db:
                    j_set = set(mf_grc_db[grcs[j]].keys())
                    common_mfs = i_set & j_set
                    if len(common_mfs):
                        count[grcs[i]][len(common_mfs)].append(grcs[j])
    return count

